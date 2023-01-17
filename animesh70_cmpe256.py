import os
import pandas
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()

#Querying languages table
QUERY = """
        SELECT repo_name, language
        FROM `bigquery-public-data.github_repos.languages`
        LIMIT 300
        """

query_job = client.query(QUERY)

#Get results into the dataframe
languageData = query_job.to_dataframe()

#filtering repos which include only a single language
iterator = query_job.result(timeout=30)
rows = list(iterator)
rows = list(filter(lambda row: len(row.language)>1,rows))

#Printing first ten repositories
for i in range(10):
    print('Repository '+str(i+1))
    for j in rows[i].language:
        print(j[u'name']+': '+str(j[u'bytes'])+' bytes')
    print('')
print('...')
print(str(len(rows))+' repositories')
languageData.head()
#create dictionary of language names to matrix columns
names = {}
for i in range(len(rows)):
    for j in rows[i].language:
        if j[u'name'] in names:
            names[j[u'name']]+=1
        else:
            names[j[u'name']]=1

#filter out languages that only occur once
names = [n for n in names if names[n]>1]
# for i in range(10):
#     print(names[i])
# print('...')

#print some languages
name_to_index = {}
for j,i in enumerate(names):
    name_to_index[i] = j
print(str(len(names))+" languages")
from math import log

#create matrix
global mat
mat = np.zeros((len(rows),len(names)))
for i,row in enumerate(rows):
    #total = sum([log(lang[u'bytes']+1) for lang in row[1]])
    for lang in row.language:
        if lang[u'name'] in name_to_index and lang[u'bytes'] > 0:
            mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes'])
            #mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes']+1)/total
mat = mat[~np.all(mat==0,axis=1)]
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

%matplotlib inline

#compute PCA
n_components = min(50,len(names))
pca = PCA(n_components=n_components)
transformed = pca.fit_transform(mat) 

#display result
evr = [1-sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]
plt.plot(range(1,n_components+1),evr)
filter_size = min(100,len(mat[0]))
mat = mat[:,range(filter_size)] if len(mat[0])>filter_size else mat #for speed


#This function gives us the sign function implementation for where the target Y is achieved
def init_mask(Y):
    f = np.vectorize(lambda x: 1 if x>0 else 0)
    return f(Y),len(Y),len(Y[0])


#This function implements the regularization parameter to minimize the overfitting
def loss(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    g = np.vectorize(lambda x: x*x)
    return 0.5*np.sum(np.multiply(g(np.subtract(np.matmul(theta,np.transpose(X)),Y)),mask))+reg_param/2*np.sum(g(args))


#This function implements gradient calculation in vectorized way
def gradient(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    X_grad = np.matmul(np.transpose(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask)),theta)+reg_param*X
    theta_grad = np.matmul(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask),X)+reg_param*theta
    return np.concatenate((np.reshape(theta_grad,-1),np.reshape(X_grad,-1)))
import scipy.optimize as op

def train(Y,mask,n_repos,n_langs,n_features=10,reg_param=0.000001):
    #reshape into 1D format preferred by fmin_cg
    theta = np.random.rand(n_repos,n_features)
    X = np.random.rand(n_langs,n_features)
    args = np.concatenate((np.reshape(theta,-1),np.reshape(X,-1)))

    #use fmin_cg to perform gradient descent
    args = op.fmin_cg(lambda x: loss(x,Y,mask,n_repos,n_langs,n_features,reg_param),args,lambda x: gradient(x,Y,mask,n_repos,n_langs,n_features,reg_param))

    #reshape into a usable format
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    
    return theta,X
def recommend(string,Y):
    #process input
    print('Training...')
    langs = string.split(' ')
    lc_names = {str(name).lower(): name_to_index[name] for name in name_to_index}

    #create extra row to append to Y matrix
    test = np.zeros((1,len(names)))
    known = set()
    for lang in langs:
        if lang.lower() in lc_names:
            test[0][lc_names[lang.lower()]] = 1
            known.add(lc_names[lang.lower()])

    #training
    Y = np.concatenate((Y,test[:,range(filter_size)]),0)
    mask,n_repos,n_langs = init_mask(Y)
    theta,X = train(Y,mask,n_repos,n_langs)
    Y = Y[:-1]
    
    #plot features
    for i in range(np.shape(X)[1]):
        col = sorted([(X[j,i],j) for j in range(n_langs)],reverse=True)
        #print('')
        #for k in range(10):
            #print(names[col[k][1]])

    #find top predictions
    predictions = np.matmul(theta,np.transpose(X))[-1].tolist()
    predictions = sorted([(abs(j),i) for i,j in enumerate(predictions)],reverse=True)

    #print predictions
    predictedLang = []
    i = 0
    for val,name in predictions:
        if name not in known:
#             print(str(i+1)+': '+names[name]+' - '+str(val))
            predictedLang.append(names[name])
            i+=1
        if i>=3:
            break
    return predictedLang
languagesList = recommend('Java',mat)
languagesList