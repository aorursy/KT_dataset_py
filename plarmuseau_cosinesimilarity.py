# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.sparse import coo_matrix, vstack
import pandas as pd
import json
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np # linear algebra
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

def cosine(plays):
    normalized = normalize(plays)
    return normalized.dot(normalized.T)


def bhattacharya(plays):
    plays.data = np.sqrt(plays.data)
    return cosine(plays)


def ochiai(plays):
    plays = csr_matrix(plays)
    plays.data = np.ones(len(plays.data))
    return cosine(plays)


def bm25_weight(data, K1=1.2, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document (artist)
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret


def bm25(plays):
    plays = bm25_weight(plays)
    return plays.dot(plays.T)

def get_largest(row, N=10):
    if N >= row.nnz:
        best = zip(row.data, row.indices)
    else:
        ind = np.argpartition(row.data, -N)[-N:]
        best = zip(row.data[ind], row.indices[ind])
    return sorted(best, reverse=True)


def calculate_similar_artists(similarity, artists, artistid):
    neighbours = similarity[artistid]
    top = get_largest(neighbours)
    return [(artists[other], score, i) for i, (score, other) in enumerate(top)]



def tf_svd_vec(data,descr,k,ngram):
    # attention: an 100.000 rows database returns a 100kx100k =10G matrix
    #from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse.linalg import svds
    
    vec = CountVectorizer(ngram_range=(1,ngram),strip_accents='ascii',min_df=0.00125 )  #,token_pattern="[a-zA-Z]*")
    # term frequency data train, data.name
    tf= vec.fit_transform(data[descr].fillna(''))
    print('term Frequency',tf.shape)
    #print('words',vec.get_feature_names())
    #svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    Ur, Si, VTr = svds(bm25_weight(coo_matrix(tf)), k=k)
    #Ur=svd.fit_transform(bm25(coo_matrix(tf)))  #all words
    #print(svd.explained_variance_ratio_)  
    #print('explained variance',svd.explained_variance_ratio_.sum())
    #reduced vectorspace to k features
    #Xrdf=pd.DataFrame(cosine_similarity(Xr,Xr[:100]))
    #print('reduced term freq',Xrdf[:5])
    print('Ur,Si,VTr shape',Ur.shape,Si.shape,VTr.shape)
    VTr=pd.DataFrame(VTr,columns=vec.get_feature_names())
    Ur=pd.DataFrame(Ur,index=data.index)
    return Ur,VTr

def OptSim(trainm,veld1,veld2,k,ngram):
    from scipy.sparse.linalg import svds
    Ux,Vx=tf_svd_vec(trainm,veld1,k,ngram)
    Uy,Vy=tf_svd_vec(trainm,veld2,k,ngram)
    comwords=list(set(Vx.columns).intersection(Vy.columns))
    comwords=[x for x in comwords if len(x)>2]
    print('nr common words to align',len(comwords))    
    Uo,So,VTo=svds( Vy[comwords].dot(Vx[comwords].T),k=int(k*.8) )
    #
    print('matrix alignment',Uo.shape,VTo.shape )
    Osim=np.matmul(Uo,VTo)

    print('Osim',Osim.shape)
    return Vx,pd.DataFrame(VTo.dot(Vx),columns=Vx.columns),Vy,pd.DataFrame(Uo.T.dot(Vy),columns=Vy.columns) 

#Ux,Vx=tf_svd_vec(train,'naam',500)
#Ux,Vx=tf_svd_vec(para,'klasnaam',100)
from sklearn.metrics.pairwise import cosine_similarity

import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
# Dataset Preparation
print ("Read Dataset ... ")
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
print(train.head())
print(test.head())
train[['id','cuisine']].groupby('cuisine').count().plot(kind='bar')

# Text Data Features

print ("Prepare text data of Train and Test ... ")
train_text = train['ingredients'].map(" ".join)
test_text = test['ingredients'].map(" ".join)
train_text.columns=['ingredients']
pd.DataFrame(train_text)
test.shape
def zoekkeywords(padf,groepb,groeptxt,ngram):
    para3d=padf.groupby(groepb).apply(lambda x : x[groeptxt].values)
    para3d=pd.DataFrame(para3d,columns=[groeptxt])
    for xi in range(0,len(para3d)):
        tottxt=''
        for yi in range(0,len(para3d.iloc[xi][groeptxt])):
            tottxt=tottxt+' '+str(para3d.iloc[xi][groeptxt][yi])
    
        para3d.iloc[xi][groeptxt]=cleanhtml(tottxt)
    
    print(para3d)
    Ux2,Vx2=tf_svd_vec(para3d,groeptxt,5,ngram)  #15 0.63
    from sklearn.metrics.pairwise import cosine_similarity

    UVklasse=pd.DataFrame(cosine_similarity(Ux2,Vx2.T),index=Ux2.index,columns=Vx2.columns)
    print(UVklasse.head())
    autoklas=pd.DataFrame([])
    autoklas['klasse']=UVklasse.index
    autoklas['woorden']='woorden'
    for xi in range(0,len(UVklasse)):
        klassind=UVklasse.index[xi]
        #print(klassind)
        tempo=pd.DataFrame(UVklasse.iloc[xi])
        laatste=tempo.sort_values(by=klassind)[-10:]
        #print(laatste)
        text=laatste.index.values
        tempo=tempo.sort_values(by=klassind,ascending=False)[:2900]
        tempo=tempo[tempo[klassind]>tempo[klassind].max()*0.31]
        #print(tempo)
        autoklas.iat[xi,1]=",".join(tempo.index.values)

    return autoklas,UVklasse

klassekeywords=pd.DataFrame([])
klassekeywords,UVsimilariteit=zoekkeywords(train,'cuisine','ingredients',1)
klassekeywords.columns=['cuisine','ingredients']
#klassekeywords.to_csv('d:klas_tagsb.csv',index=False)

#klassekeywords=zoekkeywords(para,'klasse','titel',2)
#klassekeywords['default']="Default group"
#klassekeywords.to_csv('d:klas_tagsc.csv',index=False)
klassekeywords.ingredients
TRu_,TRvh_=tf_svd_vec(pd.DataFrame((train_text.append(test_text)).append(klassekeywords.ingredients)) ,'ingredients',18,1)
TRcos=cosine_similarity( TRu_[:len(train)+len(test)],TRu_[len(train)+len(test):] ) 
TRcos=pd.DataFrame(TRcos,columns=klassekeywords.cuisine)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class = 'ovr')
#model.fit(TRcos[:len(train)], train.cuisine)
model.fit(cosine_similarity(TRu_[:len(train)],TRvh_.T) ,train.cuisine)

#(model.predict(TRcos[:len(train)])==train.cuisine).mean()
from lightgbm import LGBMClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report


params = {
    'multi_class': 'ovr',
    'solver': 'lbfgs'
}

lgbm_params = {
    'n_estimators': 250,
    'max_depth': 22,
    'learning_rate': 0.2,
    'objective': 'multiclass',
    'n_jobs': 7
}

model = LGBMClassifier(**lgbm_params)
# model = LogisticRegression(**params)
model.fit(TRcos[:len(train)], train.cuisine)
print(model)


y_true =train.cuisine
target_names = model.classes_
y_pred = model.predict(TRcos[:len(train)])
print(classification_report(y_true, y_pred, target_names=target_names))


submission = model.predict(TRcos[len(train):len(train)+len(test)])
submission_df = pd.Series(submission, index=test.index).rename('cuisine')
submission_df.to_csv("logistic_sub.csv", index=True, header=True)
print(submission_df.head())
submission = model.predict(TRcos[len(train):len(train)+len(test)])

submission_df=pd.DataFrame(submission)
submission_df['id']=test.id
submission_df.columns=['cuisine','id']
submission_df.shape
submission_df.to_csv("logistic_sub.csv", index=False, header=True)