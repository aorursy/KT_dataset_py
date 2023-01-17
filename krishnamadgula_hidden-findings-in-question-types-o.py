#All necessary packages imported here
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import gc
import re
import gensim

print(os.listdir("../input"))
df=pd.read_csv('../input/train.csv')
print (df.columns)
print (df.question_text.describe())


# Loading the Google News Word2Vec Model
model = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
tokenizer=RegexpTokenizer(r'\w+')
#Tokenizing Questions
df['tokenized_text']=[tokenizer.tokenize(i) for i in df.question_text]
#Removal of Stop Words from tokenized questions
temp=list(list(map(lambda x:x if x not in stop_words else False,df.tokenized_text[i])) for i in range(len(df)))
df['tokenized_text']=list(list(filter(lambda x: x is not False,temp[i]))for i in range(len(temp)))
del temp

#Mapping all tokenized words to embeddings
wv=[]
def assign_word_embed(word):
    try:
        return model[word]
    except:
        pass
    
print(df['tokenized_text'][0][0])
df['sentence_vectors']=list(list(map(lambda x:assign_word_embed(x),df['tokenized_text'][i])) for i in range(len(df)))
# temp2=list(map(lambda x:assign_word_embed(x),df['tokenized_text'][0][0]))
print (len(df))
#Creating sentence embedding as mean of word embeddings of the words in sentence
from statistics import mean
def mean_comp(lst):
    length=len(lst)
    sum=np.zeros(300)
    for i in lst:
        try:
            sum+=i
        except:
            """When value is None because word 
            wasn't present in the Google News vocabulary"""
            length-=1
    if length==0:
        return np.zeros(300)
    else:
        return (sum/length)
 #Sample Data for Visualization
df['mean_vectors']=[(mean_comp(df['sentence_vectors'][i])) for i in range(len(df))] 

# temp=[(df['mean_vectors'][i]) for i in range(len(df))]
print((df.size))
df=df.drop('question_text',axis=1)
df=df.drop('tokenized_text',axis=1)
print((df.size))


#Dimensionality reduction using PCA and Visualization the two types of questions 3D
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
a=[[1,1,1],[2,2,2],[3,3,3]]
temp= [df['mean_vectors'][i]for i in range(500000)]
print (type(temp[0]))
pca_result = pca.fit_transform(temp)
pca_first_comp=[i[0]for i in pca_result]
pca_second_comp=[i[1]for i in pca_result]
pca_third_comp=[i[2]for i in pca_result]
print(pca.explained_variance_ratio_)
from matplotlib import pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
color=['red'if l == 0 else 'blue' for l in df['target'][0:500000]]
fig=plt.figure()
ax = fig.gca(projection='3d')
ax=plt.scatter(pca_first_comp,pca_second_comp,pca_third_comp,c=color)
plt.show()
#Checking how the data is clustered into discriminative and sexuality based sentences
wv_sex=(model['sex']+model['incest']+model['bestiality']+model['pedophilia'])/4
# wv_racism=
wv_discrimination=(model['race']+model['color']+model['caste'])/3

#Performing Cosine Similarity 
from sklearn.metrics.pairwise import cosine_similarity as cs
from scipy import spatial
result=1-spatial.distance.cosine(df['mean_vectors'][0],wv_sex)
cs_sex=list(map(lambda x:1-spatial.distance.cosine(x,wv_sex),df['mean_vectors']))
# cs_racism=list(map(lambda x:1-spatial.distance.cosine(x,wv_sex),temp2))
cs_discrimination=list(map(lambda x:1-spatial.distance.cosine(x,wv_discrimination),df['mean_vectors']))
# cs_sex=[cs(i.reshape(1,-1),wv_sex.reshape(1,-1)) for i in temp2]
print(cs_sex[0],result)
sincere_questions={}
insincere_questions={}
sincere_questions['discrimination']=[cs_discrimination[i] for i in range(len(cs_discrimination)) if  df['target'][i]==0]
insincere_questions['discrimination']=[cs_discrimination[i] for i in range(len(cs_discrimination)) if df['target'][i]==1]
sincere_questions['sex']=[cs_sex[i] for i in range(len(cs_sex)) if  df['target'][i]==0]
insincere_questions['sex']=[cs_sex[i] for i in range(len(cs_sex)) if df['target'][i]==1]


#Plots to Visualize how the questions are distributed with respect to their distance from sexuality and discrimination 
fig = plt.figure()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
ax1.hist([insincere_questions['discrimination'][0:1200]],bins=100)
ax3.hist([sincere_questions['discrimination'][0:1200]],bins=100)
ax1.set_title("insincere discriminative")
ax2.set_title("insincere sexual")
ax3.set_title("sincere discriminative")
ax4.set_title("sincere sexual")
ax2.hist([insincere_questions['sex'][0:1200]],bins=100)
ax4.hist([sincere_questions['sex'][0:1200]],bins=100)
# print (insincere_questions)
plt.show()