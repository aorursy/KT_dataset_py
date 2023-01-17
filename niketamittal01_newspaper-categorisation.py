!unzip /content/bbc-fulltext.zip
import pandas as pd
import os
import re
f=[]
for (dirpath, dirnames, filenames) in os.walk('/content/bbc'):
    for filename in filenames:
        f.append(dirpath+'/'+filename)
d=[]
n_id=0
for filepath in f:
    try:
      with open(filepath,'r', encoding='utf-8') as infile:
          text=''.join(infile.readlines())
          d.append({
              'id':n_id,
              'filepath':filepath,
              'text':text,
              'category':filepath.split('/')[3]
          })
          n_id+=1        
    except Exception:
        pass
df=pd.DataFrame(d)
df.head()
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def normalizeText(text):
    output=text.lower()
    output=re.sub('[\s]+',' ', output)
    output=re.sub('[^a-z0-9\ ]','', output)
    return output
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def stopFilterWords(text):
    words=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    new_text=""
    for w in words:
        if w not in stop_words:
            new_text=new_text+" "+w
    return new_text
df.text=df.text.apply(normalizeText)
df.head()
df.text=df.text.apply(stopFilterWords)
df.head()
from nltk import stem
from nltk.tokenize import word_tokenize
def lemmat(text):
    lem=stem.WordNetLemmatizer()
    words=word_tokenize(text)
    new_text=""
    for word in words:
        w=lem.lemmatize(word)
        new_text=new_text+" "+w
    return new_text
df.text=df.text.apply(lemmat)
df.head()
#Calculate TF-IDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
text = df['text']
vect=CountVectorizer(ngram_range=(3,3))
vector_values=vect.fit_transform(text)
data=vect.get_feature_names()
print(np.shape(data))
vect1=TfidfVectorizer(ngram_range=(2,2), use_idf=True)
vector_values1=vect1.fit_transform(text)
data=vect1.get_feature_names()
X=vector_values1.toarray()
print(np.shape(vector_values1))
print(data[0])
print(X[0])
sums=X.sum(axis=0)
print(sums[0])
val=[]
for col, term in enumerate(data): 
    try:
        val.append((term,sums[col]))
    except Exception as e:
        pass
df1=pd.DataFrame(val, columns=['term','ranking'])
df1=df1.sort_values('ranking',ascending=False)
print(df1.head())
#Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
distance = 1-cosine_similarity(vector_values1)
print(distance)
# Commented out IPython magic to ensure Python compatibility.
#Elbow method
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# %matplotlib inline
distort=[]
#vector_values1=vector_values1.reshape(vector_values1.shape[1],2)
r=range(2,10)
for i in r:
  Kcluster=KMeans(n_clusters=i).fit(vector_values1)
  predicts=Kcluster.fit_predict(vector_values1)
  sills=silhouette_score(vector_values1,predicts,metric='euclidean')
  #print(sills)
  distort.append(sills)
plt.plot(r, distort, 'bx-')
plt.xlabel('No of Clusters')
plt.ylabel('Distortions')
plt.show()
vector_values1.ndim
vector_values1[0]
from sklearn.decomposition import TruncatedSVD
# Create a PCA instance: pca
pca = TruncatedSVD(n_components=2)
principalComponents = pca.fit_transform(vector_values1)
# Plot the explained variances
features = range(2)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
PCA_components.head()
pca = TruncatedSVD(n_components=5).fit(vector_values1)
data2D = pca.transform(vector_values1)
print(np.shape(data2D))
print(np.shape(vector_values1))
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data2D[:,0], data2D[:,1], c = 'r')
plt.scatter(data2D[:,1], data2D[:,2], c = 'b')
plt.scatter(data2D[:,2], data2D[:,3], c = 'g')
plt.scatter(data2D[:,3], data2D[:,4], c = 'c')
plt.show()
plt.scatter(PCA_components[0], PCA_components[1], alpha=0.3, color='black')
plt.scatter(PCA_components[1], PCA_components[2], alpha=0.3, color='blue')
plt.scatter(PCA_components[2], PCA_components[3], alpha=0.3, color='green')
plt.scatter(PCA_components[3], PCA_components[4], alpha=0.3, color='red')
plt.scatter(PCA_components[4], PCA_components[5], alpha=0.3, color='magenta')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
from sklearn.cluster import KMeans
len1=int(0.8*df.shape[0])
df1=df.iloc[:len1,:]
kmeans = KMeans(n_clusters=5).fit(vector_values1[:len1])
centers2D = pca.transform(kmeans.cluster_centers_)
#len=range(0,df.shape[0])
len=range(0,len1)
centres=kmeans.labels_.tolist()
for i in len:
  df1['Centre']=centres
  df1['Vector_Value']=np.array(data2D[:len1,:]).tolist()
plt.scatter(centers2D[:,0], centers2D[:,1], c='k',
            marker='x', s=200, linewidths=10)
plt.show()  
# pca = TruncatedSVD(n_components=5).fit(vector_values1)
# data2D = pca.transform(vector_values1)
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data2D[:,0], data2D[:,1], c = 'r')
plt.scatter(data2D[:,1], data2D[:,2], c = 'b')
plt.scatter(data2D[:,2], data2D[:,3], c = 'g')
plt.scatter(data2D[:,3], data2D[:,4], c = 'c')
# plt.hold(True)
plt.scatter(centers2D[:,0], centers2D[:,1], c='k',
            marker='x', s=200, linewidths=10)
plt.show()   
df1.head()
df1['Centre'].value_counts()
df1.shape
from sklearn.externals import joblib
joblib.dump(kmeans, 'model_clust.pkl')
df1.head()
df2=df.iloc[len1:,:]
vect2=vector_values1[len1:]
print(df2.shape)
df2.sample(100)
#saved_model=joblib.load('model_clust.pkl')
#saved_model.predict(vect2)
saved_model=joblib.load('model_clust.pkl')
saved_model.predict(vect2)