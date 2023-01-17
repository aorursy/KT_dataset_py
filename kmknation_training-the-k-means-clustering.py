import pandas as pd
import os
import string
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
print('NLTK (DOWNLOAD ALL PACKAGES TO PERFORM NLP OPERATION)')

print('UNCOMMENT FOLLOWING LINE To GET NLTK DOWNLOADED')
# nltk.download('all')
stopword = nltk.corpus.stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

!wget http://7c4292c7e0ca.ngrok.io/data.zip
!unzip data.zip
def preprocess(df):
    
    df = df[df.columns.drop(list(df.filter(regex='^Cat')))]
    df = df[df['Date'] != '27/06/2001']  #removing the date
    df = df[(df['Subject'] != 'RE:') & (df['Subject'] != 'FW:') & (df['Subject'] != 'Re:')]  #removing the max same subjects
    del df['Unnamed: 0']
    return df
stopword.append('re')
stopword.append('fw')
def clean_text(text):
    text_nopunct = "".join([char for char in text.lower() if char not in string.punctuation])
    tokens = re.split('\W+', text_nopunct) #tokenize
    words_without_stopwords = [word for word in tokens if word not in stopword] #remove stopwords from tokens
    return [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words_without_stopwords]
DATA = 'data' #https://data.world/brianray/enron-email-dataset

FILENAMES = [os.path.join(DATA, filename) for filename in os.listdir(DATA)]
df = pd.read_csv('data/enron_05_17_2015_with_labels_v2_100K_chunk_1_of_6.csv')
df = preprocess(df)
df = df[df['Subject'].notna()]
df = df[df['content'].notna()]
df[df['content'].isna()].shape

dfEmail = df[['Subject', 'content']]
dfEmail.head()
content_vecotrizer = pickle.load(open('tfidf_content.pickle', "rb"))
subject_vecotrizer = pickle.load(open('tfidf_subject.pickle', "rb"))
print('Vectorizer Loaded!!')
subject_matrix = subject_vecotrizer.transform(df['Subject'])
dfSubject = pd.DataFrame(subject_matrix.toarray(), columns=subject_vecotrizer.get_feature_names())
dfSubject.head()
content_matrix = content_vecotrizer.transform(df['content'])
type(content_matrix)
import scipy.sparse as sp

a = sp.csr_matrix([[1,2,3],[4,5,6]])
print("a")
print(a.toarray())
print("b")
b = sp.csr_matrix([[7,8,9],[10,11,12]])
print(b.toarray())
print("c")
c = sp.hstack((a,b))  # NOT np.vstack
print(c.toarray())

matrix =  sp.hstack((subject_matrix,content_matrix))
#Use KMeans clustering from scikit-learn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

#Find optimal cluster size by finding sum-of-squared-distances

sosd = []
#Run clustering for sizes 1 to 15 and capture inertia
K = range(5,30)
for k in K:
    km = KMeans(n_clusters=k, n_jobs=-1) #-1 will use all cores of CPU for computation
    km = km.fit(matrix)
    sosd.append(km.inertia_)
    print(str(k) + "processed")
    
print("Sum of squared distances : " ,sosd)


#Plot sosd against number of clusters
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(K, sosd, 'bx-')
plt.xlabel('Cluster count')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal Cluster Size')
plt.show()
#Split data into 9 clusters
kmeans = KMeans(n_clusters=11, n_jobs=-1).fit(matrix)

#get Cluster labels.
clusters= kmeans.labels_
clusters.shape
len(set(clusters))
pickle.dump(kmeans, open("kmeans.pkl", "wb"))
import os
os.chdir(r'../working')
from IPython.display import FileLink
FileLink(r'kmeans.pkl')
dfEmail.shape
dfEmail['cluster'] = clusters
dfEmail.head()
dfEmail['Subject'] = dfEmail['Subject'].apply(lambda x: clean_text(x))
dfEmail['content'] = dfEmail['content'].apply(lambda x: clean_text(x))
pd.set_option('max_colwidth', 800)
dfEmail[dfEmail['cluster'] == 0].head(20)
dfEmail[dfEmail['cluster'] == 1].head(20)
dfEmail[dfEmail['cluster'] == 2].head(20)
dfEmail[dfEmail['cluster'] == 6].head(20)
