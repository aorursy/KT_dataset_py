# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
data=pd.read_csv('/kaggle/input/final.csv')
data
data['class']='2'
data
posts = pd.read_csv('/kaggle/input/final.csv', usecols = ['clean_tweet'])
posts
corpus=[]



for i,row in posts.iterrows():

    row=[row.clean_tweet]

    corpus.append(row)
corpus
c=[]

for x in corpus:

    for y in x:

        c.append(y)
c
from sklearn.feature_extraction.text import CountVectorizer







vectorizer = CountVectorizer()

X = vectorizer.fit_transform(c)
from sklearn.feature_extraction.text import TfidfTransformer



transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(X)

print(tfidf.shape)
tfidf
from sklearn.cluster import KMeans



num_clusters = 5 #Change it according to your data.

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf)

clusters = km.labels_.tolist()
idea={'Post':corpus, 'Cluster':clusters} 

frame=pd.DataFrame(idea,index=[clusters], columns=['Post','Cluster']) 
print("\n")

print(frame) #Print the doc with the labeled cluster number.

print("\n")

print(frame['Cluster'].value_counts())
frame.loc[frame['Cluster'] == 0, 'Cluster'] = 'race'

frame.loc[frame['Cluster'] == 1, 'Cluster'] = 'religion'

frame.loc[frame['Cluster'] == 2, 'Cluster'] = 'Sex'

frame.loc[frame['Cluster'] == 3, 'Cluster'] = 'xxxxx'

frame.loc[frame['Cluster'] == 4, 'Cluster'] = 'yyyyy'
frame
cc=[]



for i,row in frame.iterrows():

    if row.Cluster=='race':

        

        cluster1=[row.Cluster]

        cc.append(cluster1)

        

cd=[]



for i,row in frame.iterrows():

    if row.Cluster=='religion':

        

        cluster2=[row.Cluster]

        cd.append(cluster2)

        

ce=[]



for i,row in frame.iterrows():

    if row.Cluster=='Sex':

        

        cluster3=[row.Cluster]

        ce.append(cluster3)

        

cf=[]



for i,row in frame.iterrows():

    if row.Cluster=='xxxxx':

        

        cluster4=[row.Cluster]

        cf.append(cluster4)

        

        

cg=[]



for i,row in frame.iterrows():

    if row.Cluster=='yyyyy':

        

        cluster5=[row.Cluster]

        cg.append(cluster5)
cc

cd
ce
cf
cg