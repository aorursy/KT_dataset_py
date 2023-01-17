# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv')

data.head()
data.info()
data.describe()
sns.heatmap(data.isnull(),cmap='Blues')
data['Sentiment'].value_counts()
sns.countplot(x=data['Sentiment'])
data.groupby('Sentiment').describe()
data['Length']=data['Text'].apply(lambda x:len(x))
data['Length'].plot.hist(bins=200)
data['Length'].describe()
plt.figure(figsize=(12,5))

data.hist(column='Length',by='Sentiment',bins=150)
import string

from nltk.corpus import stopwords
def clean(text):

    a=[f for f in text if f not in string.punctuation]

    a=''.join(a)

    b=[w for w in a.split() if w.lower() not in stopwords.words('english')]

    return b
check=data['Text'].head(1).apply(clean)
print(check[0])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
words=CountVectorizer(analyzer=clean).fit(data['Text']) # Cleaning all our data set from punctuations and stopwords
print(len(words.vocabulary_))
sample=data['Text'][1]

sample
trans=words.transform([sample])

print(trans)
print(trans.shape) 
words.get_feature_names()[363]
allmessgae=words.transform(data['Text'])
print(allmessgae.shape)
allmessgae.nnz # No of non - zero's value
sparsity = (100.0 * allmessgae.nnz / (allmessgae.shape[0] * allmessgae.shape[1]))

print('sparsity: {}'.format(sparsity))
tf=TfidfTransformer()

tf.fit(allmessgae)
tfidf=tf.transform(trans)

print(tfidf) 
tf.idf_[words.vocabulary_['return']] # Checking the IDF value of particular word how imp a term is in whole dataset 
final_transfrom=tf.transform(allmessgae)
modelfitting=MultinomialNB().fit(final_transfrom,data['Sentiment'])
result=modelfitting.predict(final_transfrom)
print(result)
pipe=Pipeline([

 ('cv',CountVectorizer(analyzer=clean)),

 ('tfidf',TfidfTransformer()),

 ('Classifier',MultinomialNB())

])
x_train,x_test,y_train,y_test=train_test_split(data['Text'],data['Sentiment'],test_size=0.3,random_state=101)
pipe.fit(x_train,y_train)
pipe_predict=pipe.predict(x_test)

print(classification_report(pipe_predict,y_test))
print(confusion_matrix(pipe_predict,y_test))
print(accuracy_score(pipe_predict,y_test))