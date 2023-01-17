#importing necessary libraries

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm                                                               

from sklearn.ensemble import RandomForestClassifier                                

from sklearn.linear_model import LogisticRegression
#loading files

data=pd.read_csv('../input/category/Books_small.csv')[['reviewText','overall']]

data1=pd.read_csv('../input/category/Clothing_small.csv')[['reviewText','overall']]

data2=pd.read_csv('../input/category/Electronics_small.csv')[['reviewText','overall']]

data3=pd.read_csv('../input/category/grocery.csv')[['reviewText','overall']]

data4=pd.read_csv('../input/category/patiocsv.csv')[['reviewText','overall']]
#creating sentiment column

data['sentiment']=['negative' if i<3 else 'positive' for i in data['overall']]

data1['sentiment']=['negative' if i<3 else 'positive' for i in data['overall']]

data2['sentiment']=['negative' if i<3 else 'positive' for i in data['overall']]

data3['sentiment']=['negative' if i<3 else 'positive' for i in data['overall']]

data4['sentiment']=['negative' if i<3 else 'positive' for i in data['overall']]
#creating category column

data['category']='book'

data1['category']='cloth'

data2['category']='electronics'

data3['category']='grocery'

data4['category']='patios'
# stacking all files

file=[data,data1,data2,data3,data4]

result = pd.concat(file) 

result.shape
#input and output data to be trained

x=result['reviewText']

y=result['category']
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(x)  
#using classifier models

from sklearn import svm                                                                 #linear svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, y)



from sklearn.ensemble import RandomForestClassifier                                    #random forest classifier

cls=RandomForestClassifier()

cls.fit(train_x_vectors,y)



from sklearn.linear_model import LogisticRegression                                     #logistic regression

clf_log = LogisticRegression()

clf_log.fit(train_x_vectors,y)
#prediction

test=['not working', 'loved the necklace', 'bad']

test_x_vectors = vectorizer.transform(test)

clf_log.predict(test_x_vectors)