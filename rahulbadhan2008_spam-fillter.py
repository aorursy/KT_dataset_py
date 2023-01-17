import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

'''import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot

init_notebook_mode(connected=True)

cf.go_offline()'''
spam=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')

spam
sns.countplot(spam['v1'])

plt.show()
spam.info()
x=spam['v2']

y=spam['v1']

from nltk.corpus import stopwords

sw=stopwords.words('english')

import string
def txt_cln(x):

    rp=[char for char in x if char not in string.punctuation]

    rp=''.join(rp)

    return[''.join(word) for word in rp.split() if word.lower() not in sw]
z=[]

for i in x:

    text=''.join(txt_cln(i))

    z.append(text)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(z,y,test_size=.30,random_state=0)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)



from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(analyzer=txt_cln).fit(x_train)

print(len(cv.vocabulary_))





X=cv.transform(x_train)

#print(X)

Y=cv.transform(x_test)



#print(Y)

X=X.toarray()

Y=Y.toarray()

X

Y
from sklearn.naive_bayes import BernoulliNB

bn=BernoulliNB()

bn.fit(X,y_train)

py=bn.predict(Y)

py
from sklearn.metrics import accuracy_score

print("Accurecy Score:",accuracy_score(y_test, py))
plt.scatter(y_test,py)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
# spam classification acc to nltk Python notebook using data from SMS Spam Collection Dataset ·