import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes  import BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report
data = pd.read_csv(r'/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

data.head()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

data.columns = ['label','messages']
data.head()
data.isnull().sum()
data.groupby('label').describe()
vect = TfidfVectorizer()

x = vect.fit_transform(data['messages'].tolist())

print(len(vect.get_feature_names()))

print(vect.get_feature_names()[1000:1010])

print(x.toarray()[:5])

x = x.toarray()
y = data['label'].tolist()

y = np.array(y)
print("Output Variable :",y.shape)

print("Input Variable :",x.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=10,test_size=0.3)
model = BernoulliNB()

model.fit(x_train,y_train)

prediction = model.predict(x_test)

print(accuracy_score(y_test,prediction))

print(classification_report(y_test,prediction))
model = BernoulliNB(alpha=0.74)

model.fit(x_train,y_train)

prediction = model.predict(x_test)

print(accuracy_score(y_test,prediction))

print(classification_report(y_test,prediction))
print(pd.crosstab(y_test,prediction))
sms = ["Congrats You have won a price in the xyz competion, click on the link to claim it  www.abc.advertise.com/ "]

spam = vect.transform(sms)

print(model.predict(spam))