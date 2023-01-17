# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# coding: utf-8



# In[21]:



import numpy as np  

import pandas as pd  

import matplotlib.pyplot as plt  

from scipy.io import loadmat  

get_ipython().magic('matplotlib inline')





# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df.head()
from sklearn.model_selection import train_test_split



data, data2 = train_test_split(df, test_size = 0.25)
data = data.drop('Name', 1)

data = data.drop('Ticket', 1)

data=data.drop('Cabin',1)

data=data.drop('PassengerId',1)

data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})

data['Embarked'] = data['Embarked'].map({'S': 2, 'C': 1,'Q':3})
from sklearn import svm  

#raw_data = loadmat('ex6data2.mat')



#data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  

#data['y'] = raw_data['y']

data.mean()



data=data.apply(lambda x: x.fillna(x.mean()),axis=0)



svc = svm.SVC(C=100, gamma=10, probability=True)  

svc.fit(data[['Pclass','Sex','Age','SibSp','Fare','Embarked']], data['Survived'])  



svc.score(data[['Pclass','Sex','Age','SibSp','Fare','Embarked']], data['Survived'])





# In[34]:

data2 = data2.drop('Name', 1)

data2 = data2.drop('Ticket', 1)

data2=data2.drop('Cabin',1)



data2['Sex'] = data2['Sex'].map({'female': 1, 'male': 0})

data2['Embarked'] = data2['Embarked'].map({'S': 2, 'C': 1,'Q':3})





data2=data2.apply(lambda x: x.fillna(x.mean()),axis=0)



svc.score(data2[['Pclass','Sex','Age','SibSp','Fare','Embarked']], data2['Survived'])
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  

gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]



best_score = 0  

best_params = {'C': None, 'gamma': None}



for C in C_values:  

    for gamma in gamma_values:

        

        svc3 = svm.LinearSVC(C=C, loss='hinge', max_iter=1000)  

        

        svc3.fit(data[['Pclass','Sex','Age','SibSp','Fare','Embarked']], data['Survived'])  

        score=svc3.score(data2[['Pclass','Sex','Age','SibSp','Fare','Embarked']], data2['Survived'])



        if score > best_score:

            best_score = score

            best_params['C'] = C

            best_params['gamma'] = gamma

            predict=svc3.predict(data2[['Pclass','Sex','Age','SibSp','Fare','Embarked']])

            best_svc=svc3

best_score,best_params

best_svc
data3 = pd.read_csv("../input/test.csv")



data3 = data3.drop('Name', 1)

data3 = data3.drop('Ticket', 1)

data3=data3.drop('Cabin',1)







data3['Sex'] = data3['Sex'].map({'female': 1, 'male': 0})

data3['Embarked'] = data3['Embarked'].map({'S': 2, 'C': 1,'Q':3})

data3=data3.apply(lambda x: x.fillna(x.mean()),axis=0)



data3
p=best_svc.predict(data3[['Pclass','Sex','Age','SibSp','Fare','Embarked']])

data3['Survived']=p

data4=data3[['PassengerId','Survived']]

data4.set_index('PassengerId', inplace=True)

data4
data4.to_csv("svc3.csv")