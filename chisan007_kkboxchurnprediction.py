# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sb

import matplotlib.pyplot as plt

import sklearn



from pandas import Series, DataFrame

from pylab import rcParams

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics 

from sklearn.metrics import classification_report



%matplotlib inline

rcParams['figure.figsize'] = 10, 8

sb.set_style('whitegrid')
train = pd.read_csv('../input/train.csv')

members = pd.read_csv('../input/members_v2.csv')



print(train.shape)

print(members.shape)
Data = pd.merge(train,members,on='msno', how='inner')

print(Data.shape)
sb.countplot(x='is_churn',data=Data, palette='hls')
Data.isnull().sum()
Data.info()
Data1 = Data.drop(['gender'],1)

#Data1 = Data.replace(r'\s+', np.nan, regex=True)
print(Data1)
#Data1.dropna(inplace=True)

Data1.isnull().sum()
print(Data1)
sb.heatmap(Data1.corr())
#Data2 = Data1.drop(['msno'],1)

Data2 = Data1.msno

print(Data2)
Data3 = pd.DataFrame(Data2)

ID = []

for i in range(1,len(Data3)+1):

    #print(i)

    ID.append(i)

#print(len(Data3))
Data4 = pd.DataFrame(ID)

Data4.columns = ['ID']

print(Data4)
IdData = pd.concat([Data3,Data4],axis=1)
print(IdData)
FinalData = pd.merge(Data1,IdData, on='msno',how='inner')
print(FinalData)
X = FinalData.ix[:,(2,3,4,5,6)].values

y = FinalData.ix[:,1].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
print(len(X_train))

print(len(X_test))
print(len(y_train))

print(len(y_test))
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
results = clf.predict(X_test)

print(results[0:1000])
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, results)

confusion_matrix
from sklearn.metrics import mean_squared_error

mean_squared_error(results, y_test)
#from sklearn.ensemble import RandomForestRegressor

#model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
#model.fit(X,y)
#predictions = model.predict(X_test)
#print(predictions[1:1000])
#mean_squared_error(predictions, y_test)
Submit = FinalData.ix[:,(2,3,46)].values
Results = clf.predict(Submit)
submission = pd.DataFrame(Results)

submission.columns = ['is_Churn_Prediction']

print(submission)
Final_Submission = pd.concat([FinalData,submission],axis=1)

print(Final_Submission)
Final_Submission.to_csv('D:/Kaggle/data/churn_comp_refresh/submission.csv',index=False)