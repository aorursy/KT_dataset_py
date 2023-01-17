# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
loans = pd.read_csv('../input/loan_borowwer_data.csv')
loans.info()
loans.describe()
loans.head(5)
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Credit.Policy=1')

loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='not.fully.paid=1')

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(11,7))

sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.figure(figsize=(11,7))

sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',

           col='not.fully.paid',palette='Set1')
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(final_data.drop('not.fully.paid',axis=1))
scaled_features = scaler.transform(final_data.drop('not.fully.paid',axis=1))
final_data_scaled = pd.DataFrame(scaled_features,columns=final_data.columns[:-1])

final_data_scaled.head()
from sklearn.model_selection import train_test_split
X = final_data_scaled.drop('not.fully.paid',axis=1)

y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.svm import SVC
model = SVC(C=1,gamma=0.1)

model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [10,1,0.1,0.01], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
# May take awhile!

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))