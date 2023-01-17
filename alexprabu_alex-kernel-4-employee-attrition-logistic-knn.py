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
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()
df.tail()
df.shape
df.columns
df.describe().T
# import pandas_profiling

# pandas_profiling.ProfileReport(df)
df.columns
df.Over18.value_counts()
df.EmployeeCount.value_counts()
df.StandardHours.value_counts()
df.drop(columns=['Over18', 'EmployeeCount','StandardHours'], inplace=True)
df.columns
df[['MonthlyIncome','JobLevel']].corr()
df.isna().sum()
df.drop_duplicates(inplace=True)
df.shape
df['Attrition'].replace('Yes',1,inplace=True)
df['Attrition'].replace('No',0,inplace=True)
df['Attrition']
num_cols = df.select_dtypes(include = np.number)
cat_col = df.select_dtypes(exclude=np.number)
encoded_cat_cols = pd.get_dummies(cat_col)

encoded_cat_cols
preprocessed_df = pd.concat([encoded_cat_cols, num_cols], axis=1)
preprocessed_df.head()

x = preprocessed_df.drop(columns='Attrition')
y = preprocessed_df['Attrition']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=12)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x, train_y)
train_predict = model.predict(train_x)

test_predict = model.predict(test_x)
from sklearn import metrics
metrics.confusion_matrix(train_y,train_predict)
metrics.accuracy_score(train_y,train_predict)
metrics.confusion_matrix(test_y,test_predict)
metrics.accuracy_score(test_y,test_predict)
from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
preprocessed_df.head()
train_y = train_y.ravel()

train_y = train_y.ravel()
for K in range(25):

    K_value = K+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(train_x, train_y) 

    predict_y = neigh.predict(test_x)

    print ("Accuracy is ", accuracy_score(test_y,predict_y)*100,"% for K-Value:",K_value)