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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import copy
%matplotlib inline
import xgboost        #import the xgboost library
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.describe()
df.head()
#preparing new dataframe by excluding customerID

df_new = df.filter(regex="[^customerID]")

df_new.head()
df_new.dtypes
df_new.info()
df_new.boxplot('tenure','Churn',figsize=(5,6), widths=0.2)  
df_new.isnull().values.sum()          #No missing values in the dataset !
df_new = df_new.copy()
#type(df_new)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df1= df_new.copy()
#df1.info()
#add parameter errors='coerce' for convert bad non numeric values to NaN
df1['TotalCharges']= pd.to_numeric(df['TotalCharges'],errors='coerce')
df_categorical = df1.copy()



df_categorical = df_categorical.drop(['SeniorCitizen','tenure','MonthlyCharges','TotalCharges'],axis = 1)
df_categorical.head()

df_categorical_converted = df_categorical.apply(lb_make.fit_transform)
df_categorical_converted.head()

a= df_new[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']]
df_final = pd.concat([a, df_categorical_converted],sort=False,axis=1)

df_final.head()
X= df_final.loc[:, df_final.columns != 'Churn']
#X.head()
y = df_final['Churn']
#y.head()
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 0)

#X_train
#fitting Xgboost to training set
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train,y_train)
#predict the test set results

y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
cm
#We neet to tune the hyperparameters to get very good result.
classifier.score(X_test,y_test)