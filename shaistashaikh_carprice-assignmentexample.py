# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
df=pd.read_csv('/kaggle/input/carprice-assignment/CarPrice_Assignment.csv')
df
cat_features=[ col for col in list(df.columns) if df[col].dtype =='object']
cat_features
df.head()
df.info()
 
df['price']=df['price'].apply(lambda x: 0 if x <= 10000 else 1)
df
numeric_features =[ col for col in list(df.columns) if df[col].dtype =='float' or  df[col].dtype =='int']
numeric_features
cat_features =[ col for col in list(df.columns) if df[col].dtype =='object']
cat_features
df_cat_features=pd.get_dummies(df[cat_features])
df_cat_features
df[numeric_features]=df[numeric_features].astype(int)
df[numeric_features].info()
final_df=pd.concat([df_cat_features,df[numeric_features]])
final_df
final_df=final_df.replace(np.nan,0)
final_df
final_df=final_df.astype(int)
final_df
final_df.isna().sum().sum()
x=final_df.drop(['price'],axis=1)
y=final_df['price']
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier  
classifier1 = DecisionTreeClassifier(criterion='gini')  
classifier1.fit(X_train, y_train)
pred=Model.predict(X_test)
print(pred.astype(int))
pred=list(pred.astype(int))
print(pred.count(1))
print(pred.count(0))
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 
acc = accuracy_score(y_test,pred)
print("Accuracy for  model {} %".format(acc*100))
print(confusion_matrix(y_test,pred))
print('f1 Score -->' ,f1_score(y_test,pred))
from sklearn.linear_model import LogisticRegression
LRModel = LogisticRegression() 
LRModel.fit(X_train, y_train)
LRpred=LRModel.predict(X_test)
LRpred
acc = accuracy_score(y_test,LRpred)
print("Accuracy for  model {} %".format(acc*100))
print(confusion_matrix(y_test,LRpred))
print('f1 Score -->' ,f1_score(y_test,LRpred))
from sklearn.neighbors import KNeighborsClassifier


Model2 = KNeighborsClassifier(n_neighbors= 7)  
Model2.fit(X_train, y_train)
KNpred=Model2.predict(X_test)
KNpred
acc = accuracy_score(y_test,KNpred)
print("Accuracy for  model {} %".format(acc*100))
print(confusion_matrix(y_test,KNpred))
print('f1 Score -->' ,f1_score(y_test,KNpred))
#Manual Ensemble learning 

ensemble_df = pd.DataFrame()
ensemble_df['Pred1'] = pred
ensemble_df['Pred2'] = LRpred
ensemble_df['Pred3'] = KNpred
ensemble_df['Sum'] = ensemble_df.sum(axis = 1)
ensemble_df['Final'] = ensemble_df['Sum'] > 2 
ensemble_df['Final'] = ensemble_df['Final'].astype(int)
print(ensemble_df.head())

acc = accuracy_score(y_test,ensemble_df['Final'])
print("Accuracy for Emsemble model {} %".format(acc*100))
print(confusion_matrix(y_test,ensemble_df['Final']))
print('f1 Score -->' ,f1_score(y_test,ensemble_df['Final']))

# 61  -->  Car Price less  10 k   - rightly predicted  # True Negative  ( target ==0 )
# 16  -->  Car Price more  10 k  - rightly predicted  # True Positive  ( target ==1 )

# 1   -->  Car Price less 10 k  - model say - Car Price more than 10K # False Positive 
# 4  -->  Car Price more 10 k  - model say - Car Price less than $50K # False Negative