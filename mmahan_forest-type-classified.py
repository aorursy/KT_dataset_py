# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from vecstack import stacking

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso, LogisticRegression







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)  
df_train = pd.read_csv('/kaggle/input/learn-together/train.csv')

df_test = pd.read_csv('/kaggle/input/learn-together/test.csv')
df_train.head()

df_train.info()

df_train['Cover_Type'].nunique()

#number of occurrence of each cover type

df_train['Cover_Type'].value_counts()
df_train['Cover_Type'].value_counts().plot(kind= 'bar')
#Explore Soil Type

df_train['Soil_Type1'].value_counts().plot(kind= 'bar')
#Find location of columns Soil Types

print(df_train.columns.get_loc("Soil_Type1"))

print(df_train.columns.get_loc("Soil_Type40"))



df_soil= df_train.iloc[ :, 15:55]



#Find all the Soil Type columns which have all zeroes

df_soil.columns[(df_soil == 0).all()] # this gives columns which have all values zeroes
#Drop those columns

df_train= df_train.drop(['Soil_Type7', 'Soil_Type15'], axis=1 )# Explore Wilderness Area
df_train['Wilderness_Area4'].value_counts()
df_train[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].apply(pd.Series.value_counts)



df_train[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].apply(pd.Series.value_counts).plot(kind='bar')

#this shows most of forests come under wilderness area 3 or Comanche Peak Wilderness Area



#drop id 

df_train = df_train.drop(['Id'], axis= 1)


cor = df_train.corr()

plt.figure(figsize=(14,14))

sns.heatmap(cor, cbar= True, square = True, cmap= 'coolwarm')

#plt.show()
#Correlation between the target variable Cover_Type and the other features

df_train[df_train.columns[1:]].corr()['Cover_Type'][:]
#Correlation with output variable

cor_target = abs(cor["Cover_Type"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.10]

relevant_features



#We can keep all the features
y= 'Cover_Type'

features = df_train.drop(['Cover_Type'], axis=1)
X= features.columns[:len(features.columns)]
X_train, X_test, y_train, y_test = train_test_split(df_train[X], df_train[y], test_size=.20, random_state =0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
rfc= RandomForestClassifier(n_estimators= 100)
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
rfc_accuracy = accuracy_score(rfc_prediction, y_test)
print('Random Forest with scaling Final prediction score: [%.8f]' % rfc_accuracy)



# Actual versus Predicted Values



new_df_pred_actual = pd.DataFrame({'Actual': y_test, 'Predicted': rfc_prediction})

print(new_df_pred_actual)
#Drop the columns from the test dataset too

df_test_final = df_test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis=1 )

rfc_cover_type_predicted = rfc.predict(df_test_final)
final =pd.DataFrame(df_test['Id'])

final['Cover_Type']=rfc_cover_type_predicted
final.to_csv("rfc_cover_type_predicted.csv", index=False)
final.info()