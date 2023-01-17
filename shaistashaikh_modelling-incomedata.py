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
df=pd.read_csv('/kaggle/input/income/train.csv')
df
df.info()
df_Us_Country=df.loc[df['native-country'] == 'United-States']

df_Us_Country
numeric_features = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
numeric_features
cat_features=[ col for col in list(df.columns) if df[col].dtype =='object']
cat_features
df_Us_Countrydummy=pd.get_dummies(df_Us_Country[cat_features])
df_Us_Countrydummy

df_Us_Countrydummy.shape
final_df_Us_Countrydummy=pd.concat([df_Us_Countrydummy , df_Us_Country[numeric_features],df_Us_Country['income_>50K']], axis = 1)
final_df_Us_Countrydummy.info()
final_df_Us_Countrydummy.isna()
final_df_Us_Countrydummy=final_df_Us_Countrydummy.fillna(0)
final_df_Us_Countrydummy
final_df_Us_Countrydummy.isna().sum().sum()


x=final_df_Us_Countrydummy.drop(['income_>50K'],axis=1)
y=final_df_Us_Countrydummy['income_>50K']
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
# Training using Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier  
classifier1 = DecisionTreeClassifier(criterion='gini')  
classifier1.fit(X_train, y_train) 
y_pred_1 = classifier1.predict(X_test)  
print(y_pred_1)
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc_1 = accuracy_score(y_test,y_pred_1)
print("Accuracy for Gini model {} %".format(acc_1*100))

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_1))
#[TN FP ] 
#[FN TP ]
# 7806  -->  people having less $50 k  - rightly predicted  # True Negative  ( target ==0 )
# 1786 -->  people having more $50 k  - rightly predicted  # True Positive  ( target ==1 )

# 1191 -->  people having less $50 k  - model say - they have more than $50K # False Positive 
# 1043  -->  people having more $50 k  - model say - they have less than $50K # False Negative

accuracy  = (7806 + 1786)/(7806+1043 +1191+1786)
print(accuracy)

# precision  = True Positive / ( True Positive + False Positive)
precision = 1786/(1786 +1191)
print(precision)



# Recall  = True Positive / (True Positive + False Negative)
recall = 1786 /(1786+1043  )
print(recall)