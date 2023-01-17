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
churn_data = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")

churn_data.head()
churn_data.describe()
import matplotlib.pyplot as plt

%matplotlib inline
churn_data.hist(bins=50,figsize=(20,15))
#There are two string cols Geography and gender. Both are having less cardinality and also they are not in any range to set label encoding.

#Hence going for one hot



feature_cols = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']



churn_data1 = churn_data[feature_cols]



from sklearn.preprocessing import OneHotEncoder

s1 = (churn_data1.dtypes == 'object')

object_cols1 = list(s1[s1].index)

print(object_cols1)



one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)



churn_OH = pd.DataFrame(one_hot.fit_transform(churn_data1[object_cols1]))



churn_OH.index = churn_data1.index





num_churn_OH = churn_data1.drop(object_cols1,axis=1)



OH_churn_data = pd.concat([num_churn_OH,churn_OH],axis=1)



OH_churn_data.head()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(OH_churn_data,OH_churn_data['NumOfProducts']):

    strat_train_set = OH_churn_data.loc[train_index]

    strat_test_set = OH_churn_data.loc[test_index]
corr_matrix = churn_data.corr()

corr_matrix['Exited'].sort_values(ascending=True)
from pandas.plotting import scatter_matrix

attributes = ['Exited','IsActiveMember','HasCrCard','NumOfProducts','CreditScore','Tenure','EstimatedSalary']

scatter_matrix(churn_data[attributes],figsize=(12,8))
y_strat_train = strat_train_set.Exited

strat_train_set.drop(['Exited'],axis=1,inplace=True)

strat_train_set.head()
y_strat_test = strat_test_set.Exited

strat_test_set.drop(['Exited'],axis=1,inplace=True)

strat_test_set.head()
import lightgbm as lgb

clf = lgb.LGBMClassifier(n_estimators=100,learning_rate=0.05,max_depth=7,num_leaves=15)

clf.fit(strat_train_set, y_strat_train)
y_pred=clf.predict(strat_test_set)
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred, y_strat_test)

print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_strat_test, y_pred)))
#Confusion matrix

from sklearn.metrics import confusion_matrix

cmlg = confusion_matrix(y_strat_test,y_pred)

#Accuracy

from sklearn.metrics import accuracy_score

accuracylg=accuracy_score(y_pred,y_strat_test)
print(cmlg)

print(accuracylg)

print("accuracy_classfier = ", (cmlg[0][0] 

                                + cmlg[1][1])*100/2000,"%")
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=5)

my_model.fit(strat_train_set, y_strat_train)
prediction1 = my_model.predict(strat_test_set)

y_strat_test

len(y_strat_test)
from sklearn import metrics



score = metrics.roc_auc_score(y_strat_test, prediction1)



print(f"Test AUC score: {score}")



prediction_c = prediction1

for i in range(0,len(prediction_c)):

    if prediction_c[i]>=.5:       # setting threshold to .5

       prediction_c[i]=1

    else:  

       prediction_c[i]=0



prediction_c
#Confusion matrix



cm = confusion_matrix(y_strat_test,prediction_c)

#Accuracy



accuracy=accuracy_score(prediction_c,y_strat_test)
print(cm)

print(accuracy)

print("accuracy = ", (cm[0][0] + cm[1][1])*100/2000,"%")
from xgboost import XGBClassifier

my_model1 = XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=5)

my_model1.fit(strat_train_set,y_strat_train)

pred_class = my_model1.predict(strat_test_set)

pred_class
score = metrics.roc_auc_score(y_strat_test, pred_class)



print(f"Test AUC score: {score}")
cm1 = confusion_matrix(y_strat_test,pred_class)

#Accuracy



accuracy1=accuracy_score(pred_class,y_strat_test)
print(cm1)

print(accuracy1)

print("accuracy_classfier = ", (cm1[0][0] 

                                + cm1[1][1])*100/2000,"%")