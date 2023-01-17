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
#read the partial data from files

#loading only part of the data

train_num=pd.read_csv("../input/train_numeric.csv",nrows=100000)

print(train_num.shape)

train_num.head()
train_num=train_num.dropna(axis=1,thresh = int(len(train_data_num)*0.3)) #drop if a feature has NaN more than %

train_num = train_num.fillna(0) #Kepping it simple by filling NaN with 0

print(train_num.shape)

train_num.head() #features reduced
#Check if data is imabalanced

pd.value_counts(train_num["Response"].values, sort=False) #Highly unbalance dataset
print('Proportion:', round(99432/ 568, 2), ': 1')
#A visual of unbalanced data

train_num["Response"].value_counts().plot(kind='bar', title='Count (target)');
#Re-sampling data



# Divide by class

df_class_0 = train_num[train_num["Response"] == 0]

df_class_1 = train_num[train_num["Response"] == 1]
#undersampling

df_class_0_under = df_class_0.sample(40000)



#oversampling

df_class_1_over = df_class_1.sample(60000,replace=True)



df_test = pd.concat([df_class_0_under, df_class_1_over], axis=0)



print('Random under-sampling:')

print(df_test["Response"].value_counts())



df_test["Response"].value_counts().plot(kind='bar', title='Count (target)');

#Remove ID and Response



X=df_test

y=df_test["Response"]

X=X.iloc[:,1:-1] #remove I and Response

print(X.head(),y.head())
#Import `RandomForestClassifier`

#Feature Selection

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

%matplotlib inline

names = X.columns.values

# Build the model

rfc = RandomForestClassifier(random_state=0)

# Fit the model

rfc.fit(X, y)

from sklearn.feature_selection import SelectFromModel

importances = rfc.feature_importances_

indices = np.argsort(importances)





plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices])

plt.xlabel('Relative Importance')



sfm = SelectFromModel(rfc, threshold=0.01)

sfm.fit(X,y)
names = X.columns.values

RF_col = []

for feature_list_index in sfm.get_support(indices=True): 

    #print(names[feature_list_index])

    RF_col.append(names[feature_list_index])

print('\n Columns chosen using Random Forest:\n',RF_col)



X=X[RF_col]

print('\n Shape of data after choosing important features:',X.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

lr = LogisticRegression(penalty='l2',random_state=42, solver='saga')

lr.fit(X_train,y_train)

print('Logistic regression,with features extracted by RF')

print('Score for training data:',lr.score(X_train,y_train))

y_val_pred=lr.predict(X_val)

print('Score for validation data:',lr.score(X_val,y_val))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_val, y_val_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_val, y_val_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_val, lr.predict(X_val))

fpr, tpr, thresholds = roc_curve(y_val, lr.predict_proba(X_val)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
test_num=pd.read_csv("../input/test_numeric.csv",nrows=100000)

X_test=test_num.iloc[:,1:] #remove Id and Response

X_test= X_test.fillna(0) #Kepping it simple by filling NaN with 0

X_test=X_test[RF_col]

X_test.head()





prediction=lr.predict(X_test)



my_submission = pd.DataFrame({'Id': test_num.Id, 'Pass/Fail': prediction})

my_submission.to_csv('submission.csv', index=False)