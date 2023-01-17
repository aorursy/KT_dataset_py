# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/paribas-claim-feature-selection/train.csv',nrows=50000)

df.head()
import pandas as pd 

import numpy as np 



import matplotlib.pyplot as plt 

import seaborn as sns 



from sklearn.model_selection import train_test_split



from sklearn.linear_model import Lasso,LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler
# Creating a copy of the data 

df_copy = df.copy()

df.shape
# In practise feature selection should be done after pre processing the data categorical data should be encoded and then only we need to access how deterministic they are of the target

# Here we will be considering Numerical Variables 

# Selecting the numerical columns with the below lines of code 



numerics = ['int16','int32','int64','float16','float32','float64']

numerical_vars = list(df.select_dtypes(include = numerics).columns)

df = df[numerical_vars]

df.shape
# seperate train and test sets 

X_train,X_test,y_train,y_test = train_test_split(df.drop(labels=['target','ID'],axis=1),df['target'],test_size=0.3,random_state=0)

X_train.shape,X_test.shape
# Linear Model Benifits with feature Scaling



scaler = StandardScaler()

scaler.fit(X_train.fillna(0))
# We will be doing model fitting and feature selection together 

# We will specify Logistic regression and select Lasso (L1) penalty 

# SelectFromModel from sklearn will be used to slect the features for which coefficients are non-zero



#sel_ = SelectFromModel(LogisticRegression(C=1, penalty= 'l1'))



sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))

sel_.fit(scaler.transform(X_train.fillna(0)), y_train)



# I used penalty as none as l1 option was not working
# this command lets us to vizualise which features were kept

sel_.get_support()
# We can now make a list of selected features 

selected_feat = X_train.columns[(sel_.get_support())]



print('Total features:{}'.format((X_train.shape[1])))

print('Selected features:{}'.format(len(selected_feat)))

print('Features with coefficients shrank to zero:{}'.format(np.sum(sel_.estimator_.coef_==0)))
# The number of features which coefficient was shrank to zero 

np.sum(sel_.estimator_.coef_==0)
# Identifying the removed features 

removed_feats =X_train.columns[(sel_.estimator_.coef_==0).ravel().tolist()]

removed_feats

selected_feat
df.shape
# We can now remove the features from training and test set 

X_train_selected = sel_.transform(X_train.fillna(0))

X_test_selected = sel_.transform(X_test.fillna(0))

X_train_selected.shape,X_test_selected.shape
"""from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train_selected,y_train)"""

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train_selected,y_train)
#y_test=lm.predict(X_test_selected)

y_pred=classifier.predict(X_test_selected)

print(y_test)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print('Accuracy of model is:',accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
print(classification_report(y_test,y_pred))
df1=df_copy

df1.shape
# In practise feature selection should be done after pre processing the data categorical data should be encoded and then only we need to access how deterministic they are of the target

# Here we will be considering Numerical Variables 

# Selecting the numerical columns with the below lines of code 



numerics = ['int16','int32','int64','float16','float32','float64']

numerical_vars = list(df1.select_dtypes(include = numerics).columns)

df1 = df1[numerical_vars]

df1.shape
df.shape
# seperate train and test sets 

X_train,X_test,y_train,y_test = train_test_split(df1.drop(labels=['target','ID'],axis=1),df1['target'],test_size=0.3,random_state=0)

X_train.shape,X_test.shape
# Linear Model Benifits with feature Scaling



scaler = StandardScaler()

scaler.fit(X_train.fillna(0))
sel_ = SelectFromModel(LogisticRegression(C=1000,penalty='l2'))

sel_.fit(scaler.transform(X_train.fillna(0)), y_train)
# We will be doing model fitting and feature selection together 

# We will specify Logistic regression and select Lasso (L1) penalty 

# SelectFromModel from sklearn will be used to slect the features for which coefficients are non-zero



#sel_ = SelectFromModel(LogisticRegression(C=1, penalty= 'l1'))



sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))

sel_.fit(scaler.transform(X_train.fillna(0)), y_train)



# I used penalty as none as l1 option was not working
sel_.get_support()
selected_feat = X_train.columns[(sel_.get_support())]

len(selected_feat)
np.sum(sel_.estimator_.coef_==0)
sel_.estimator_.coef_.mean()
pd.Series(sel_.estimator_.coef_.ravel()).hist();
np.abs(sel_.estimator_.coef_).mean()
pd.Series(np.abs(sel_.estimator_.coef_).ravel()).hist();
# Comparing the number of selected features with the coefficients who have value above the mean of thw absoulte value of the coefficents

print('Total features: {}'.format((X_train.shape[1])))

print('Selected features: {}'.format(len(selected_feat)))

print('Features with coefficients greater than the mean coefficient:{}'.format(np.sum(np.abs(sel_.estimator_.coef_)>np.abs(sel_.estimator_.coef_).mean())))

