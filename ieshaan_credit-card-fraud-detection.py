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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')

df
df.isna().sum()
df.shape
df.describe()
plt.figure(figsize=(20,30))

sns.heatmap(df.corr(),annot=True,linewidths=1.0)

plt.show()
df.drop("Class", axis=1).apply(lambda x: x.corr(df.Class))
df.hist(figsize=(20,20),edgecolor="k")

plt.tight_layout()

plt.show()
X=df.iloc[:,1:]
X=df.iloc[:,:-1]

Y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=177)
sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
model = LogisticRegression()



# fit the model with the training data

model.fit(X_train,Y_train)



# coefficeints of the trained model

print('Coefficient of model :', model.coef_)



# intercept of the model

print('Intercept of model',model.intercept_)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = DecisionTreeClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# depth of the decision tree

print('Depth of the Decision Tree :', model.get_depth())



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)

model = RandomForestClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = GradientBoostingClassifier(n_estimators=100,max_depth=5)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
model = XGBClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)