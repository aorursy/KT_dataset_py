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
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

print(data.shape)

data.head()
data.info()
import seaborn as sns

sns.heatmap(data.corr(),annot=True)
data.head(3)
round(data.isnull().sum()/len(data.index),2)
from sklearn.model_selection import train_test_split

X = data.iloc[:,:-1]

y = data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 100,test_size = 0.2,stratify = y)
from xgboost import XGBClassifier

from sklearn.metrics import log_loss

my_model = XGBClassifier(n_estimators = 100)

my_model.fit(X_train,y_train,

            early_stopping_rounds = 5,

            eval_set=[(X_test,y_test)],

            verbose = False)
y_pred1 = my_model.predict(X_test)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred1)



print('The accuracy using XGBoost is: {}'.format(score))
log_loss(y_test, y_pred1, eps=1e-15)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred2 = lr.predict(X_test)

score = accuracy_score(y_test,y_pred2)

print('The accuracy using Logistic Regression is: {}'.format(score))

log_loss(y_test, y_pred2, eps=1e-15)

from sklearn.ensemble import RandomForestClassifier

Rf = RandomForestClassifier()

Rf.fit(X_train,y_train)

y_pred3 = Rf.predict(X_test)

score = accuracy_score(y_test,y_pred3)

print('The accuracy using RandomForest Classifier is: {}'.format(score))

log_loss(y_test, y_pred3, eps=1e-15)

y_pred3
def inp(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):

    x = np.zeros(len(X.columns))

    x[0]=Pregnancies

    x[1]=Glucose

    x[2]=BloodPressure

    x[3]=SkinThickness

    x[4]=Insulin

    x[5]=BMI

    x[6]=DiabetesPedigreeFunction

    x[7]=Age

    out = Rf.predict([x])[0]

    return out

inp(1,85,66,29,0,26.6,0.351,31)
