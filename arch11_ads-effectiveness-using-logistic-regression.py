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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing, model_selection, metrics
df = pd.read_csv("/kaggle/input/advertising/advertising.csv",parse_dates=True)

print(df.info())

display(df.head())



#No missing data
df["month"] = pd.to_datetime(df["Timestamp"]).dt.month

df["day"] = pd.to_datetime(df["Timestamp"]).dt.day



df.drop("Timestamp",axis=1,inplace=True)

sns.set_style('whitegrid')

sns.distplot(df["Age"],kde = False,bins=40)
sns.jointplot(x="Area Income",y="Age",data=df)
sns.jointplot(x="Area Income",y="Age",data=df,kind="kde",color="red")
sns.jointplot(x="Daily Time Spent on Site", y="Daily Internet Usage",data=df)
sns.heatmap(df.corr(),annot=True)
display(df.head())

display(df.columns)
#Creating dummy variable for country



country = pd.get_dummies(df["Country"],drop_first=True)



features = ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male','month', 'day']



X = df[features]

X=pd.concat([X,country],axis=1)

X.head()
y = df['Clicked on Ad']
from sklearn.model_selection import cross_val_score



X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=11,test_size=0.2)





log_reg = LogisticRegression(max_iter=1000,penalty="l2")   #using penalty due to potential uselessness of some features

log_reg.fit(X_train,y_train)

print("Accuracy: " + str(log_reg.score(X_test,y_test)))



scores = cross_val_score(log_reg, X, y, cv=5,scoring="accuracy")



scores.mean()
y_predict = log_reg.predict(X_test)
print(metrics.classification_report(y_test,y_predict))
import sklearn

sklearn.metrics.SCORERS.keys()