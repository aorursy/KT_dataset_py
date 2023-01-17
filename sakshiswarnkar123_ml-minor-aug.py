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
df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df
#check whether it has null values or not
df.info()
import matplotlib.pyplot as plt
plt.hist(df['Pregnancies'])
plt.hist(df['Glucose'])
plt.hist(df['BloodPressure'])
plt.hist(df['SkinThickness'])
plt.hist(df['Insulin'])
plt.hist(df['BMI'])
plt.hist(df['DiabetesPedigreeFunction'])
plt.hist(df['Age'])
plt.hist(df['Outcome'])
X=df['Pregnancies']
Y=df['Age']
plt.scatter(X,Y)
X=df['Glucose']
Y=df['Insulin']
plt.scatter(X,Y)
X=df['Age']
Y=df['BMI']
plt.scatter(X,Y)
X=df['Pregnancies']
Y=df['DiabetesPedigreeFunction']
plt.scatter(X,Y)
X=df['DiabetesPedigreeFunction']
Y=df['Insulin']
plt.scatter(X,Y)
x=df.drop('Outcome',axis=1)
y=df['Outcome']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(leaf_size=1,p=2,n_neighbors=30)
model.fit(x_train,y_train)
model.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
model.score(x_test,y_test)
feature_important = model.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(x_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
pred=model.predict(x_test)
pred