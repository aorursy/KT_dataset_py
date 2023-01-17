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
## import the necessary libraries

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
## load in the data

data = pd.read_csv("../input/diabetes/diabetes.csv")
## check the data

data.head()
## check the info of the data

data.info()
## check for missing values

data.isnull().sum()
## check the columns 

data.columns
## check the shape of the dataset

data.shape
## descriptive analysis

data.describe()
## relationship between features

sns.pairplot(data)
## correlation

data.corr()
## heatmap correlation

plt.figure(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, cmap="RdYlGn")
## age

sns.distplot(data["Age"])
## Replace zeros in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Pregnancies'] with mean

data["Glucose"] = data["Glucose"].replace(0,data["Glucose"].mean())

data["BloodPressure"] = data["BloodPressure"].replace(0,data["BloodPressure"].mean())

data["SkinThickness"] = data["SkinThickness"].replace(0,data["SkinThickness"].mean())

data["Insulin"] = data["Insulin"].replace(0,data["Insulin"].mean())

data["BMI"] = data["BMI"].replace(0,data["BMI"].mean())

data["Pregnancies"] = data["Pregnancies"].replace(0,data["Pregnancies"].mean())
## check descriiptive analysis after filling missing value and compare with the previous one

data.describe()
## select the independent and dependent features 

X = data.drop(columns="Outcome") #you can also use this method - data = data.drop(["Outcome"], axis=1)

y = data["Outcome"]
## split the data into training and test set 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
##check the shape of the trianing set 

X_train.shape
## shape of the test size

X_test.shape
## Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
## modelling and predicting 



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



y_prediction = random_forest.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(classification_report(y_test, y_prediction))
## confusion matrix

cm = confusion_matrix(y_test, y_prediction)

cm
## plot confusion matrix

sns.heatmap(cm, annot=True, cmap="mako")