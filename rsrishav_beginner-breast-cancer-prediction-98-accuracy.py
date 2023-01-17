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

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()

data.info()
data = data.drop(["Unnamed: 32", "id"], axis=1)

data.head()
encoder = LabelEncoder()

data["diagnosis"] = encoder.fit_transform(data["diagnosis"])

data.head()
X=data.drop('diagnosis',axis=1)

y=data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 10)
print("X train shape: ", X_train.shape)

print("y train shape: ", y_train.shape)

print("X test shape: ", X_test.shape)

print("y test shape: ", y_test.shape)
from sklearn.linear_model import LogisticRegression



lreg_model = LogisticRegression(max_iter=10000)



lreg_model.fit(X_train, y_train)

y_pred_lreg_model = lreg_model.predict(X_test)



lreg_model_score = lreg_model.score(X_test, y_test)

print(lreg_model_score)
from sklearn import svm



svm_model = svm.SVC(C=100000)



svm_model.fit(X_train, y_train)

y_pred_svm_model  = svm_model.predict(X_test)



svm_model_score = svm_model.score(X_test, y_test)

print(svm_model_score)
from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()



nb_model.fit(X_train, y_train)

y_pred_nb_model = nb_model.predict(X_test)



nb_model_score = nb_model.score(X_test, y_test)

print(nb_model_score)
from sklearn.tree import DecisionTreeClassifier



dtc_model = DecisionTreeClassifier(random_state=10)



dtc_model.fit(X_train, y_train)

y_pred_dtc_model = dtc_model.predict(X_test)



dtc_model_score = dtc_model.score(X_test, y_test)

print(dtc_model_score)
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier(n_estimators=1000, random_state=10)



rf_model.fit(X_train, y_train)

y_pred_rf_model = rf_model.predict(X_test)



rf_model_score = rf_model.score(X_test, y_test)

print(rf_model_score)
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors=10)



knn_model.fit(X_train, y_train)

y_pred_knn_model = knn_model.predict(X_test)



knn_model_score = knn_model.score(X_test, y_test)

print(knn_model_score)
models = list()

scores = list()

for vars in dir():

    if vars.endswith("_model_score"):

        print(f"{vars}: {eval(vars)}")

        models.append(vars)

        scores.append(eval(vars))
df = {'models': models, 'scores': scores}

pd.DataFrame.from_dict(df)
count  = df["scores"]

# plt.figure(figsize=(16,9))

sns.barplot(df["models"], df["scores"], alpha=1)

# plt.title('Tweets vs User Location')

plt.ylabel('Number of Occurrences', fontsize=12)

# plt.xlabel('State', fontsize=12)

plt.xticks(rotation=75)

plt.show()