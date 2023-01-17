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
# Additional imports

import matplotlib.pyplot as plt

import seaborn as sns



# Data splitting/parameter tuning

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV





# ML models

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier



# Feature processing

from sklearn.feature_selection import SelectPercentile, chi2



# Evaluation metrics

from sklearn.metrics import confusion_matrix
heart_path = "../input/heart.csv"

heart_data = pd.read_csv(heart_path)
heart_data.head(5)
print("Heart data shape is:", heart_data.shape[0], "x", heart_data.shape[1])
# Missing values

heart_data.isnull().sum()
sns.distplot(heart_data["age"], bins=4, kde=False)
sns.countplot(heart_data["sex"])
total = len(heart_data["sex"])

males = heart_data["sex"].sum()

females = len(heart_data["sex"]) - males

print("Porcentage of males:", round(males/total*100, 3))

print("procentage of females:", round(females/total*100, 3))
sex_graph = sns.countplot(heart_data["sex"], hue=heart_data["target"])

sex_graph.set_ylabel("amount")
cp_graph = sns.countplot(heart_data["cp"], color="purple")

cp_graph.set_xlabel("type of chest pain")

cp_graph.set_ylabel("amount")
plt.figure(figsize=(10, 10))

sns.heatmap(heart_data.corr(), annot=True, fmt='.2f')
heart_data.dtypes
heart_data['sex'] = heart_data['sex'].astype('object')

heart_data['cp'] = heart_data['cp'].astype('object')

heart_data['fbs'] = heart_data['fbs'].astype('object')

heart_data['restecg'] = heart_data['restecg'].astype('object')

heart_data['exang'] = heart_data['exang'].astype('object')

heart_data['slope'] = heart_data['slope'].astype('object')

heart_data['thal'] = heart_data['thal'].astype('object')
heart_data.dtypes
heart_data = pd.get_dummies(heart_data)

heart_data.head()
print("Heart data shape is:", heart_data.shape[0], "x", heart_data.shape[1])
# Getting features and target

X = heart_data.drop(["target"], axis=1)

y = heart_data["target"]
# Random Forest

rf_model = RandomForestClassifier(n_estimators=100)

rf_predictions = cross_val_predict(rf_model, X, y, cv=5)

print(confusion_matrix(y, rf_predictions))

rf_scores = cross_val_score(rf_model, X, y, scoring="recall", cv=5)

print("recall:", rf_scores.mean())
# Logistic Regression

lr_model = LogisticRegression(solver="liblinear")

lr_predictions = cross_val_predict(lr_model, X, y, cv=5)

print(confusion_matrix(y, lr_predictions))

lr_scores = cross_val_score(lr_model, X, y, scoring="recall", cv=5)

print("recall:", lr_scores.mean())
# Support Vector Machine

svc_model = SVC(gamma="auto")

svc_predictions = cross_val_predict(svc_model, X, y, cv=5)

print(confusion_matrix(y, svc_predictions))

svc_scores = cross_val_score(svc_model, X, y, scoring="recall", cv=5)

print("recall:", svc_scores.mean())
# Naive Bayes

nb_model = GaussianNB()

nb_predictions = cross_val_predict(nb_model, X, y, cv=5)

print(confusion_matrix(y, nb_predictions))

nb_scores = cross_val_score(nb_model, X, y, scoring="recall", cv=5)

print("recall:", nb_scores.mean())
# XGBoost (The most popular model for kaggle competitions)

xgb_model = XGBClassifier()

xgb_predictions = cross_val_predict(nb_model, X, y, cv=5)

print(confusion_matrix(y, xgb_predictions))

xgb_scores = cross_val_score(xgb_model, X, y, scoring="recall", cv=5)

print("recall:", xgb_scores.mean())
X["age"] = X["age"].map(lambda x: (x - X["age"].min()) / (X["age"].max() - X["age"].min()))

X["trestbps"] = X["trestbps"].map(lambda x: (x - X["trestbps"].min()) / (X["trestbps"].max() - X["trestbps"].min()))

X["chol"] = X["chol"].map(lambda x: (x - X["chol"].min()) / (X["chol"].max() - X["chol"].min()))

X["thalach"] = X["thalach"].map(lambda x: (x - X["thalach"].min()) / (X["thalach"].max() - X["thalach"].min()))

X["oldpeak"] = X["oldpeak"].map(lambda x: (x - X["oldpeak"].min()) / (X["oldpeak"].max() - X["oldpeak"].min()))
# Support Vector Machine

svc_model = SVC(gamma="auto")

svc_predictions = cross_val_predict(svc_model, X, y, cv=5)

print(confusion_matrix(y, svc_predictions))

svc_scores = cross_val_score(svc_model, X, y, scoring="recall", cv=5)

print("recall:", svc_scores.mean())
best_recall = 0

for n in range(1, 101):

    X_new = SelectPercentile(chi2, percentile=n).fit_transform(X, y)



    svc_model = SVC(gamma="auto")

    svc_predictions = cross_val_predict(svc_model, X_new, y, cv=5)

    svc_scores = cross_val_score(svc_model, X_new, y, scoring="recall", cv=5)

    

    if svc_scores.mean() > best_recall:

        best_recall = svc_scores.mean()

        print(confusion_matrix(y, svc_predictions))

        print("the best porcentage so far:", n)

        print("the best recall so far", svc_scores.mean(), "\n")

        
X_new = SelectPercentile(chi2, percentile=33).fit_transform(X, y)        

svc_model = SVC(gamma="auto")

svc_predictions = cross_val_predict(svc_model, X_new, y, cv=5)

print(confusion_matrix(y, svc_predictions))

svc_scores = cross_val_score(svc_model, X_new, y, scoring="recall", cv=5)

print("recall:", svc_scores.mean(), "\n")



print("Old number of features used:",X.shape[1])

print("New number of features used:",X_new.shape[1])
# EXPLORE FOR BETTER WAY TO KNOW BEST FEATURES AFTER FEATURE SELECTION



#plt.figure(figsize=(20, 20))

#sns.heatmap(heart_data.corr(), annot=True, fmt='.2f')




Cs = [1, 10, 100, 1000]

kernels = ["linear", "rbf", "poly"]



for c in Cs:

    for k in kernels:

        

        print("C:", c)

        print("Kernel:", k)

        svc_model = SVC(gamma="auto", C=c, kernel=k)

        svc_predictions = cross_val_predict(svc_model, X_new, y, cv=5)

        print(confusion_matrix(y, svc_predictions))

        svc_scores = cross_val_score(svc_model, X_new, y, scoring="recall", cv=5)

        print("recall:", svc_scores.mean(), "\n")




