import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import warnings

warnings.filterwarnings('ignore')
# Train data

df_train = pd.read_csv("../input/blood-train.csv")

df_train.head()
# Test data

df_test = pd.read_csv("../input/blood-test.csv")

df_test.head()
#labelling

df_train.rename(columns={"Unnamed: 0":"Donor_id"},inplace=True)

df_train.head()
df_test.rename(columns={"Unnamed: 0":"Donor_id"},inplace=True)

df_test.head()
df_train.shape, df_test.shape
df_train.info()

print("\n--------------------------------------\n")

df_test.info()
#Statistical Inference



df_train.describe()
df_test.describe()
# Correlation

train_corr = df_train.corr()

sns.heatmap(train_corr)
test_corr = df_test.corr()

sns.heatmap(test_corr)
# Training data

X_train = df_train.iloc[:,[1,2,3,4]].values

y_train = df_train.iloc[:,-1].values
X_train,y_train
# Test data

X_test = df_test.iloc[:,[1,2,3,4]].values
X_test
#Feature Scaling

from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

X_train = Scaler.fit_transform(X_train)



X_test = Scaler.fit_transform(X_test)
X_train, X_test
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_train,y_train)

score
#Applying k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)

mean = accuracies.mean()

std = accuracies.std()
mean,std
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train,y_train)

score = rf.score(X_train,y_train)

score
y_pred = rf.predict(X_test)
from xgboost import XGBClassifier

xg = XGBClassifier()

xg.fit(X_train,y_train)

score = xg.score(X_train,y_train)

score
#Applying k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=xg,X=X_train,y=y_train,cv=10)

mean = accuracies.mean()

std = accuracies.std()
mean,std
y_pred = xg.predict(X_test)