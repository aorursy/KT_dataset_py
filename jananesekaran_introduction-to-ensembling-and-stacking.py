import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/Iris.csv")

data.head()
# Lets convert species names to numbers and drop the ID column.

data['Species'] = data['Species'].map( {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3} ).astype(int)

data = data.drop('Id', axis = 1)

data.head()
data_full=data.as_matrix()

X = data_full[:,:4]

y = data_full[:,4:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('data shape: ', X_train.shape)

print('labels shape: ', y_train.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
colormap = plt.cm.viridis

plt.figure(figsize=(5,5))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5).fit(X_train, y_train)

model1_train = bagging.predict(X_train)

model1_test = bagging.predict(X_test)
random_forest=RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=2, random_state=0).fit(X_train,y_train)

model2_train = random_forest.predict(X_train)

model2_test = random_forest.predict(X_test)
Ada_boost=AdaBoostClassifier(n_estimators=500).fit(X_train,y_train)

model3_train=Ada_boost.predict(X_train) 

model3_test=Ada_boost.predict(X_test)
from sklearn.ensemble import GradientBoostingClassifier

Gradient_boost = GradientBoostingClassifier(n_estimators=500).fit(X_train, y_train)

model4_train=Gradient_boost.predict(X_train)

model4_test=Gradient_boost.predict(X_test)                 
base_predictions_train = pd.DataFrame( {'Bagging': model1_train.ravel(),

     'Random_Forest': model2_train.ravel(),

     'AdaBoost': model3_train.ravel(),

      'GradientBoost': model4_train.ravel()

    })

base_predictions_test = pd.DataFrame( {'Bagging': model1_test.ravel(),

     'Random_Forest': model2_test.ravel(),

     'AdaBoost': model3_test.ravel(),

      'GradientBoost': model4_test.ravel()

    })

X_new_train = base_predictions_train.as_matrix()

X_new_test = base_predictions_test.as_matrix()

base_predictions_train.head()
sns.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
xgboost = xgb.XGBClassifier(n_estimators= 5,max_depth= 10,min_child_weight= 2,gamma=0.9,subsample=0.8,colsample_bytree=0.8,

 objective= 'binary:logistic',nthread= -1,scale_pos_weight=1).fit(X_new_train, y_train)

y_train_pred = xgboost.predict(X_new_train)

y_test_pred = xgboost.predict(X_new_test)

train_score = accuracy_score(y_train, y_train_pred)

test_score = accuracy_score(y_test, y_test_pred)

print("Train_Score:",train_score)

print("Test_Score:",test_score)