import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv ("/kaggle/input/breast-cancer-dataset-for-beginners/train.csv")
df.head()
df.drop("Id", axis = 1, inplace = True)
# Nice one

df.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

plt.figure(figsize = (15,15))

sns.heatmap(df.corr(), annot =True, fmt = ".2g")
# We will keep the Threshold of  the 0.60

final_col = df.corr()["class"].reset_index().sort_values(by= "class", ascending = False)[1:-2]["index"].to_list()
X = df[final_col]

y= df["class"]
sns.set()

sns.countplot(y)

plt.show()
# Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state =42, stratify = y)
# Scaling the data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Fitting the model 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import  KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
classifiers = [LogisticRegression(), 

                KNeighborsClassifier(),

               GaussianNB(),

               SVC(),

               DecisionTreeClassifier(),

               RandomForestClassifier()

              ]
# Fitting all classifier one by one and calculate  accuracy

clfs = []

score_mean = []

from sklearn.model_selection import cross_val_score

for clf in classifiers:

    score = cross_val_score(clf, X_train_scaled, y_train, cv =  5)

    clfs.append(clf.__class__.__name__)

    score_mean.append(score.mean())

new_frame = pd.DataFrame({"classifier": clfs, 'score':score_mean})

sns.set(font_scale=1.2)

plt.figure(figsize = (8,8))

plt.title('Cross Val score of Model on training set')

sns.barplot(y="classifier", x ="score", data = new_frame, palette="coolwarm")

plt.show()
new_frame
# We Will use the  Random Forest For Furthur fitting 

from sklearn.metrics import confusion_matrix, classification_report

rfe = RandomForestClassifier(max_depth = 6, n_estimators=100)

rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)
conf = confusion_matrix(y_test, y_pred)

sns.heatmap(conf, annot = True,  fmt = ".2g")

plt.show()
print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
param = {

    'criterion': ['gini', 'entropy'],

  'max_depth':[4,6,8],

    'max_features' :  [0.4, 'auto'],

  'min_samples_leaf' : [0.10],

  'n_estimators': [100, 200],



}
rfe2 = RandomForestClassifier(n_jobs= -1,verbose=1, random_state=42)
from sklearn.model_selection import GridSearchCV

gsv = GridSearchCV(rfe2, param, scoring="roc_auc",cv= 3) 

gsv.fit(X_train_scaled, y_train)
gsv.best_params_
gsv.best_score_
# Fitting the Model with the Best Estimator 

final_rfe = RandomForestClassifier(**gsv.best_params_,n_jobs= -1,verbose=1, random_state=42)
final_rfe.fit(X_train_scaled, y_train)

y_pred = final_rfe.predict(X_test_scaled)

print(round(accuracy_score(y_test,y_pred) *100, 2))
conf = confusion_matrix(y_test, y_pred)

sns.heatmap(conf, annot = True,  fmt = ".2g")

plt.show()
# By tuning the predictions are improve
print(classification_report(y_test, y_pred))