import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/diabetes-data-set/diabetes-dataset.csv")
df.head()
sns.set()

sns.countplot(df["Outcome"])

plt.show()
df.isna().sum()
from sklearn.preprocessing import StandardScaler
# Split the data 

from sklearn.model_selection import train_test_split

X = df.drop(columns = "Outcome")

y = df["Outcome"]

X_train , X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42, stratify = y)

# Scaling the data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Fitting the model to the training data

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import xgboost
from sklearn.model_selection import KFold, cross_val_score

for model in [ 

    DummyClassifier,

    LogisticRegression,

    DecisionTreeClassifier,

    KNeighborsClassifier,

    GaussianNB,

    SVC,

    RandomForestClassifier,

    xgboost.XGBClassifier,

]:

    cls = model()

    kf = KFold(n_splits = 5, random_state = 45)

    score = cross_val_score(cls, X_train_scaled, y_train, cv = kf, scoring="roc_auc")

    print(

        f"{model.__name__:22}  AUC: "

        

        f"\t {score.mean():.3f} STD: {score.std():.2f}"

        

    )
# Withour any certain hyper param tuning the Random_Forest Model is best

# leta get model working



rfe = RandomForestClassifier(n_estimators=1000, random_state = 42)
rfe.fit(X_train_scaled, y_train)
# Evaluating the Random Forest Model

print ("Accuacy on test set is ", round(rfe.score(X_test_scaled, y_test) * 100, 2), "%")
from sklearn.metrics import precision_score

print("precision Score is ", round (precision_score(y_test, rfe.predict(X_test_scaled)) * 100 , 2))
print("Feature importance is \n" )

for i,j in zip(X_train.columns.to_list(), rfe.feature_importances_.tolist()):

    if(i == "DiabetesPedigreeFunction"):

        print (i , "\t \t ", j)

    elif(i=="Age" or i =="BMI"):

        print (i , "\t \t \t\t\t", j)

    else :   

        print (i , "\t \t \t \t", j)
# As we seleted the model we can try the hyperparam tuning

from sklearn.model_selection import GridSearchCV



new_rfe = RandomForestClassifier()



params = {

     "max_features": [0.4, "auto"],

     "n_estimators": [15, 200, 500, 1000],

     "min_samples_leaf": [1, 0.1],

     "random_state": [42],

}



cvs = GridSearchCV(new_rfe, params, n_jobs = -1).fit(X_train_scaled, y_train)
print(cvs.best_score_)
print(cvs.best_estimator_)
print(cvs.best_params_)
# fitting the model with best params

rfe_final = RandomForestClassifier(

**{'max_features': 0.4, 'min_samples_leaf': 1, 'n_estimators': 1000, 'random_state': 42}

)
rfe_final.fit(X_train_scaled, y_train)
y_pred = rfe_final.predict(X_test_scaled)
print(precision_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix, classification_report
sns.set()

sns.heatmap(confusion_matrix(y_test, y_pred), annot =True)

plt.show()
print(classification_report(y_test, y_pred))
# print Roc_auc_score

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_pred))