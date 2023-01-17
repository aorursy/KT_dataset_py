# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew,skewtest
from sklearn.impute import KNNImputer
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.info()
data.describe().T
data.columns
# I checked that if features has space or not in their names. We dont want to search for mistakes later.
data.tail()
data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.nan)
print(data.isnull().sum())
p = data.hist(figsize = (20,20),bins=25)
plt.show()
c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"Glucose",bins=25)
plt.show()
c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"BloodPressure",bins=25)
plt.show()
c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"SkinThickness",bins=25)
plt.show()
c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"Insulin",bins=25)
plt.show()
c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"BMI",bins=25)
plt.show()
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()
imputer = KNNImputer(n_neighbors=5)
data_filled = imputer.fit_transform(data)
data_filled_df = pd.DataFrame(data_filled, index=data.index,columns=data.columns)
data_filled_df.isnull().sum()
# All missing values are filled.
corr = data_filled_df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
X = data_filled_df.drop("Outcome",axis=1)
y = data_filled_df["Outcome"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

print("Train Accuracy: ",knn.score(X_train,y_train))
print("Test Accuracy: ",knn.score(X_test,y_test))
random_state = 42
classifier = [SVC(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier(),
             xgb.XGBClassifier(seed=123, objective="reg:logistic")]

svc_params = {"kernel" : ["rbf"],
              "gamma": [0.001, 0.01, 0.1, 1],
              "C": [1,10,50,100,200,300,1000]}

logreg_params = {"C":np.logspace(-3,3,7),
                 "penalty": ["l1","l2"]}

knn_params = {"n_neighbors": np.arange(1,20,1),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan","minkowski"]}

xgb_params = {"n_estimators": np.arange(1,15,1),
              "max_depth" : np.arange(1,10,1),
              "num_boost_round": np.arange(1,10,1)}

classifier_params = [svc_params,
                    logreg_params,
                    knn_params,
                    xgb_params]
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_params[i], cv = 5, scoring = "accuracy", n_jobs = -1,verbose=1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
cv_results = pd.DataFrame({"ML Models":["SVM",
                                        "LogisticRegression",
                                        "KNeighborsClassifier",
                                        "XGBClassifier"],
                           "Cross Validation Means":cv_result})
print(cv_results)


g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")