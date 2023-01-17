# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()
data.tail()
data.info()
data=data.drop("Serial No.",axis=1)
data.head()
data.columns
data.columns = ["gre_score","toefl_score","uni_rating","sop","lor","cgpa","research","admit_chance"]
data.columns
data.describe()
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()
def modify(feature):
    if feature["admit_chance"] >= 0.7:
        return 1
    return 0 

data["admit_chance"] = data.apply(modify,axis=1)
data.head()
sns.countplot(x = "admit_chance",data=data,palette="RdBu")
plt.show()
c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"gre_score",bins=25)
plt.show()
data.groupby("admit_chance")["gre_score"].mean()
c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"toefl_score",bins=25)
plt.show()
data.groupby("admit_chance")["toefl_score"].mean()
c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"cgpa",bins=25)
plt.show()
data.groupby("admit_chance")["cgpa"].mean()
sns.distplot(data["cgpa"])
plt.show()
sns.catplot(x="research",y="admit_chance",data=data,kind="bar")
plt.show()
sns.catplot(x="uni_rating",y="admit_chance",data=data,kind="bar")
plt.show()
def modify(feature):
    if feature["uni_rating"] >= 3:
        return 1
    return 0 

data["uni_rating"] = data.apply(modify,axis=1)
data.head()
sns.catplot(x="uni_rating",y="admit_chance",data=data,kind="bar")
plt.show()
sns.catplot(x="sop",y="admit_chance",data=data,kind="bar")
plt.show()
def modify(feature):
    if feature["sop"] >= 3.0:
        return 1
    return 0 

data["sop"] = data.apply(modify,axis=1)
data.head()
sns.catplot(x="sop",y="admit_chance",data=data,kind="bar")
plt.show()
sns.catplot(x="lor",y="admit_chance",data=data,kind="bar")
plt.show()
def modify(feature):
    if feature["lor"] >= 3.0:
        return 1
    return 0 

data["lor"] = data.apply(modify,axis=1)
data.head()
sns.catplot(x="lor",y="admit_chance",data=data,kind="bar")
plt.show()
sns.scatterplot(data = data, x ="gre_score", y="toefl_score",hue="admit_chance")
plt.show()
sns.scatterplot(data = data, x ="gre_score", y="cgpa",hue="admit_chance")
plt.show()
sns.scatterplot(data = data, x ="toefl_score", y="cgpa",hue="admit_chance")
plt.show()
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
x = data.drop("admit_chance",axis = 1)
y = data["admit_chance"]

scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print("x_train:", len(x_train))
print("x_test:", len(x_test))
print("y_train:", len(y_train))
print("y_test:", len(y_test))
logreg = LogisticRegression(max_iter=500)

logreg.fit(x_train,y_train)

print("Train Accuracy:", logreg.score(x_train,y_train))
print("Test Accuracy:", logreg.score(x_test,y_test))
classifier = [KNeighborsClassifier(),
              DecisionTreeClassifier(random_state=42),
              LogisticRegression(random_state=42),
              SVC(random_state=42),
              RandomForestClassifier(random_state=42),
              XGBClassifier(random_state=42, objective="binary:logistic")]

knn_params = {"n_neighbors": np.linspace(1,19,10,dtype=int),
                 "weights": ["uniform","distance"],
                 "metric": ["euclidean","manhattan","minkowski"]}

dt_params = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)} 

lr_params = {"C":np.logspace(-3,3,7),
             "penalty": ["l1","l2"]}

svm_params = {"kernel" : ["rbf"],
              "gamma": [0.001, 0.01, 0.1, 1],
              "C": [1,10,50,100,200,300,1000]}

rf_params = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

xgb_params = {"learning_rate":[0.01,0.1,1],
              "n_estimators":[50,100,150],
              "max_depth":[3,5,7],
              "gamma":[1,2,3,4]}

classifier_params = [knn_params,
                     dt_params,
                     lr_params,
                     svm_params,
                     rf_params,
                     xgb_params]
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(estimator = classifier[i], param_grid = classifier_params[i],
                       cv = StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs= -1, verbose= 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
cv_results = pd.DataFrame({"ML Models":["KNeighborsClassifier",
                                        "Decision Tree",
                                        "Logistic Regression",
                                        "SVM",
                                        "Random Forest",
                                        "XGBClassifier"],
                           "Cross Validation Means":cv_result})
print(cv_results)