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
import re
import matplotlib.pyplot as plt
import seaborn as sb
import scikitplot as skplt
train = pd.read_csv("/kaggle/input/titanic/train.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.shape, test.shape
train.head() # Pclass - Ticket class, 
             # SibSp - # of siblings / spouses aboard the Titanic, 
             # Parch - # of parents / children aboard the Titanic,
             # Embarked - Port of Embarked
np.unique(train["Pclass"]), np.unique(train["SibSp"]), np.unique(train["Parch"]), train["Embarked"].unique()
train["Cabin"].unique()
# null values in the dataframe :
train.isnull().sum()
# removing null value of embarked by droppping it 
train = train.dropna(axis=0, subset=["Embarked"])
train.shape
print(test.isnull().sum())
train.head()
age = train[train["Survived"]==1]["Age"]
plt.hist(age,rwidth=0.85)
age = train[train["Survived"]==0]["Age"]
plt.hist(age,rwidth=0.85)
train["Sex"].value_counts()
male_survived = train[train["Sex"]=='male']["Survived"]
female_survived = train[train["Sex"]=='female']["Survived"]
# survival rate : 
print("Survival rate for Men : {}\nSurvival rate for Women : {}".format(sum(male_survived)/len(male_survived) *100, sum(female_survived)/len(female_survived)*100))
# Let's check for the kids having age(<18)

kids = train[train["Age"]<=18]["Survived"]
print("Survival rate for kids : {}".format(sum(kids)/len(kids)*100))
# Compairing the price of the passengers who has cabin and who doesn't have a cabin(their cabin are not known)
ticket_price_with_cabin = np.array(train[train['Cabin'].notnull()]["Fare"])
ticket_price_without_cabin = np.array(train[train['Cabin'].isnull()]["Fare"])
# Peoples without cabins: Lower class or Middle Class (Financial status)
# People with cabin : Upper class (Financial status)

print("************ People without cabin **********************")
print("Mean : {}\tSD : {}".format(ticket_price_without_cabin.mean(), ticket_price_without_cabin.std()))
print("Min : {}\tMax : {}".format(min(ticket_price_without_cabin), max(ticket_price_without_cabin)))

print("************ People with cabin ************************")
print("Mean : {}\tSD : {}".format(ticket_price_with_cabin.mean(), ticket_price_with_cabin.std()))
print("Min : {}\tMax : {}".format(min(ticket_price_with_cabin), max(ticket_price_with_cabin)))
plt.subplot(1, 2, 1)
sb.distplot(ticket_price_without_cabin, kde=True)

plt.subplot(1, 2, 2)
sb.distplot(ticket_price_with_cabin, kde=True)
plt.show()
train["Pclass"].value_counts() 
without_cabin  = dict(train[train['Cabin'].isnull()]["Pclass"].value_counts())
with_cabin = dict(train[train['Cabin'].notnull()]["Pclass"].value_counts())
without_cabin = [without_cabin[1], without_cabin[2], without_cabin[3]]
with_cabin = [with_cabin[1], with_cabin[2], with_cabin[3]]

data = pd.DataFrame({"With cabin": with_cabin, "Without cabin": without_cabin}, index=[1,2,3])
data.plot.bar()
# Are the cabin crew preferenced upper class people while saving them over middle and lower class people.

upper_class = train[train["Pclass"]==3]["Survived"]
middle_class = train[train["Pclass"]==2]["Survived"]
lower_class = train[train["Pclass"]==1]["Survived"]
upper = [upper_class.sum(), len(upper_class)-upper_class.sum()]
middle = [middle_class.sum(), len(middle_class)-middle_class.sum()] 
lower = [lower_class.sum() , len(lower_class)-lower_class.sum()]

data = pd.DataFrame({"upper":upper,"middle":middle, "lower":lower})

data. plot.bar(figsize=(10,7))   # 0 - Not survived 
                                 # 1 - Survived
# Age binning
# 1 : 1-10
# 2 : 10-20 ......
def age(data):
    arr = []
    for i in data:
        try:
            arr.append(int(i / 10))
        except:
            arr.append(0)
    return arr
train["Age"] = age(train["Age"])
test["Age"] = age(test["Age"])
def fare(data):
    fare = []
    for i in data:
        if 0<i<10: fare.append(1)
        elif 10<i<30: fare.append(2)
        elif 30<i<50: fare.append(3)
        elif 50<i<100: fare.append(4)
        else: fare.append(5)
    return fare
train["Fare"] = fare(train["Fare"])
test["Fare"] = fare(test["Fare"])
gender = {"male":1, "female":0}
embarked = {'S':1, 'C':0, 'Q':2}

train["Sex"] = train["Sex"].map(gender)
test["Sex"] = test["Sex"].map(gender)

train["Embarked"] = train["Embarked"].map(embarked)
test["Embarked"] = test["Embarked"].map(embarked)
train["Cabin"] = train["Cabin"].fillna(0)
test["Cabin"] = test["Cabin"].fillna(0)
def cabin_f(data):
    cabin = []
    cabinet_no= []
    cabinet = {'A':1,'B':2,'C':3,'D':4,'E':5,
              'F':6,'G':7,'T':8}
    for i in data:
        if i!=0:
            cabin.append(cabinet["".join(re.split("[^a-zA-Z]*", i))[0]])
            temp = re.split("[^0-9]",i)[1]
            if temp:
                cabinet_no.append(int(temp))
            else: cabinet_no.append(0)
        else:
            cabin.append(0)
            cabinet_no.append(0)
    return cabin, cabinet_no

cabin, cabinet_no = cabin_f(train["Cabin"])
train["Cabin_type"] = cabin
train["Cabin_no"] = cabinet_no

cabin, cabinet_no = cabin_f(test["Cabin"])
test["Cabin_type"] = cabin
test["Cabinet_no"] = cabinet_no

train = train.drop("Cabin", axis=1)
test = test.drop("Cabin", axis=1)
train["Title"] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train["Title"] = train["Title"].replace(['Lady', 'Capt', 'Don', 'Col', 'Countess', 'Dona', 'Jonkheer', 'Dr', 'Major', 'Rev', 'Sir'], 'rare')

test["Title"] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
test["Title"] = test["Title"].replace(['Lady', 'Capt', 'Don', 'Col', 'Countess', 'Dona', 'Jonkheer', 'Dr', 'Major', 'Rev', 'Sir'], 'rare')

mapper = {'rare':1, 'Mr':2, 'Mrs':3, 'Miss':4,'Master':5 }

train['Title'] = train['Title'].map(mapper)
test['Title'] = test['Title'].map(mapper)
train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].fillna(0)
train["Family_member"] = train["SibSp"] + train["Parch"] + 1
test["Family_member"] = test["SibSp"] + test["Parch"] +1
train["Alone"] = [1 if i==1 else 0 for i in train["Family_member"]]
test["Alone"] = [1 if i==1 else 0 for i in test["Family_member"]]
train.head()
data = train[["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Fare", "Embarked", "Cabin_type", "Cabin_no", "Survived"]]

fig, ax = plt.subplots(figsize=(12, 8)) 
sb.heatmap(data.corr(),  ax= ax, annot= True)
train["Survived"].value_counts()    # Not imbalanced looks fine
y = train["Survived"]

test_id = test["PassengerId"]
test = test.drop(["PassengerId", "Name", "Ticket"], axis=1)
train = train.drop(["PassengerId", "Name", "Ticket", "Survived"], axis=1)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
train = scaler.fit_transform(train)
scaler = StandardScaler()
test = scaler.fit_transform(test)
xtrain, xtest, ytrain, ytest = train_test_split(train, y, train_size=0.80)
xtrain.shape, xtest.shape
grid={"C":np.logspace(-3,3,7), 
      "penalty":['l1','l2'], 
      "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

model = LogisticRegression()
model = GridSearchCV(model,grid,cv=10)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)
model = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
lr_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))

prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 25)}

model = DecisionTreeClassifier()
model = GridSearchCV(model,grid,cv=10)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)
model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
dt_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
"""grid={ 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

model = RandomForestClassifier()
model = GridSearchCV(model,grid,cv=10)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)"""
model = RandomForestClassifier(criterion= 'gini', max_depth=3, max_features='auto', n_estimators= 200)
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
rf_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
"""grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

model = SVC()
model = GridSearchCV(model,grid,cv=10)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)"""
model = SVC(C= 10, gamma=0.1, kernel='rbf')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
svc_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
grid = {'n_neighbors':[3,5,11,19],'weights':['uniform', 'distance'], 'metric':['euclidean','manhattan']}

model = KNeighborsClassifier()
model = GridSearchCV(model,grid,cv=10)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)
model = KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='distance')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
knn_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
"""grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

model = AdaBoostClassifier(base_estimator = tree)
model = GridSearchCV(model, grid, scoring = 'roc_auc')
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)"""
tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

model = AdaBoostClassifier(base_estimator=tree)
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
ada_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
"""grid = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.075, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 8),
    "min_samples_leaf": np.linspace(0.1, 0.5, 8),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.8, 0.9,  1.0],
    "n_estimators":[10]
    }

model = GridSearchCV(GradientBoostingClassifier(), grid, cv=10, n_jobs=-1)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)"""
model = GradientBoostingClassifier(criterion='friedman_mse', learning_rate= 0.15, 
                                   loss= 'deviance', max_depth= 8, max_features='sqrt', 
                                   min_samples_leaf= 0.15714285714285714, min_samples_split= 0.5, 
                                   n_estimators= 10, subsample=1.0)
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
gb_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
"""grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

model =ExtraTreesClassifier()

model = GridSearchCV(model,grid, scoring="accuracy", n_jobs= 4, verbose = 1)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)"""
model = ExtraTreesClassifier(bootstrap=False, criterion='gini', max_depth= None, 
                             max_features= 3, min_samples_leaf= 1, min_samples_split= 10, 
                             n_estimators= 300)
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
etc_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
param = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

model =XGBClassifier()

model = GridSearchCV(model,grid, scoring="accuracy", n_jobs= 4, verbose = 1)
model.fit(xtrain,ytrain)

print("tuned hpyerparameters :(best parameters) ",model.best_params_)
print("accuracy :",model.best_score_)
model = XGBClassifier(metric= 'euclidean', n_neighbors= 3, weights= 'uniform')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
xgb_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))

prob = model.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, prob)
plt.show()
# Compare accuracy : 
acc = [lr_acc, dt_acc, rf_acc, svc_acc, knn_acc, ada_acc, gb_acc, etc_acc, xgb_acc]
plt.plot(acc,color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.xticks(np.arange(9), ('LR','DT','RF','SVC','KNN','AB','GB','ETC','XGB'))
           #('LogiticRegression','DicisionTree','RandomForest','SVC','KNN','Adaboost','GradientBoost','ExtraTreeclassifier'))
model = XGBClassifier(objective = 'binary:logistic',eta=0.3, min_child_weight =1, subsample=1, colsample_by_tree=0.4,
                      max_depth=9, learning_rate=0.03,metric= 'euclidean', n_neighbors= 3, weights= 'uniform',
                     gamma=0, reg_lambda =2.8,reg_alpha=0, scale_pos_weight=1, n_estimator= 600 )
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)

print("Training accuracy : ",model.score(xtrain, ytrain))
print("Test accuracy : ",model.score(xtest, ytest))
xgb_acc = model.score(xtest, ytest)
print("F1 score : ", f1_score(prediction, ytest))
print("Confusion metrics : \n", confusion_matrix(ytest, prediction))
# highest I got highest by Adaboost on test data
prediction = model.predict(test)
data = {"PassengerId": test_id, "Survived":prediction}
results = pd.DataFrame(data)
results.to_csv("ensemble_python_voting.csv",index=False)

