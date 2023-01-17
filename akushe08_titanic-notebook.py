from IPython.display import display
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
data1 = [train, test]
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
sns.countplot(train['Survived'])
# Making Column 'FamilySize'
for dataset in data1:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
train.corrwith(train['FamilySize'])
sns.heatmap(train.corr(), annot=True)
sns.boxplot(x='Pclass', y='Fare', data=train)
sns.countplot(x='Pclass', hue='FamilySize', data=train)
train['FamilySize'].value_counts()
for dataset in data1:
    display(dataset['Cabin'].isnull().sum() / len(dataset['Cabin']))
train['Cabin']
for dataset in data1:
    #dataset['TravellingAlone'] = np.nan
    dataset.loc[dataset['FamilySize'] == 1 ,'TravellingAlone'] = 1
    dataset.loc[dataset['FamilySize'] != 1 ,'TravellingAlone'] = 0
train.head()
for dataset in data1:
    dataset['TravellingAlone'] = dataset['TravellingAlone'].astype('int64')
train.head()
train.corrwith(train['TravellingAlone'])
for dataset in data1:
    print(dataset.isnull().sum())
    print('-'*19)
train.corrwith(train['Age'])
sns.boxplot(x='Pclass', y='Age', data=train)
sns.distplot(train['Age'])
train.loc[(train['Pclass'] == 1) ,'Age'].isnull().sum()
train.loc[(train['Pclass'] == 2) ,'Age'].isnull().sum()
train.loc[(train['Pclass'] == 3) ,'Age'].isnull().sum()
train['Age'].isnull().sum()
for dataset in data1:
    dataset.loc[(dataset['Pclass'] == 1), 'Age'] = dataset.loc[(dataset['Pclass'] == 1), 'Age'].fillna(dataset.loc[(dataset['Pclass'] == 1), 'Age'].median())
    dataset.loc[(dataset['Pclass'] == 2), 'Age'] = dataset.loc[(dataset['Pclass'] == 2), 'Age'].fillna(dataset.loc[(dataset['Pclass'] == 2), 'Age'].median())
    dataset.loc[(dataset['Pclass'] == 3), 'Age'] = dataset.loc[(dataset['Pclass'] == 3), 'Age'].fillna(dataset.loc[(dataset['Pclass'] == 3), 'Age'].median())
train['Age'].isnull().sum()
for dataset in data1:
    print(dataset.isnull().sum())
    print('#'*100)
sns.distplot(train['Fare'])
sns.boxplot(x='Survived', y='Fare', data=train)
train.corrwith(train['Fare'])
for dataset in data1:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
test['Fare'].isnull().sum()
type(train['Embarked'].mode().values[0])
for dataset in data1:
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode().values[0])
passenger_id = test['PassengerId']
for dataset in data1:
    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize'], axis=1, inplace=True)
train.head()
train['Age'].hist()
for dataset in data1:
    dataset['Age'] = dataset['Age'].astype('int64')
train['CategoricalAge'] = pd.cut(train['Age'], 5)
train['CategoricalFare'] = pd.qcut(train['Fare'], 3)
train['CategoricalAge'].value_counts()
train['CategoricalFare'].value_counts()
for dataset in data1:
    dataset.loc[(dataset['Age'] < 16), 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 16) & (dataset['Age'] < 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] >= 32) & (dataset['Age'] < 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] >= 48) & (dataset['Age'] < 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] >= 64), 'Age'] = 4
    dataset.loc[(dataset['Fare'] < 9.00), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] >=9.00 ) & (dataset['Fare'] < 26.00), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] >= 26.00), 'Fare'] = 2
train.drop(['CategoricalAge', 'CategoricalFare'] ,axis=1, inplace=True)
sns.countplot(x='Survived', hue='Sex', data=train)
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder
labelEnc = LabelEncoder()
for dataset in data1:
    dataset['Sex_En'] = labelEnc.fit_transform(dataset['Sex'])
    dataset['Embarked_En'] = labelEnc.fit_transform(dataset['Embarked'])
    dataset['Pclass_En'] = labelEnc.fit_transform(dataset['Pclass'])
    dataset.drop(['Sex', 'Embarked', 'Pclass'], axis=1, inplace=True)
train.head()
test.head()
X = train.drop('Survived', axis=1)
y = y_train = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
y_pred = lr_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test,y_pred)
cm
ac = accuracy_score(y_test,y_pred)
ac
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
ac = accuracy_score(y_test,y_pred_rf)
ac
cm = confusion_matrix(y_test,y_pred_rf)
cm
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
y_pred_dt = dt_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred_dt)
cm
ac = accuracy_score(y_test,y_pred_dt)
ac
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(3)
knn_model.fit(X_train,y_train)
y_pred_knn = knn_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred_knn)
cm
ac = accuracy_score(y_test,y_pred_knn)
ac
y_pred_test = lr_model.predict(test)
from sklearn.svm import SVC
svc_model = SVC(probability=True)
svc_model.fit(X_train,y_train)
y_pred_svc = svc_model.predict(X_test)
ac = accuracy_score(y_test,y_pred_svc)
ac
y_pred_svc_test = svc_model.predict(test)
from sklearn.metrics import confusion_matrix, classification_report
def evaluate_models(models):
    for model in models:
        print("Evaluation for {}".format(type(model).__name__))
        print("----"*20)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        print("\nConfusion Matrix:\n",cm)
        ac = accuracy_score(y_test,y_pred)
        print("\nAccuracy:\n",ac)
        print("\nClassification Report:\n")
        print(classification_report(y_test,y_pred))
models = [lr_model,rf_model,dt_model, knn_model, svc_model]
evaluate_models(models)
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
def cross_validate_models(models, splits):
    kf = KFold(n_splits=splits,shuffle=True)
    for model in models:
        scores = cross_val_score(model,
                                 X_train,
                                 y_train,
                                 cv=kf,
                                 n_jobs=12,
                                 scoring="accuracy")
        print("Cross-Validation for {}:\n".format(type(model).__name__))
        print("Mean score: ", np.mean(scores))
        print("Variance of score: ", np.std(scores)**2)
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(111)
        ax = sns.distplot(scores)
        ax.set_xlabel("Cross-Validated Accuracy scores")
        ax.set_ylabel("Frequency")
        ax.set_title('Frequency Distribution of Cross-Validated Accuracy scores for {}'.format(type(model).__name__), fontsize = 15)
cross_validate_models(models,100)
lr_params = {"penalty" : ["l1", "l2"],
             "C" : np.logspace(0, 4, 10),
             "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
dt_params = {"criterion":["gini","entropy"],
             "splitter":["best","random"],
             "max_depth":[3,9,81,200],
             "min_samples_split":[25,30,35,50]}
knn_params = {"n_neighbors" : [1,3,5,7,9,11,13,15,17,19,21],
              "metric" :  ['euclidean', 'manhattan', 'minkowski'],
              "weights" : ['uniform', 'distance']}             
rf_params = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

svc_params = {'kernel' : ['linear', 'rbf', 'poly'],
'gamma' : [0.1, 1, 10, 100],
'C' : [0.1, 1, 10, 100, 1000],
'degree' : [0, 1, 2, 3, 4, 5, 6]}

models = [lr_model,rf_model,dt_model, knn_model, svc_model]
params = [lr_params,rf_params,dt_params, knn_params, svc_params]
tuned_models = []
import time
def hyper_param_tuning(models,params,splits,scorer):
    for i in range(len(models)):
        gsearch = RandomizedSearchCV(estimator=models[i],
                               param_distributions=params[i],
                               scoring=scorer,
                               verbose=2,
                               n_jobs=-1,
                               cv=5)
        start = time.time()
        gsearch.fit(X_train,y_train)
        end = time.time()
        
        print("Grid Search Results for {}:\n".format(type(models[i]).__name__))
        print("Time taken for tuning (in secs): \n", end-start)
        print("Best parameters: \n",gsearch.best_params_)
        print("Best score: \n",gsearch.best_score_)
        tuned_models.append(gsearch.best_estimator_)
        print("\n\n")
hyper_param_tuning(models,params,100,"accuracy")
tuned_models
dt_model_updated =  DecisionTreeClassifier(criterion='entropy', max_depth=81, min_samples_split=25,
                        splitter='random')
dt_model_updated.fit(X_train,y_train)
y_pred_dt_updated = dt_model_updated.predict(test)
submission1 = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": y_pred_dt_updated
    })
submission1.to_csv('mysubmission7.csv', index=False)
