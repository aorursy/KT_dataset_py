import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
print(check_output(['ls', '../input']).decode('utf8'))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
final_res_data = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.head()
test_df.head()
train_df.describe()
train_df.info()
test_df.describe()
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
# This is done after some cells after reliasing the pclass and fare relationships
test_df.loc[test_df['Fare'].isnull(), 'Fare'] = 5
test_df.isnull().sum()
sns.countplot(data = train_df, x = 'Embarked')
pd.crosstab(train_df['Embarked'], train_df['Survived'])
pd.crosstab(train_df['Embarked'], train_df['Survived'], margins = True)
train_df['Embarked'].fillna('S', inplace=True)
train_df['Embarked'].isnull().any()
train_df.head()
train_df.head()
for dataset in combine:
    dataset['isAlone'] = 0
    dataset['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1
train_df.head()
all_data = pd.concat([train_df, test_df], sort = False, ignore_index = True)
all_data.tail()
all_data['LastName'] = all_data['Name'].apply(lambda x: str.split(x, ',')[0])
all_data['FamilySurvival'] = 0.5
all_data.head(2)
for grp, grp_df in all_data.groupby(['LastName', 'Fare']):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()

            pass_id = row['PassengerId']
            if (smax == 1):
                all_data.loc[all_data['PassengerId'] == pass_id, 'FamilySurvival'] = 1
            elif (smin == 0):
                all_data.loc[all_data['PassengerId'] == pass_id, 'FamilySurvival'] = 0
for grp, grp_df in all_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['FamilySurvival'] == 0) | (row['FamilySurvival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()

                pass_id = row['PassengerId']
                if (smax == 1):
                    all_data.loc[all_data['PassengerId'] == pass_id, 'FamilySurvival'] = 1
                elif (smin == 0):
                    all_data.loc[all_data['PassengerId'] == pass_id, 'FamilySurvival'] = 0

train_df['FamilySurvival'] = all_data['FamilySurvival'][:891]
test_df['FamilySurvival'] = np.array(all_data['FamilySurvival'][891:])
test_df['FamilySurvival'].unique()
train_df[['isAlone', 'Survived']].groupby('isAlone').mean()
for dataset in combine:
    dataset.drop(['SibSp', 'Parch', 'FamilySize', 'PassengerId', 'Ticket', 'Cabin'], axis = 1, inplace = True)
train_df.head()
train_df.tail()
pd.crosstab(train_df['Pclass'], train_df['Survived'], margins = True)
plt.hist(train_df['Fare'], bins = 20)
train_df['FareGroup'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareGroup', 'Survived']].groupby('FareGroup')['Survived'].mean()
for dataset in combine:
    dataset['FareGroup'] = pd.qcut(dataset['Fare'], 4)
    dataset.drop('Fare', axis = 1, inplace = True)
train_df.head()
pd.crosstab([train_df['Pclass'], train_df['Sex']], train_df['Survived'], margins = True)
graph = sns.FacetGrid(train_df, row = 'Pclass', col = 'Survived', size = 2.2, aspect=1.6)
graph.map(sns.countplot, 'Sex')
g = sns.FacetGrid(train_df, row = 'Pclass', col = 'Survived', size = 2.2, aspect = 1.6)
g.map(plt.hist, 'Age', bins = 20)
g.add_legend()
g = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex', size = 2.2, aspect = 1.6)
g.map(plt.hist, 'Age', bins = 20)
train_df.head()
f, ax = plt.subplots(1, 2, figsize = (18, 8))
sns.countplot(data = train_df[train_df['Survived'] == 0], x = 'Pclass', hue = 'Sex', ax = ax[0])
sns.countplot(data = train_df[train_df['Survived'] == 1], x = 'Pclass', hue = 'Sex', ax = ax[1])
plt.legend()
for dataset in combine:
    dataset['Initial'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
pd.crosstab(train_df['Sex'], train_df['Initial'])
for dataset in combine:
    dataset['Initial'] = dataset['Initial'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Countess', 'Dona'], 'Others')
    dataset['Initial'] = dataset['Initial'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Initial'] = dataset['Initial'].replace(['Lady', 'Mme'], 'Mrs')
train_df['Initial'].unique()
train_df.head()
train_df.groupby('Initial')['Age'].mean()
for dataset in combine:
    dataset.loc[dataset['Age'].isnull() & (dataset['Initial'] == 'Master'), 'Age'] = 5
    dataset.loc[dataset['Age'].isnull() & (dataset['Initial'] == 'Miss'), 'Age'] = 21
    dataset.loc[dataset['Age'].isnull() & (dataset['Initial'] == 'Mr'), 'Age'] = 32
    dataset.loc[dataset['Age'].isnull() & (dataset['Initial'] == 'Mrs'), 'Age'] = 36
    dataset.loc[dataset['Age'].isnull() & (dataset['Initial'] == 'Others'), 'Age'] = 45
for dataset in combine:
    print(dataset.isnull().any())
train_df.head()
train_df.hist('Age')
train_df['AgeGroup'] = pd.qcut(train_df['Age'], 4)
train_df[['AgeGroup', 'Survived']].groupby('AgeGroup')['Survived'].mean()
for dataset in combine:
    dataset['Person'] = dataset['Sex']
#     dataset.loc[dataset['Initial'] == 'Master', 'Person'] = 'Child'
for dataset in combine:
    dataset['AgeGroup'] = pd.qcut(dataset['Age'], 4)
    dataset.drop(['Age', 'Name', 'Sex'], axis = 1, inplace = True)
train_df.head()
# Label Encoder
for dataset in combine:
    encoder = LabelEncoder()
    dataset['Embarked'] = encoder.fit_transform(dataset['Embarked'])
    dataset['FareGroup'] = encoder.fit_transform(dataset['FareGroup'])
    dataset['Initial'] = encoder.fit_transform(dataset['Initial'])
    dataset['AgeGroup'] = encoder.fit_transform(dataset['AgeGroup'])
    dataset['Person'] = encoder.fit_transform(dataset['Person'])
train_df.head()
sns.heatmap(train_df.corr(), annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)
test_df.head()
X = train_df.drop('Survived', axis = 1)
y = train_df['Survived']
X_res = test_df
X_res.head()
X.head()
X_Pclass = pd.get_dummies(X['Pclass'], drop_first = True, prefix='Pclass')
X_Embarked = pd.get_dummies(X['Embarked'], drop_first = True, prefix = 'Embarked')
X_FareGroup = pd.get_dummies(X['FareGroup'], drop_first = True, prefix='FareGroup')
X_Initial = pd.get_dummies(X['Initial'], drop_first = True, prefix='Initial')
X_AgeGroup = pd.get_dummies(X['AgeGroup'], drop_first = True, prefix='AgeGroup')
X_Person = pd.get_dummies(X['Person'], drop_first = True, prefix='Person')
X.drop(['Pclass', 'Embarked', 'FareGroup', 'Initial', 'AgeGroup', 'Person'], axis = 1, inplace = True)
X = X.join([X_Pclass, X_Embarked, X_FareGroup, X_Initial, X_AgeGroup, X_Person])
X.head()
X_res_Pclass = pd.get_dummies(X_res['Pclass'], drop_first = True, prefix='Pclass')
X_res_Embarked = pd.get_dummies(X_res['Embarked'], drop_first = True, prefix = 'Embarked')
X_res_FareGroup = pd.get_dummies(X_res['FareGroup'], drop_first = True, prefix='FareGroup')
X_res_Initial = pd.get_dummies(X_res['Initial'], drop_first = True, prefix='Initial')
X_res_AgeGroup = pd.get_dummies(X_res['AgeGroup'], drop_first = True, prefix='AgeGroup')
X_res_Person = pd.get_dummies(X_res['Person'], drop_first = True, prefix='Person')
X_res.drop(['Pclass', 'Embarked', 'FareGroup', 'Initial', 'AgeGroup', 'Person'], axis = 1, inplace = True)
X_res = X_res.join([X_res_Pclass, X_res_Embarked, X_res_FareGroup, X_res_Initial, X_res_AgeGroup, X_res_Person])
X_res.head()
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, random_state = 101, test_size = 0.3)
print(X_res.shape)
# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = SVC()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = LinearSVC()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = GaussianNB()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
model = xgb.XGBClassifier()
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
k_index = list(range(1, 11, 1))
k_error = []
for i in range(1, 11, 1):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X, y)
    predictions = model.predict(X_test)
    k_error.append(1 - accuracy_score(y_test, predictions))

plt.plot(k_index, k_error)
model = KNeighborsClassifier(n_neighbors = 6)
model.fit(X, y)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
# cross-valfrom sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifiers = ['Logistic Regression', 'SVC', 'LinearSVC', 'KNeighborsClassifier', 'DecisionTree', 'RandomForest', 'GaussianNB', 'XGBoost']
models = [LogisticRegression(), SVC(), LinearSVC(), KNeighborsClassifier(n_neighbors = 7), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100), GaussianNB(), xgb.XGBClassifier()]
mean_score = []

for i in range(len(classifiers)):
    model = models[i]
    cv_results = cross_val_score(model, X, y, cv = 10, scoring = 'accuracy')
    cv_results = cv_results.mean()
    mean_score.append(cv_results)

cv_df = pd.DataFrame({'Classifiers': classifiers, 'CV Score': mean_score})
cv_df
# Grid Search on Linear SVC
model = LinearSVC()
C = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grid_parameters = {'C': C}
gm = GridSearchCV(model, grid_parameters)
gm.fit(X, y)
print(gm.best_score_)
print(gm.best_params_)
# Grid Search on Kernel SVC
model = SVC()
C = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grid_parameters = {'C': C, 'gamma': gamma}
gm = GridSearchCV(model, grid_parameters)
gm.fit(X, y)
print(gm.best_score_)
print(gm.best_params_)
# Grid Search on RandomForest
model = RandomForestClassifier()
n_estimators = list(range(100, 1000, 100))
grid_parameters = {'n_estimators': n_estimators}
gm = GridSearchCV(model, grid_parameters)
gm.fit(X, y)
print(gm.best_score_)
print(gm.best_params_)
# Grid Search on XGBoost
model = xgb.XGBClassifier()
learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_estimators = list(range(100, 1000, 100))
grid_parameters = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
gm = GridSearchCV(model, grid_parameters)
gm.fit(X, y)
print(gm.best_score_)
print(gm.best_params_)
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(X,y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(X_test,y_test))
cross=cross_val_score(ensemble_lin_rbf,X,y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(X, y)
prediction=model.predict(X_test)
print('The accuracy for bagged KNN is:',accuracy_score(prediction,y_test))
result=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(X,y)
prediction=model.predict(X_test)
print('The accuracy for bagged Decision Tree is:',accuracy_score(prediction,y_test))
result=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(X, y)
predictions = final_model.predict(X_test)
print(accuracy_score(y_test, predictions))
y_pred = final_model.predict(X_res)
# Submission
submission = {'PassengerId': final_res_data['PassengerId'], 'Survived': y_pred}
submission_df = pd.DataFrame(submission)
submission_df.head()
submission_df.to_csv('gender_submission_final_10.csv', index=False)
