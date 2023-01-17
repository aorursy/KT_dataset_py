%matplotlib notebook



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
df_train.describe()
df_train.head()
df_test.describe()
df_train['Cabin'] = df_train['Cabin'].str.get(0).fillna('Z')

df_test['Cabin'] = df_test['Cabin'].str.get(0).fillna('Z')



df_train['Sex'] = (df_train['Sex'] == 'female').astype(int)

df_test['Sex'] = (df_test['Sex'] == 'female').astype(int)



df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

df_test['Age'] = df_test['Age'].fillna(df_train['Age'].median())



df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mean())
ax1 = plt.subplot(3, 2, 1)

plt.hist(x = [df_train[df_train['Survived']==1]['Age'], df_train[df_train['Survived']==0]['Age']], 

         color = ['lightgreen','orangered'],label = ['Survived','Dead'], log=True)

plt.xlabel('Age')



plt.subplot(3, 2, 2, sharey=ax1)

plt.hist(x = [df_train[df_train['Survived']==1]['Sex'], df_train[df_train['Survived']==0]['Sex']], 

         color = ['lightgreen','orangered'], alpha=0.8, label = ['Survived','Dead'], log=True)

plt.xlabel('Sex')



plt.subplot(3, 2, 3, sharey=ax1)

plt.hist(x = [df_train[df_train['Survived']==1]['Pclass'], df_train[df_train['Survived']==0]['Pclass']], 

         color = ['lightgreen','orangered'], alpha=0.8, label = ['Survived','Dead'], log=True)

plt.xlabel('Pclass')



plt.subplot(3, 2, 4, sharey=ax1)

plt.hist(x = [df_train[df_train['Survived']==1]['Fare'], df_train[df_train['Survived']==0]['Fare']], 

         color = ['lightgreen','orangered'], alpha=0.8, label = ['Survived','Dead'], log=True)

plt.xlabel('Fare')



plt.subplot(3, 2, 5, sharey=ax1)

plt.hist(x = [df_train[df_train['Survived']==1]['SibSp'], df_train[df_train['Survived']==0]['SibSp']], 

         color = ['lightgreen','orangered'], alpha=0.8, label = ['Survived','Dead'], log=True)

plt.xlabel('SibSp')



plt.subplot(3, 2, 6, sharey=ax1)

plt.hist(x = [df_train[df_train['Survived']==1]['Parch'], df_train[df_train['Survived']==0]['Parch']], 

         color = ['lightgreen','orangered'], alpha=0.8, label = ['Survived','Dead'], log=True)

plt.xlabel('Parch')
import matplotlib.gridspec as gridspec



fare_dt = df_train['Fare'].copy().sort_values().reset_index()['Fare']



plt.figure()

gspec = gridspec.GridSpec(3, 3)



side_histogram = plt.subplot(gspec[0:, 0])

lower_right = plt.subplot(gspec[0:, 1:])



lower_right.plot(fare_dt)

s = side_histogram.hist(fare_dt, bins=50, orientation='horizontal', normed=True)

side_histogram.invert_xaxis()
df_train['Fare'] = df_train['Fare'].apply(lambda x: np.log(x) if x > 0 else 2)

df_test['Fare'] = df_test['Fare'].apply(lambda x: np.log(x) if x > 0 else 2)
df_train['Title'] = df_train['Name'].str.extract('(,.*\.)', expand=False).str.replace(',', '').str.strip()

df_test['Title'] = df_test['Name'].str.extract('(,.*\.)', expand=False).str.replace(',', '').str.strip()

df_train['Title'].unique()
df_test['Title'].unique()
df_train.loc[513, 'Title'] = 'Mrs.'

df_train['Title'].unique()
if 0:

    title_map = {'Mr.': 1, 'Mrs.': 1, 'Miss.': 1, 'Master.': 1, 'Don.': 2, 'Rev.': 2, 'Dr.': 2, 'Mme.': 1,

           'Ms.': 1, 'Major.': 2, 'Lady.': 3, 'Sir.': 3, 'Mlle.': 3, 'Col.': 3, 'Capt.': 3,

           'the Countess.': 3, 'Jonkheer.': 3, 'Col.': 2, 'Dona.': 2}



    df_train['Title'] = df_train['Title'].apply(lambda x: title_map[x])



    df_test['Title'] = df_test['Title'].apply(lambda x: title_map[x])



    print(df_train['Title'].unique())

    print(df_test['Title'].unique())
basefeatures = ['Pclass', 'Sex', 'Age', 'Title', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Cabin']



base_train = pd.get_dummies(df_train[basefeatures], columns = ['Title', 'Embarked', 'Cabin'])

finaltest = pd.get_dummies(df_test[basefeatures], columns = ['Title', 'Embarked', 'Cabin'])

finaltest.head()
missing_cols = set( base_train.columns ) - set( finaltest.columns )

print("Missing columns: " + str(missing_cols) + "\n")

for c in missing_cols:

    print("Adding "+str(c))

    finaltest[c] = 0
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(base_train,

                                                    df_train['Survived'], random_state=0)
from sklearn.preprocessing import MinMaxScaler



# for manual train-test-split

scaler = MinMaxScaler()

X_train[:] = scaler.fit_transform(X_train)

X_test[:] = scaler.transform(X_test)



# for use with gridsearch

scaler2 = MinMaxScaler()

X_gridsearch = base_train.copy()

X_gridsearch[:] = scaler2.fit_transform(X_gridsearch)

y_gridsearch = df_train['Survived']



# for prediction with final test data

finaltest = finaltest[X_train.columns]

finaltest[:] = scaler2.transform(finaltest)
from sklearn.dummy import DummyClassifier



dc = DummyClassifier(strategy = 'most_frequent')

dc.fit(X_train, y_train)



print("Score of dummy classifier: " + str(dc.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



tuned_parameters = [{'C': np.arange(0.01, 0.5, 0.005)}]



clf = GridSearchCV(LogisticRegression(max_iter=100000), tuned_parameters, cv=5,

                   scoring='accuracy')

clf.fit(X_gridsearch, y_gridsearch)



print('Best parameter: ' + str(clf.best_params_))

print("Score of logistic regression: " + str(clf.best_score_))
from sklearn import tree



dt = tree.DecisionTreeClassifier()

dt = dt.fit(X_train, y_train)

dt.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier



tuned_parameters = [{'n_estimators': [25, 30, 35] ,

                        'max_depth': range(4, 7), 

                        'max_features': [x for x in range (5, 12)]}]



rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,

                   scoring='accuracy')

rf.fit(X_gridsearch, y_gridsearch)



print('Best parameter: ' + str(rf.best_params_))

print("Score of Random Forest: " + str(rf.best_score_))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV



tuned_parameters = [{'n_estimators': [18, 20, 22, 23, 24, 25, 26, 27] ,

                    'max_depth': range(2, 6)}]



gbdt = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='accuracy')

gbdt.fit(X_gridsearch, y_gridsearch)

print('Best parameter: ' + str(gbdt.best_params_))    

print('Score of Gradient Boosting: ' + str(gbdt.best_score_))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



tuned_parameters = [{'n_neighbors': range(2, 8)}]



knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')

knn.fit(X_gridsearch, y_gridsearch)

print('Best parameter: ' + str(knn.best_params_))    

print('Score of Neares neigbors: ' + str(knn.best_score_))
from sklearn.neural_network import MLPClassifier



nnclf = MLPClassifier(hidden_layer_sizes = [1000, 500, 100, 10, 5], solver='adam',

                     random_state = 0, max_iter=10000, alpha=0.00001).fit(X_train, y_train)

nnclf.score(X_test, y_test)
from sklearn import metrics



pred_dummy = dc.predict_proba(X_test)[:,1]



fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_dummy)

print("Dummy " + str(metrics.auc(fpr, tpr)))



pred_logreg = clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_logreg)

print("LogReg " + str(metrics.auc(fpr, tpr)))



pred_knn = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_knn)

print("KNN " + str(metrics.auc(fpr, tpr)))



pred_dt = dt.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_dt)

print("Decision tree " + str(metrics.auc(fpr, tpr)))



pred_rf = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_rf)

print("Random forest " + str(metrics.auc(fpr, tpr)))



pred_gbdt = gbdt.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_gbdt)

print("Gradient boosted dt " + str(metrics.auc(fpr, tpr)))



pred_nnclf = nnclf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_nnclf)

print("Neural network " + str(metrics.auc(fpr, tpr)))
pred = rf.predict(finaltest)

final = pd.DataFrame(pred, index=df_test['PassengerId'])

final.columns = ['Survived']

final.to_csv('titanic_result.csv')