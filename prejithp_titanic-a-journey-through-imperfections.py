import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head(10)
train_df.shape
test_df.shape
train_df.dtypes
train_df.isna().sum()
train_df['Cabin'].nunique()
train_df['Embarked'].unique()
train_df['Embarked'].nunique()
sns.distplot(train_df['Age'].dropna())
train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
train_df = train_df[train_df.Embarked.notna()]
train_df.head()
passengerList = test_df['PassengerId']
test_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
test_df.head()
X = train_df.loc[:, train_df.columns != 'Survived']
y = train_df.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y)
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train['Age'].to_numpy().reshape(-1, 1))

X_train['Age'] = imputer.transform(X_train['Age'].to_numpy().reshape(-1, 1))
X_test['Age'] = imputer.transform(X_test['Age'].to_numpy().reshape(-1, 1))

test_df['Age'] = imputer.transform(test_df['Age'].to_numpy().reshape(-1, 1))
X_train['Age'] = X_train['Age'].round()
X_test['Age'] = X_test['Age'].round()

X_train['Fare'] = X_train['Fare'].round()
X_test['Fare'] = X_test['Fare'].round()

test_df['Age'] = test_df['Age'].round()
test_df['Fare'] = test_df['Age'].round()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
test_df = pd.get_dummies(test_df)
gridParameters = {'n_estimators': [1, 5, 10, 50, 100],
                 'max_depth': [None, 1, 5, 10, 50, 100]}

model = RandomForestClassifier()
clf = GridSearchCV(model, gridParameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
#Function to run different classifications algorithms. Returns the clf object of the classifier that gave highest accuracy

def getBestClassifier(X_train, y_train, X_test, y_test):    
    classifierList = {
        'SVM': SVC(),
        'Neural Network': MLPClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    classifierParams = {
        'SVM': {'C': [0.01, 0.1, 1, 10, 100],
               'kernel': ['linear', 'rbf', 'sigmoid']},
        'Neural Network': {'activation': ['identity', 'logistic', 'tanh', 'relu']},
        'Random Forest': {'n_estimators': [1, 5, 10, 50, 100],
                     'max_depth': [None, 1, 5, 10, 50, 100]}
    }
    
    fittedClassifiersParam = {}
    
    for key, classifier in classifierList.items():
        clf = GridSearchCV(classifier, classifierParams[key])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fittedClassifiersParam[key] = [accuracy_score(y_test, y_pred), clf.best_estimator_]
        print('Accuracy of {0:20s}: {1}'.format(key, str(accuracy_score(y_test, y_pred))))
    
    return fittedClassifiersParam[sorted(fittedClassifiersParam, key = lambda k : fittedClassifiersParam[k][0], reverse=True)[0]]
#Used to plot a confusion matrix

def confusionMatrix(y_test, y_pred):
    fig, ax = plt.subplots()
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat, annot = True, fmt='d')

    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    ax.set_ylim([0, 2])
bestClassifier = getBestClassifier(X_train, y_train, X_test, y_test)
bestClassifier
confusionMatrix(y_test, bestClassifier[1].predict(X_test))
if bestClassifier[1].__class__.__name__ == 'MLPClassifier':
    print("Feature Importance not available for the model chosen: " + str(bestClassifier[1].__class__.__name__) )
else:
    plt.figure(figsize =(12, 6))
    plt.title(bestClassifier[1].__class__.__name__)
    sns.barplot(x=X_train.columns, y=bestClassifier[1].feature_importances_)
    plt.xticks(rotation=45, horizontalalignment='right')
X_train['MemCount'] =  X_train['SibSp'] + X_train['Parch']
X_test['MemCount'] =  X_test['SibSp'] + X_test['Parch']
test_df['MemCount'] =  test_df['SibSp'] + test_df['Parch']

X_train.drop(['SibSp', 'Parch'], inplace=True, axis=1)
X_test.drop(['SibSp', 'Parch'], inplace=True, axis=1)
test_df.drop(['SibSp', 'Parch'], inplace=True, axis=1)

X_train['isAlone'] = X_train['MemCount'].apply(lambda x: 1 if x > 0 else 0)
X_test['isAlone'] = X_test['MemCount'].apply(lambda x: 1 if x > 0 else 0)
test_df['isAlone'] = test_df['MemCount'].apply(lambda x: 1 if x > 0 else 0)

X_train['Age'] = pd.cut(X_train['Age'], 4, labels=[1, 2, 3, 4])
X_test['Age'] = pd.cut(X_test['Age'], 4, labels=[1, 2, 3, 4])
test_df['Age'] = pd.cut(test_df['Age'], 4, labels=[1, 2, 3, 4])

X_train['Fare'] = pd.cut(X_train['Fare'], 4, labels=[1, 2, 3, 4])
X_test['Fare'] = pd.cut(X_test['Fare'], 4, labels=[1, 2, 3, 4])
test_df['Fare'] = pd.cut(test_df['Fare'], 4, labels=[1, 2, 3, 4])
X_train['Age'] = X_train['Age'].astype('int')
X_train['Fare'] = X_train['Fare'].astype('int')

X_test['Age'] = X_test['Age'].astype('int')
X_test['Fare'] = X_test['Fare'].astype('int')

test_df['Fare'] = test_df['Fare'].astype('int')
test_df['Age'] = test_df['Age'].astype('int')
bestClassifier = getBestClassifier(X_train, y_train, X_test, y_test)
confusionMatrix(y_test, bestClassifier[1].predict(X_test))
if bestClassifier[1].__class__.__name__ in ['MLPClassifier', 'SVC']:
    print("Feature Importance not available for the model chosen: " + str(bestClassifier[1].__class__.__name__) )
else:
    plt.figure(figsize =(12, 6))
    sns.barplot(x=X_train.columns, y=bestClassifier[1].feature_importances_)
    plt.xticks(rotation=45, horizontalalignment='right')
titleTrainDf = pd.read_csv('../input/titanic/train.csv')
titleTestDf = pd.read_csv('../input/titanic/test.csv')
titleTrainDf = titleTrainDf.filter(['Name'])
titleTestDf = titleTestDf.filter(['Name'])
trainTitleSeries = titleTrainDf['Name'].str.split(", ").apply(lambda x: x[1]).str.split(".").apply(lambda x : x[0])
trainTitleSeries = trainTitleSeries.rename('Title')

testTitleSeries = titleTestDf['Name'].str.split(", ").apply(lambda x: x[1]).str.split(".").apply(lambda x : x[0])
testTitleSeries = testTitleSeries.rename('Title')
trainTitleSeries.value_counts()
# Function to map the names to appropriate titles

def classifyTitles(x):
    return 'Rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x
trainTitleSeries = trainTitleSeries.apply(classifyTitles)
testTitleSeries = testTitleSeries.apply(classifyTitles)
X_train = X_train.merge(trainTitleSeries, left_index=True, right_index=True)
X_test = X_test.merge(trainTitleSeries, left_index=True, right_index=True)

test_df = test_df.merge(testTitleSeries, left_index=True, right_index=True)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
test_df = pd.get_dummies(test_df)
classifierList = {
    'SVM': SVC(),
    'Neural Network': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(silent=1, verbose_eval=False),
    'CatBoost': CatBoostClassifier(logging_level='Silent'),
    'LightGBM': lgb.LGBMClassifier()
}

gridEstimatorCount = [1, 5, 10, 50, 100]
gridMaxDepth = [1, 2, 4, 5, 8, 10]
gridLearningRate = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]

classifierParams = {
    'SVM': {'C': [0.01, 0.1, 1, 10, 100],
           'kernel': ['linear', 'rbf', 'sigmoid']},
    'Neural Network': {'activation': ['identity', 'logistic', 'tanh', 'relu']},
    'Random Forest': {'n_estimators': gridEstimatorCount,
                 'max_depth': gridMaxDepth},
    'XGBoost': {'learning_rate': gridLearningRate, 
            'max_depth': gridMaxDepth, 
            'n_estimators': gridEstimatorCount},
    'CatBoost': {'n_estimators': gridEstimatorCount,
                'max_depth': gridMaxDepth},
    'LightGBM': {'n_estimators': gridEstimatorCount,
                'max_depth': gridMaxDepth}
}
# Create an ensemble and return the ensemble object

def createEnsemble(X_train, y_train, X_test=[], y_test=[], classifierList={}, isFullDataset=False):        
    fittedClassifiers = {}
    
    if not classifierList:
        return
    
    for key, classifier in classifierList.items():
        
        print("Now training: ", key)
        
        clf = GridSearchCV(classifier, classifierParams[key], cv=5, n_jobs=-1, scoring='accuracy')
        clf.fit(X_train, y_train)
        fittedClassifiers.update({key: clf.best_estimator_})

        if not isFullDataset:
            y_pred = clf.predict(X_test)
            print(key + ' has accuracy: ' + str(accuracy_score(y_test, y_pred)))
    
    ensemble = VotingClassifier(estimators=[(k, v) for k, v in fittedClassifiers.items()])
    return ensemble
classifierList = {
    'Neural Network': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'LightGBM': lgb.LGBMClassifier()
}
bestClassifier = createEnsemble(X_train, y_train, X_test, y_test, classifierList=classifierList)
finalTrainingDataset = pd.concat([X_train, X_test])
finalTargetDataset = pd.concat([y_train, y_test])

ensemble = createEnsemble(finalTrainingDataset, finalTargetDataset, classifierList=classifierList, isFullDataset=True)

results = ensemble.fit(finalTrainingDataset, finalTargetDataset).predict(test_df)
pd.concat([passengerList, pd.Series(results, name='Survived')], axis=1).to_csv('080620_final_ensemble.csv', index=False)