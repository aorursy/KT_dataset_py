# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# print(os.listdir('input'))

# Any results you write to the current directory are saved as output.
import matplotlib as mlp
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc
import pickle
traindf = pd.read_csv("../input/train.csv")
# traindf = pd.read_csv('input/train.csv')
traindf.head(10)
traindf.tail(10)
traindf.describe()
traindf.describe(include = 'object')
traindf.info()
traindf.hist(figsize=(10,10), yrot=-45)
plt.show()
sns.countplot(y='Sex', data=traindf)
plt.show()
sns.countplot(y='Embarked', data=traindf)
plt.show()
sns.violinplot(y='Survived', x='Embarked', data=traindf)
plt.grid()
plt.show()
sns.lmplot(y='Age', x='Fare', hue='Survived', data=traindf)
plt.show()
traindf.head()
traindf = traindf.drop(['Name', 'Ticket', 'Cabin'], axis=1)
traindf.head()
print(traindf.shape)
print(traindf.drop_duplicates().shape)
traindf.select_dtypes(include=['object']).isnull().sum()
traindf['Embarked'] = traindf['Embarked'].fillna('M')
traindf.select_dtypes(exclude=['object']).isnull().sum()
traindf['Age_missing'] = traindf.Age.isnull().astype(int)
traindf.Age.fillna(0, inplace=True)
traindf.select_dtypes(exclude=['object']).isnull().sum()
traindf.head()
traindf.Embarked.replace(['Q', 'M'], 'O', inplace=True)
traindf = pd.get_dummies(traindf, columns=['Sex', 'Embarked'])
traindf.head()
y = traindf.Survived
X = traindf.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=808)
print(len(X_train), len(X_test), len(y_train), len(y_test))
pipelines = {
    'l1' : make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', random_state=808)),
    'l2' : make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=808)),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier(random_state=808)),
    'gb' : make_pipeline(GradientBoostingClassifier(random_state=808))
}
for key, value in pipelines.items():
    print( key, type(value) )
l1_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10)
}

l2_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10)
}

enet_hyperparameters = {
    'elasticnet__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

rf_hyperparameters = {
    'randomforestclassifier__max_features' : ['auto', 'sqrt', 0.33],
    'randomforestclassifier__n_estimators' : [100, 200]
}

gb_hyperparameters = {
    'gradientboostingclassifier__max_features' : ['auto', 'sqrt', 0.33],
    'gradientboostingclassifier__n_estimators' : [100, 200],
    'gradientboostingclassifier__learning_rate' : [0.05, 0.1, 0.2]
}

hyperparameters = {
    'l1' : l1_hyperparameters,
    'l2' : l2_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}
for key in ['l1', 'l2', 'gb', 'rf']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipelines[name], hyperparameters[name], cv=10, n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted')
for key, value in fitted_models.items():
    print( key, type(value) )
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    print(name, auc(fpr, tpr))
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['l1'].best_estimator_, f)
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)
model
submission_df = pd.read_csv("../input/test.csv")
# submission_df = pd.read_csv('input/test.csv')
submission_df.head()
def clean_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Embarked'] = df['Embarked'].fillna('M')
    df['Age_missing'] = df.Age.isnull().astype(int)
    df.Age.fillna(0, inplace=True)
    df.Fare.fillna(0, inplace=True)
    return df
cleaned_submission = clean_data(submission_df)
cleaned_submission.head()
def engineer_features(df):
    df.Embarked.replace(['Q', 'M'], 'O', inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    return df
augmented_submission = engineer_features(cleaned_submission)
augmented_submission.head()
pred = model.predict(augmented_submission)
print(pred[:5])
final_submission = pd.DataFrame({'PassengerId' : augmented_submission.PassengerId,
                                 'Survived' : pred})
final_submission.head()
final_submission.to_csv("final_submission.csv", index=None)
#final_submission.to_csv('input/final_submission.csv')
