import pandas as pd

import pandas_profiling as pp

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import os



# Metrics

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Validation

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.pipeline import Pipeline



# Tuning

from sklearn.model_selection import GridSearchCV



# Feature Extraction

from sklearn.feature_selection import RFE



# Preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer



# Models

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



# Ensembles

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



warnings.filterwarnings('ignore')

%matplotlib inline



sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')



df_train = pd.read_csv('../input/learn-together/train.csv')

df_test = pd.read_csv('../input/learn-together/test.csv')











for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train.head()
print('Train size: ',df_train.shape)

print('Test size: ', df_test.shape)
df_train.columns
df_train.info() #you can check the dtypes using this method df_train.dtypes or df_train.info()
df_train.isna().sum()
df_train.duplicated().sum()
df_train.describe()
colormap = plt.cm.RdBu

plt.figure(figsize=(50,35))

plt.title('Pearson Correlation of Features', y=1.05, size=50)

sns.heatmap(df_train.corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
df_train.hist(figsize=(20,30));
target = df_train.Cover_Type.value_counts()

sns.countplot(x='Cover_Type', data=df_train)

plt.title('Class Distribution');

print(target)
pp.ProfileReport(df_train)
lst = ['Id', 'Cover_Type']



X = df_train.drop(lst, axis=1)

y = df_train.Cover_Type

test_X = df_test.drop('Id', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,

                                                   y,

                                                   test_size=0.3,

                                                   random_state=0)



print('X_train: ',X_train.shape)

print('X_test: ',X_test.shape)

print('y_train: ',y_train.shape)

print('y_test: ',y_test.shape)
from sklearn.model_selection import KFold

models = []

models.append(( ' LR ' , LogisticRegression()))

models.append(( ' LDA ' , LinearDiscriminantAnalysis()))

models.append(( ' KNN ' , KNeighborsClassifier()))

models.append(( ' NB ' , GaussianNB()))

models.append(( ' SVM ' , SVC()))



results = []

names = []



for name, model in models:

    Kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());

    print(msg)
from sklearn.model_selection import KFold

models = []

models.append(( 'Adab' , AdaBoostClassifier()))

models.append(( 'Bagging' , BaggingClassifier()))

models.append(( 'GBC' , GradientBoostingClassifier()))

models.append(( 'RF' , RandomForestClassifier()))





results = []

names = []



for name, model in models:

    Kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());

    print(msg)
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)





model = RandomForestClassifier()

param_grid = { 

    'n_estimators': [10,20,50,100],

    'max_features': ['auto', 'sqrt', 'log2']

}



kfold = KFold(n_splits=10, random_state=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)

grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))







scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test =scaler.transform(X_test)

RF = RandomForestClassifier(n_estimators=100, max_features='auto').fit(X_train,y_train)

y_pred = RF.predict(test_X)





y_pred_1 = RF.predict(test_X)

sub = pd.DataFrame({'ID': df_test.Id,

                       'TARGET': y_pred_1})

sub.to_csv('submission.csv')