# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

# data mining
#from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn import impute
from sklearn_pandas import categorical_imputer, CategoricalImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# machine learning
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

## scikit modeling libraries
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier,
                             VotingClassifier)

from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)

## Load metrics for predictive modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE, rfe
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Save the model
import pickle
import joblib


## Warnings and other tools
import itertools
import warnings
warnings.filterwarnings("ignore")
# Carga de los ficheros anteriores ya preprocesados - cambiar la ruta según vuestro directorio o raíz
train_titanic = pd.read_csv('../input/titanic-eda-part-1/train_eda.csv')
test_titanic = pd.read_csv('../input/titanic-eda-part-1/test_eda.csv')
train_titanic.head(10)
test_titanic.head(10)
PassengerId = test_titanic['PassengerId']
X = train_titanic
Y = test_titanic.drop('PassengerId', axis=1).copy()
PassengerId
Y
# combinamos train and test en un solo dataframe
dataset = [X, Y]
col_Z = ['SibSp', 'Parch', 'FamilySize']
features = dataset[0][col_Z]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
dataset[0][col_Z] = features
dataset[0].head(10)
features = dataset[1][col_Z]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
dataset[1][col_Z] = features
dataset[1].head(10)
cols = ['Pclass', 'Embarked', 'FareGroup', 'AgeGroup', 'Title', 'Deck', 'RoomGroup']
titanic_categorical = dataset[0][cols]
titanic_categorical = pd.concat([pd.get_dummies(titanic_categorical[col], prefix=col) for col in titanic_categorical], axis=1)
titanic_categorical.head()
dataset[0] = pd.concat([dataset[0][dataset[0].columns[~dataset[0].columns.isin(cols)]], titanic_categorical], axis=1)
dataset[0].head()
titanic_categorical = dataset[1][cols]
titanic_categorical = pd.concat([pd.get_dummies(titanic_categorical[col], prefix=col) for col in titanic_categorical], axis=1)
dataset[1] = pd.concat([dataset[1][dataset[1].columns[~dataset[1].columns.isin(cols)]], titanic_categorical], axis=1)
dataset[1].head()
# We remove Deck_T from dataset0
dataset[0].columns.tolist()
dataset[1].columns.tolist()
# Eliminamos `Deck_T` por estar incluida en testing
dataset[0] = dataset[0].drop('Deck_T', axis=1).copy()
dataset[0]
dataset[0].dtypes
dataset[0].dtypes
# Creación de la matriz de correlación
k = 16 #number of variables for heatmap
corrmat = dataset[0].corr(method='spearman')
# picking the top 15 correlated features
cols15 = corrmat.nlargest(k, 'Survived')['Survived'].index
# Show 15 features with most correlation ratio - Pearson
corr = dataset[0].corr(method='pearson')
print (corr['Survived'].sort_values(ascending=False)[:15], '\n')
print (corr['Survived'].sort_values(ascending=False)[-5:])
# Show 15 features with most correlation ratio - Kendall
corr = dataset[0].corr(method='kendall')
print (corr['Survived'].sort_values(ascending=False)[:15], '\n')
print (corr['Survived'].sort_values(ascending=False)[-5:])
# Show 15 features with most correlation ratio - Spearman
corr = dataset[0].corr(method='spearman')
print (corr['Survived'].sort_values(ascending=False)[:15], '\n')
print (corr['Survived'].sort_values(ascending=False)[-5:])
X_feat = dataset[0].drop('Survived', axis=1).copy()
y_feat = dataset[0]['Survived']
X_feat
features = X_feat.columns.tolist()
# Get k
n = 17

# Create model by Logistic Regression and calculate RMSE
lr = LogisticRegression(n_jobs=4, verbose=2)
rfe = RFE(lr, n, verbose=2, )
rfe = rfe.fit(X_feat, y_feat.values.ravel())
rfe.estimator_
rfe.n_features_
# Print Support and Ranking
print(rfe.support_)
print(rfe.ranking_)
z=zip(features, rfe.support_, rfe.ranking_)
list(z)
colsRNK10 = []
for v, s, r in zip(features, rfe.support_, rfe.ranking_):
    if r >=1 and r <=10:
        colsRNK10.append(v)
colsRNK15 = []
for v, s, r in zip(features, rfe.support_, rfe.ranking_):
    if r >=1 and r <=15:
        colsRNK15.append(v)
colsRNK20 = []
for v, s, r in zip(features, rfe.support_, rfe.ranking_):
    if r >=1 and r <=20:
        colsRNK20.append(v)
colsSPT = []
for v, s, r in zip(features, rfe.support_, rfe.ranking_):
    if s == True:
        colsSPT.append(v)
print(len(colsRNK10)) # Are the features selected by Ranking10
print(len(colsRNK15)) # Are the features selected by Ranking15
print(len(colsRNK20)) # Are the features selected by Ranking20

print(len(colsSPT)) # Are the features selected by Support
# Show all Features selected
print(cols15)
print("*-*"*20)
print(colsRNK10)
print("*-*"*20)
print(colsRNK15)
print("*-*"*20)
print(colsRNK20)
print("*-*"*20)
print(colsSPT)
# Eliminamos Survived 
cols15 = cols15.drop('Survived')
# Save dataset0 and dataset1 for next step: Modeling
dataset[0].to_csv('/kaggle/working/train_feat.csv', index=False)
dataset[1].to_csv('/kaggle/working/test_feat.csv', index=False)
# Guardamos PassengerID que también podemos recuperar del fichero original test['PassengerId']
# PassengerId = test_titanic['PassengerId']
# Procederemos con guardar las columnas como Series y utilziarlas para nuestros modelos
corr_columns = [cols15, colsRNK10, colsRNK15, colsRNK20, colsSPT]
cols_feat = pd.Series(corr_columns)
# Guardamos el listado en formato Pickle
cols_feat.to_pickle('/kaggle/working/cols_feat.pkl')
pd.read_pickle('/kaggle/working/cols_feat.pkl')[0]
pd.read_csv('/kaggle/working/train_feat.csv').head(10)