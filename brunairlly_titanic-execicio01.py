%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import linear_model, svm, gaussian_process

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
df_train_original = pd.read_csv('../input/train.csv')

df_test_original = pd.read_csv('../input/test.csv')
df_train = df_train_original

df_test = df_test_original



df_train = df_train.drop(['Name'], axis=1)

df_test = df_test.drop(['Name'], axis=1)



df_train = df_train.drop(['Cabin'], axis=1)

df_test = df_test.drop(['Cabin'], axis=1)



df_train = df_train.drop(['Ticket'], axis=1)

df_test = df_test.drop(['Ticket'], axis=1)



df_train = df_train.drop(['SibSp'], axis=1)

df_test = df_test.drop(['SibSp'], axis=1)



df_train = df_train.drop(['Parch'], axis=1)

df_test = df_test.drop(['Parch'], axis=1)
all_data = pd.concat((df_train.loc[:,'Sex':'Fare'],

                     df_test.loc[:,'Sex':'Fare']))
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

#all_data = all_data.fillna(0)
X_train = all_data[:df_train.shape[0]]
X_test = all_data[df_train.shape[0]:]
# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.Survived
from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.formula.api as smf

from sklearn.preprocessing import scale

gbr = GradientBoostingRegressor()



gbr.fit(X_train, y)
yhat_Train = gbr.predict(X_train)

yhat_Train
yhat_test = gbr.predict(X_test)
yhat_test
yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')

print (yhat_gbr)
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_gbr = df_test

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr = df_test_gbr.drop(['Pclass'], axis=1)

df_test_gbr = df_test_gbr.drop(['Sex'], axis=1)

df_test_gbr = df_test_gbr.drop(['Age'], axis=1)

df_test_gbr = df_test_gbr.drop(['Fare'], axis=1)

df_test_gbr = df_test_gbr.drop(['Embarked'], axis=1)

df_test_gbr.to_csv('Titanic_GBR.csv', index = False)
# 3) Aplly Logistic Regression Model

from sklearn.linear_model import LogisticRegression



# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y)
yhat_test = logreg.predict(X_test)
yhat_test
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_logreg = df_test

df_test_logreg['Survived'] = yhat_test

df_test_logreg = df_test_logreg.drop(['Pclass'], axis=1)

df_test_logreg = df_test_logreg.drop(['Sex'], axis=1)

df_test_logreg = df_test_logreg.drop(['Age'], axis=1)

df_test_logreg = df_test_logreg.drop(['Fare'], axis=1)

df_test_logreg = df_test_logreg.drop(['Embarked'], axis=1)

df_test_logreg.to_csv('Titanic_LOGREG.csv', index = False)
# 5) Aplly XGBOOST Model

from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(X_train, y)
yhat_test = xgb.predict(X_test)
yhat_test
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_xgboost = df_test

df_test_xgboost['Survived'] = yhat_test

df_test_xgboost = df_test_xgboost.drop(['Pclass'], axis=1)

df_test_xgboost = df_test_xgboost.drop(['Sex'], axis=1)

df_test_xgboost = df_test_xgboost.drop(['Age'], axis=1)

df_test_xgboost = df_test_xgboost.drop(['Fare'], axis=1)

df_test_xgboost = df_test_xgboost.drop(['Embarked'], axis=1)

df_test_xgboost.to_csv('Titanic_XGBOOST.csv', index = False)
# 6) Aplly KNeighbors Model

knn = KNeighborsClassifier()

knn.fit(X_train, y)

knn_test = knn.predict(X_test)
knn_test
# Gerando um CSV para o resultado obtido com o KNeighbors Model:

df_test_knn = df_test

df_test_knn['Survived'] = knn_test

df_test_knn = df_test_knn.drop(['Pclass'], axis=1)

df_test_knn = df_test_knn.drop(['Sex'], axis=1)

df_test_knn = df_test_knn.drop(['Age'], axis=1)

df_test_knn = df_test_knn.drop(['Fare'], axis=1)

df_test_knn = df_test_knn.drop(['Embarked'], axis=1)

df_test_knn.to_csv('Titanic_KNN.csv', index = False)
# Aplly SVC Model

svc = SVC(probability=True)

svc.fit(X_train, y)
svc_test = svc.predict(X_test)
svc_test
# Gerando um CSV para o resultado obtido com o SVC Model:

df_test_svc = df_test

df_test_svc['Survived'] = svc_test

df_test_svc = df_test_svc.drop(['Pclass'], axis=1)

df_test_svc = df_test_svc.drop(['Sex'], axis=1)

df_test_svc = df_test_svc.drop(['Age'], axis=1)

df_test_svc = df_test_svc.drop(['Fare'], axis=1)

df_test_svc = df_test_svc.drop(['Embarked'], axis=1)

df_test_svc.to_csv('Titanic_SVC.csv', index = False)
# Aplly Decision Tree Model

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y)
dtc_test = dtc.predict(X_test)
dtc_test
# Gerando um CSV para o resultado obtido com o Decision Tree Model:

df_test_dtc = df_test

df_test_dtc['Survived'] = dtc_test

df_test_dtc = df_test_dtc.drop(['Pclass'], axis=1)

df_test_dtc = df_test_dtc.drop(['Sex'], axis=1)

df_test_dtc = df_test_dtc.drop(['Age'], axis=1)

df_test_dtc = df_test_dtc.drop(['Fare'], axis=1)

df_test_dtc = df_test_dtc.drop(['Embarked'], axis=1)

df_test_dtc.to_csv('Titanic_DTC.csv', index = False)
# Aplly GaussianNB Model

gnb = GaussianNB()

gnb.fit(X_train, y)
gnb_test = gnb.predict(X_test)
gnb_test
# Gerando um CSV para o resultado obtido com o GaussianNB Model:

df_test_gnb = df_test

df_test_gnb['Survived'] = gnb_test

df_test_gnb = df_test_gnb.drop(['Pclass'], axis=1)

df_test_gnb = df_test_gnb.drop(['Sex'], axis=1)

df_test_gnb = df_test_gnb.drop(['Age'], axis=1)

df_test_gnb = df_test_gnb.drop(['Fare'], axis=1)

df_test_gnb = df_test_gnb.drop(['Embarked'], axis=1)

df_test_gnb.to_csv('Titanic_GNB.csv', index = False)
# Aplly Neural Model

nn = MLPClassifier(hidden_layer_sizes=(100,100,50))

nn.fit(X_train, y)
nn_test = nn.predict(X_test)
nn_test
# Gerando um CSV para o resultado obtido com o Neural Model:

df_test_nn = df_test

df_test_nn['Survived'] = nn_test

df_test_nn = df_test_nn.drop(['Pclass'], axis=1)

df_test_nn = df_test_nn.drop(['Sex'], axis=1)

df_test_nn = df_test_nn.drop(['Age'], axis=1)

df_test_nn = df_test_nn.drop(['Fare'], axis=1)

df_test_nn = df_test_nn.drop(['Embarked'], axis=1)

df_test_nn.to_csv('Titanic_NN.csv', index = False)