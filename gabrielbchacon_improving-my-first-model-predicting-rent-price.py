# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

dataset.head()
dataset.info()
dataset.describe().T
dataset['city'].value_counts()
dataset['floor'].value_counts()
del dataset['floor']

del dataset['area']

del dataset['fire insurance (R$)']

del dataset['total (R$)']
dataset['animal'].value_counts()
dataset['furniture'].value_counts()
dt_trasformed = pd.get_dummies(dataset)

dt_trasformed = dt_trasformed[['hoa (R$)', 'property tax (R$)', 'rooms', 'bathroom', 'parking spaces', 'city_Belo Horizonte', 'city_Campinas', 'city_Porto Alegre', 'city_Rio de Janeiro', 'city_São Paulo', 'animal_acept', 'animal_not acept', 'furniture_furnished', 'furniture_not furnished',  'rent amount (R$)']]
dt_trasformed.info()
X = dt_trasformed.iloc[:, :-1]

y = dt_trasformed.iloc[:, -1]
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
#model = ExtraTreesClassifier()

#model.fit(X,y)

#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

#feat_importances = pd.Series(model.feature_importances_, index=X.columns)

#feat_importances.nlargest(10).plot(kind='barh')

#plt.show()
#get correlations of each features in dataset

corrmat = dt_trasformed.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(dt_trasformed[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X = dt_trasformed.iloc[:, 0:5].values

y = dt_trasformed.iloc[:, -1].values

y = y.reshape(len(y),1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_test)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

y_train = sc_y.fit_transform(y_train)

X_test = sc_X.transform(X_test)

y_test = sc_y.transform(y_test)
print(X_train[144])

print(y_train)
##Linear Model

from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()

lin_regressor.fit(X_train, y_train)
#Polynomial Model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)

poly_regressor = LinearRegression()

poly_regressor.fit(X_poly, y_train)
#Random Florest

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_regressor.fit(X_train, y_train)
#Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor(random_state = 0)

dt_regressor.fit(X_train, y_train)
#SVR Model

from sklearn.svm import SVR

svr_regressor = SVR(kernel = 'rbf')

svr_regressor.fit(X_train, y_train)
#Linear Prediction

y_pred_lin = lin_regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred_lin.reshape(len(y_pred_lin),1), y_test.reshape(len(y_test),1)),1))
#Polynomian Prediction

y_pred_poly = poly_regressor.predict(poly_reg.transform(X_test))

np.set_printoptions(precision=2)

print(np.concatenate((y_pred_poly.reshape(len(y_pred_poly),1), y_test.reshape(len(y_test),1)),1))
#Random Florest

y_pred_rf = rf_regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))
#Decison Tree

y_pred_dt = dt_regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred_dt.reshape(len(y_pred_dt),1), y_test.reshape(len(y_test),1)),1))
#SVR Prediction

#y_pred_svr = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X_test)))

#np.set_printoptions(precision=2)

#print(np.concatenate((y_pred_svr.reshape(len(y_pred_svr),1), y_test.reshape(len(y_test),1)),1))
print('Multiple Linear Regression R2: ' + str(r2_score(y_test, y_pred_lin)))

print('Polynomian Regression R2: ' + str(r2_score(y_test, y_pred_poly)))

print('Random Florest Regression R2: ' + str(r2_score(y_test, y_pred_rf)))

print('Decison Tree Regression R2: ' + str(r2_score(y_test, y_pred_dt)))