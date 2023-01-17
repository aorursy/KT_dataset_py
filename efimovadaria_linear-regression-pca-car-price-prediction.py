import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import shapiro

from scipy.stats import anderson

from scipy.stats import normaltest

from scipy.stats import norm

from scipy.stats import skew

import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler

from scipy import stats

import re

import warnings

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

from wordcloud import WordCloud, STOPWORDS 

warnings.filterwarnings('ignore')

%matplotlib inline



data = pd.read_csv('../input/carprice-dataset/CarPrice.csv')

data.head()

print(data.shape)

print(data.columns)

print(data.dtypes)
data = data.drop('car_ID', 1)

data = data.drop('symboling', 1)
for column in data:

    print("\n" + column + ":" + str(data[column].isnull().sum()))

    
sns.boxplot(x=data['price'])
z_score = stats.zscore(data['price'])

outlier = data[np.abs(z_score) > 3]

print(outlier)
print(sns.distplot(data['price']))
data['price'] = np.log1p(data['price'])

print(sns.distplot(data['price']))
#correlation matrix

corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, annot_kws={'size': 10}, annot = True, square=True);
data.plot.scatter(x='enginesize', y='price');
pca_columns = ["enginesize", "curbweight", "horsepower", 

               "carwidth", "carlength", "wheelbase", 

               "boreratio", "citympg", "highwaympg"]

pca_columns

data_pca = data[pca_columns]

#data_train.head()

#standardizing data

StandardScaler().fit_transform(data_pca)

data_pca.head()
from sklearn.decomposition import PCA

pca_train = PCA(n_components=2)

principal_components = pca_train.fit_transform(data_pca)

pca_train.explained_variance_ratio_
principal_data = pd.DataFrame(data = principal_components, columns = ['pca_1', 'pca_2'])

principal_data.head()
data['price'] = np.log1p(data['price'])

print(sns.distplot(data['price']))

principal_data['price'] = data['price']

principal_data.head()

principal_data.plot.scatter(x='pca_1', y='price');

print(sns.distplot(principal_data['pca_1']))
#remove numeric variables which we already used in PCA

data_rest = data._get_numeric_data()

data_rest.drop(pca_columns, axis=1, inplace = True)

data_rest.head()
#scatterplot

sns.set()

sns.pairplot(data_rest, size = 2.5)

plt.show()
dummy_data = data.select_dtypes(include=['object'])

dummy_data = dummy_data.drop('CarName', 1)

dummy_data.head()
dummy_data = pd.get_dummies(dummy_data)

print(dummy_data.shape)
dummy_data = pd.concat([data['price'], dummy_data], axis=1)
corrmat_dummy = dummy_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat_dummy, vmax=.8, annot_kws={'size': 5}, annot = True, square=True);
positive_corr = corrmat_dummy.sort_values('price', )[corrmat_dummy['price']>0.5]['price']

negatie_corr =  corrmat_dummy.sort_values('price', )[corrmat_dummy['price']<-0.5]['price']

correlated_dummy_cols = pd.concat([negatie_corr,positive_corr], axis=0).index

cm = np.corrcoef(dummy_data[correlated_dummy_cols].values.T)

f, ax = plt.subplots(figsize=(10, 8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, 

                 yticklabels=correlated_dummy_cols.values, xticklabels=correlated_dummy_cols.values)

plt.show()
sns.boxplot(x='drivewheel_rwd', y="price", data = dummy_data)
correlated_dummy_cols = correlated_dummy_cols.drop('drivewheel_fwd')

data_to_model = pd.concat([principal_data['pca_1'], dummy_data[correlated_dummy_cols]], axis=1)

data_to_model.head()
import sklearn.model_selection as model_selection



def split_data(data_to_model, pred, random_state):

    data_train, data_test, y_train, y_test = model_selection.train_test_split(data_to_model, pred,

                                                                              train_size=0.7,

                                                                              test_size=0.3, 

                                                                              random_state=random_state)

    print(data_train.shape)

    print(data_test.shape)

    print(y_train.shape)

    print(y_test.shape)

    return data_train, data_test, y_train, y_test
from sklearn.metrics import r2_score



def adjusted_r2(r2_score, n, p):

    len_score = (n-1)/(n-p-1)

    score = (1 - r2_score) * len_score

    return 1- score
from sklearn.linear_model import LinearRegression



pred = data_to_model['price']

pc1_data = pd.DataFrame(data = data_to_model['pca_1'], columns = ['pca_1'])



data_train, data_test, y_train, y_test = split_data(pc1_data, pred, random_state = 303)

lmPCA = LinearRegression()

lmPCA.fit(data_train,y_train)



print(lmPCA.intercept_)

print(lmPCA.coef_)



y_train_pred = lmPCA.predict(data_train)



print(r2_score(y_train,y_train_pred))

print(adjusted_r2(r2_score(y_train,y_train_pred), len(data_train), len(data_train.columns)))

plt.figure(figsize=(8,5))

p=plt.scatter(x=data_to_model['pca_1'],y=data_to_model['price'])

plt.xlabel("PCA 1",fontsize=15)

plt.ylabel("Response",fontsize=15)

plt.show()
possible_outliers_1 = data_to_model[(data_to_model['pca_1']>0) & (data_to_model['pca_1']<500) & (data_to_model['price'] > 2.4) & (data_to_model['price'] < 2.45)].index

print(possible_outliers_1)

possible_outlier_2 = data_to_model[(data_to_model['pca_1']>500) & (data_to_model['pca_1']<1000) & (data_to_model['price'] >2.3) & (data_to_model['price'] < 2.325)].index

print(possible_outlier_2)

data_to_model.drop(possible_outliers_1, axis=0, inplace = True)

data_to_model.drop(possible_outlier_2, axis=0, inplace = True)



pred = data_to_model['price']

pc1_data = pd.DataFrame(data = data_to_model['pca_1'], columns = ['pca_1'])



data_train, data_test, y_train, y_test = split_data(pc1_data, pred, random_state = 303)

lmPCA = LinearRegression()

lmPCA.fit(data_train,y_train)



print(lmPCA.intercept_)

print(lmPCA.coef_)



y_train_pred = lmPCA.predict(data_train)



print(r2_score(y_train,y_train_pred))

print(adjusted_r2(r2_score(y_train,y_train_pred), len(data_train), len(data_train.columns)))

pred = data_to_model['price']

data_to_model.drop('price', 1, inplace = True)



data_train, data_test, y_train, y_test = split_data(data_to_model, pred, random_state = 303)



lm = LinearRegression()

lm.fit(data_train,y_train)



print(lm.intercept_)

print(lm.coef_)



y_train_pred = lm.predict(data_train)



print(r2_score(y_train,y_train_pred))

print(adjusted_r2(r2_score(y_train,y_train_pred), len(data_train), len(data_train.columns)))
y_test_pred = lm.predict(data_test)



print(r2_score(y_test,y_test_pred))
sns.residplot(y_train_pred.reshape(-1), y_train, lowess=True,

                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 1})

plt.xlabel("Fitted values")

plt.ylabel("Residuals")

plt.title('Residual plot')
residuals = y_train - y_train_pred



model_norm_residuals_abs_sqrt=np.sqrt(np.abs(residuals))



plt.figure(figsize=(8,8))

sns.regplot(y_train_pred.reshape(-1), model_norm_residuals_abs_sqrt,

              scatter=True,

              lowess=True,

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plt.ylabel("Standarized residuals")

plt.xlabel("Fitted value")
plt.figure(figsize=(7,7))

stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Normal Q-Q Plot")
plt.figure(figsize=(7,7))

sns.distplot(residuals)

plt.title("Residuals distribution plot")

plt.xlabel("Residuals",fontsize=15)
from scipy.stats import shapiro

stat, p = shapiro(residuals)

print('Statistics=%.3f, p=%.3f' % (stat, p))