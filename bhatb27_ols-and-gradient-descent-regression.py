import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_test['revenue'] = None
data_all = data_train.append(data_test,ignore_index=True)
data_all.shape
data_all.info()
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop(['belongs_to_collection','homepage'],axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('tagline',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['p_company'] = data_all.production_companies.str.split(':').str.len()

data_all['p_company'].fillna(0,inplace=True)
data_all.drop('production_companies',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['s_genre'] = data_all.genres.str.split('}').str[0]
data_all['s_genre'] = data_all.s_genre.str.split(':').str[2]
data_all['s_genre'].fillna('Other',inplace=True)
data_all['s_genre'] = data_all.s_genre.str.strip(' \'')
data_all['s_genre'].value_counts()
data_all.drop('genres',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['n_keyword'] = data_all.Keywords.str.split('{').str.len()
data_all['n_keyword'].fillna(0,inplace=True)
data_all.drop('Keywords',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['p_country'] = data_all.production_countries.str.split(':').str[1]
data_all['p_country'] = data_all.p_country.str.split(',').str[0]
data_all['p_country'] = data_all.p_country.str.strip(' \' ')
data_all['p_country'].fillna('Other',inplace=True)
Country_weight = pd.DataFrame(data_all['p_country'].value_counts())
data_all = data_all.merge(Country_weight,how='left',left_on='p_country',right_index=True)
data_all.drop('p_country_x',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('production_countries',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['s_lang'] = data_all.spoken_languages.str.split(':').str[1]
data_all['s_lang'] = data_all.s_lang.str.split(',').str[0]
data_all['s_lang'] = data_all.s_lang.str.strip(' \' ')
data_all.isnull().sum()*100/len(data_all.index)
data_all['s_lang'].fillna('Other',inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('spoken_languages',axis=1,inplace=True)
lang_count = pd.DataFrame(data_all['s_lang'].value_counts())
data_all = data_all.merge(lang_count,how='left',left_on='s_lang',right_index=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('s_lang_x',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
o_lang = pd.DataFrame(data_all['original_language'].value_counts())
data_all = data_all.merge(o_lang,how='left',left_on='original_language',right_index=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('original_language_x',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['crew'] = data_all.crew.str.split('{').str.len()
data_all['crew'].fillna(0,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('cast',axis=1,inplace=True)
data_all.drop(['title','status'],axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['overview'] = data_all.overview.str.len()
data_all['overview'].fillna(0,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all.drop('poster_path',axis=1,inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['runtime'].fillna(data_all['runtime'].mean(),inplace=True)
data_all.isnull().sum()*100/len(data_all.index)
data_all['release_date'] = pd.to_datetime(data_all['release_date'])
data_all['r_year'] = data_all.release_date.dt.year
data_all['r_month'] = data_all.release_date.dt.month
data_all.info()
data_all['r_year'].fillna(data_all['r_year'].mean(),inplace=True)
data_all['r_month'].fillna(data_all['r_month'].mean(),inplace=True)
data_all.info()
data_all.drop('release_date',axis=1,inplace=True)
data_all.drop(['imdb_id','original_title'],axis=1,inplace=True)
data_all.info()
#Creating Dummy for s_genre

genre = pd.get_dummies(data_all['s_genre'], drop_first = True)

data_all = pd.concat([data_all, genre], axis = 1)

data_all.drop('s_genre',axis=1,inplace=True)
data_all.info()
data_all.loc[data_all['budget']==0,'budget'] = data_all['budget'].median()
d_train = data_all.loc[~data_all['revenue'].isnull(),]
d_test = data_all.loc[data_all['revenue'].isnull(),]
d_train['revenue'] = d_train['revenue'].astype(int)
sns.pairplot(d_train[['revenue','budget','overview','popularity','runtime','crew','p_company','n_keyword']])

plt.show()
sns.pairplot(d_train[['revenue','p_country_y','s_lang_y','original_language_y','r_year','r_month']])

plt.show()
d_train.loc[d_train['r_year']>2019,'r_year'] = d_train.loc[d_train['r_year']>2019,'r_year']-100
sns.pairplot(d_train[['revenue','p_country_y','s_lang_y','original_language_y','r_year','r_month']])

plt.show()
cor_mat = d_train[['revenue','p_country_y','s_lang_y','original_language_y','r_year','r_month','budget','overview','popularity','runtime','crew','p_company','n_keyword']].corr()
#Heat map of correlation metrix

plt.figure(figsize = (16, 10))

sns.heatmap(cor_mat, annot = True, cmap="YlGnBu")

plt.show()
d_train['ln_revenue'] = np.log(d_train['revenue'])
cor_mat = d_train[['revenue','ln_revenue','p_country_y','s_lang_y','original_language_y','r_year','r_month','budget','overview','popularity','runtime','crew','p_company','n_keyword']].corr()
#Heat map of correlation metrix

plt.figure(figsize = (16, 10))

sns.heatmap(cor_mat, annot = True, cmap="YlGnBu")

plt.show()
d_train.drop('ln_revenue',axis=1,inplace=True)
d_train_X = d_train.drop('revenue',axis=1)

d_train_y = d_train['revenue']
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
col_list = ['budget','popularity','crew','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Science Fiction','runtime','n_keyword']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



d_train_X[col_list] = scaler.fit_transform(d_train_X[col_list])
d_train_X = d_train_X[col_list]
import statsmodels.api as sm

d_train_X = sm.add_constant(d_train_X)
lm_1 = sm.OLS(d_train_y,d_train_X).fit()
print(lm_1.summary())
d_train_X.drop(['const','Drama'],axis=1,inplace=True)
import statsmodels.api as sm

d_train_X = sm.add_constant(d_train_X)
lm_2 = sm.OLS(d_train_y,d_train_X).fit()
print(lm_2.summary())
d_train_X.drop(['const','Documentary','Crime'],axis=1,inplace=True)
import statsmodels.api as sm

d_train_X = sm.add_constant(d_train_X)
lm_3 = sm.OLS(d_train_y,d_train_X).fit()
print(lm_3.summary())
d_train_X.drop(['const','crew'],axis=1,inplace=True)
import statsmodels.api as sm

d_train_X = sm.add_constant(d_train_X)
lm_4 = sm.OLS(d_train_y,d_train_X).fit()
print(lm_4.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = d_train_X.drop('const',axis=1)

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_pred = lm_4.predict(d_train_X)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((d_train_y - y_pred), bins = 20)

fig.suptitle('Residual Hist', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 10)
# Plot the scatter of the error terms/price

fig = plt.figure()

plt.scatter(d_train_y,d_train_y-y_pred)

fig.suptitle('Error Vs Price', fontsize = 20)                  # Plot heading 

plt.ylabel('Errors', fontsize = 10)

plt.xlabel('Price', fontsize = 10)
# Plot the scatter of the error terms/price

fig = plt.figure()

plt.scatter(d_train_y,y_pred)

fig.suptitle('Correlation b/w actual and predicted price', fontsize = 20)                  # Plot heading 

plt.ylabel('Predicted price', fontsize = 10)

plt.xlabel('Price', fontsize = 10)
d_test_X = d_test.drop('revenue',axis=1)

d_test_y = d_test['revenue']
d_test_X[col_list] = scaler.transform(d_test_X[col_list])
d_test_X = d_test_X[col_list]
import statsmodels.api as sm

d_test_X = sm.add_constant(d_test_X)
d_test['revenue'] = lm_4.predict(d_test_X[list(d_train_X.columns)])
submit_1 = d_test[['id','revenue']]
submit_1.to_csv('submit1.csv')
from sklearn import linear_model 
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
d_train_X2 = d_train.drop('revenue',axis=1)

d_train_y2 = d_train['revenue']
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(data_all[col_list])
d_train_X2[col_list] = scaler1.transform(d_train_X2[col_list])
col_2 = list(d_train_X.columns)
col_2.pop(0)
col_2
fit_1 = clf.fit(d_train_X2[col_2],d_train_y2)
y_pred2 = fit_1.predict(d_train_X2[col_2])
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((d_train_y2 - y_pred2), bins = 20)

fig.suptitle('Residual Hist', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 10)
# Plot the scatter of the error terms/price

fig = plt.figure()

plt.scatter(d_train_y2,d_train_y2-y_pred2)

fig.suptitle('Error Vs Price', fontsize = 20)                  # Plot heading 

plt.ylabel('Errors', fontsize = 10)

plt.xlabel('Price', fontsize = 10)
fit_1.score(d_train_X2[col_2],d_train_y2)
d_test_X2 = d_test.drop('revenue',axis=1)
d_test_X2[col_list] = scaler1.transform(d_test_X2[col_list])
d_test['revenue2'] = fit_1.predict(d_test_X2[col_2])
submit2 = d_test[['id','revenue2']]
submit2.to_csv('submit2.csv')
d_train_3 = d_train.copy()
d_train_3.info()
from sklearn.preprocessing import StandardScaler

scaler3 = StandardScaler()



d_train_3[['revenue']] = scaler3.fit_transform(d_train_3[['revenue']])
d_train_3[col_list] = scaler1.transform(d_train_3[col_list])
d_train_X3 = d_train_3[col_list]
d_train_y3 = d_train_3[['revenue']]
clf1 = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf1.fit(d_train_X3[col_2],d_train_y3)
d_train_pred3 = clf1.predict(d_train_X3[col_2])
scaler3.inverse_transform(d_train_pred3)
d_test['revenue3'] = clf1.predict(d_test_X2[col_2])
d_test['revenue3'] = scaler3.inverse_transform(d_test['revenue3'])
submit3 = d_test[['id','revenue3']]
submit3.to_csv('submit3.csv')
np.corrcoef(d_train['budget'],d_train['revenue'])
d_train