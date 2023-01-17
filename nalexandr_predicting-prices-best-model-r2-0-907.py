import pandas as pd

import numpy as np

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.head()
del df['id']
# 1) Leave the first 8 characters

df['date'] = df['date'].apply(lambda x: x[:8])

# 2) Convert str to datetime type

df['date'] = df['date'].astype('datetime64[ns]')

# 3) Convert the date to a day from the beginning of the year

df['date'] = df['date'].apply(lambda x: x.timetuple().tm_yday)
df.head()
# number of null values

df.isnull().sum().max()
# general statistics df

df.describe()
h = df.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

sns.despine(left=True, bottom=True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
f = plt.figure(figsize=(16, 10))

corr = df.iloc[:,1:].corr()

sns.heatmap(corr, cmap='coolwarm_r', annot=True, annot_kws={'size':10})

plt.title('Матрица корреляции', fontsize=12);
with sns.plotting_context("notebook",font_scale=2.5):

    g = sns.pairplot(df[['price','sqft_living','sqft_above','sqft_living15',

                         'bathrooms','bedrooms','sqft_basement','lat', 'grade']], 

                 hue='grade', palette='tab20', size=6)

g.set(xticklabels=[]);
ax = sns.lmplot(x="sqft_living", y="price", data=df, hue=None)
ax = sns.lmplot(x="sqft_living", y="price", data=df, hue="grade", palette='tab20')
ax = sns.lmplot(x="sqft_living", y="price", data=df, palette='tab20', hue="grade", col="grade", col_wrap=3, height=4)
df_lg = df.copy()

df_lg[['price','sqft_living','sqft_above','sqft_living15']] = np.log10(df[['price','sqft_living','sqft_above','sqft_living15']])

   

df_lg.head()
data = df_lg[['price','sqft_living','sqft_above','sqft_living15']]



h = data.hist(bins=20,figsize=(12,10),xlabelsize='10',ylabelsize='10')

sns.despine(left=True, bottom=True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
sns.set(style="whitegrid")

ax = sns.lmplot( x="long", y="lat", data=df, fit_reg=False, hue='grade', 

                legend=False, palette="Blues", height=8.27, aspect=1.4)

plt.legend(loc='lower right')

plt.show()
from sklearn.model_selection import train_test_split
# divide the data into a training set for data in its original form

train, test = train_test_split(df,train_size = 0.8, random_state = 42)



# for data on a logarithmic scale

train_lg, test_lg = train_test_split(df_lg,train_size = 0.8, random_state = 42)
x_train = train.drop(['price'], axis=1)

y_train = train.price



x_test = test.drop(['price'], axis=1)

y_test = test.price





x_train_lg = train_lg.drop(['price'], axis=1)

y_train_lg = train_lg.price



x_test_lg = test_lg.drop(['price'], axis=1)

y_test_lg = test_lg.price
print('Average price (y_train):', np.mean(y_train))

print('Average price (y_test):', np.mean(y_test))

print('Average price (y_train_lg):', 10 ** np.mean(y_train_lg))

print('Average price (y_test_lg):', 10 ** np.mean(y_test_lg))
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error
name = 'Linear Regression'

details = '-'



model = LinearRegression()

model.fit(x_train, y_train)
print('Quality of the model R2: train -', "%.3f" %  model.score(x_train, y_train), 

      'test -', "%.3f" %  model.score(x_test, y_test))
coef = pd.DataFrame(zip(['intercept'] + x_train.columns.tolist(), [model.intercept_] + model.coef_.tolist()),

                    columns=['predictor', 'coef'])

coef
from scipy import stats



def regression_coef(model, X, y):

    coef = pd.DataFrame(zip(['intercept'] + X.columns.tolist(), [model.intercept_] + model.coef_.tolist()),

                    columns=['predictor', 'coef'])

    X1 = np.append(np.ones((len(X),1)), X, axis=1)

    b = np.append(model.intercept_, model.coef_)

    MSE = np.sum((model.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

    var_b = MSE * (np.linalg.inv(np.dot(X1.T, X1)).diagonal())

    sd_b = np.sqrt(var_b)

    t = b / sd_b

    coef['pvalue'] = [2 * (1 - stats.t.cdf(np.abs(i), (len(X1) - 1))) for i in t]

    return coef
regression_coef(model, x_test, y_test)
stat_sign = ['sqft_living','sqft_above','sqft_basement']



x_train_st = x_train[stat_sign]

x_test_st = x_test[stat_sign]



x_train_lg_st = x_train_lg[stat_sign]

x_test_lg_st = x_test_lg[stat_sign]
table = pd.DataFrame(columns=['Regressor', 'Details', 'R^2 (train)', 'R^2 (test)', 

                              'mae (train)', 'mae (test)', 'rmse (train)', 'rmse (test)'])

table_lg = pd.DataFrame(columns=['Regressor', 'Details', 'R^2 (train)', 'R^2 (test)',

                                 'mae (train)', 'mae (test)', 'rmse (train)', 'rmse (test)'])
# creating a list with quality metrics

def model_quality(model, x_train, y_train, x_test, y_test):

    k = list()

    k.append(model.score(x_train, y_train))

    k.append(model.score(x_test, y_test))

    k.append(mean_absolute_error(model.predict(x_train), y_train))

    k.append(mean_absolute_error(model.predict(x_test), y_test))

    k.append(np.sqrt(mean_squared_error(model.predict(x_train), y_train)))

    k.append(np.sqrt(mean_squared_error(model.predict(x_test), y_test)))

    return k



# for values on a logarithmic scale

def model_quality_lg(model, x_train, y_train, x_test, y_test):

    k = list()

    k.append(model.score(x_train, y_train))

    k.append(model.score(x_test, y_test))

    k.append(mean_absolute_error(10 ** model.predict(x_train), 10 ** y_train))

    k.append(mean_absolute_error(10 ** model.predict(x_test), 10 ** y_test))

    k.append(np.sqrt(mean_squared_error(10 ** model.predict(x_train), 10 ** y_train)))

    k.append(np.sqrt(mean_squared_error(10 ** model.predict(x_test), 10 ** y_test)))

    return k



# print metric values

def print_quality(k):

    print ('R2 - train:', "%.3f" % k[0], 'test:', "%.3f" % k[1])

    print ('mae - train:', "%.3f" % k[2], 'test:', "%.3f" % k[3])

    print ('rmse - train:', "%.3f" % k[4], 'test:', "%.3f" % k[5])

    

def add_to_table(table, name, details, k):

    table.loc[len(table)] = [name, details, k[0], k[1], k[2], k[3], k[4], k[5]]
# Исходный набор

name_lr = 'Multiple'

details_lr = '-'

lr = LinearRegression()

lr.fit(x_train, y_train)

k_lr = model_quality(lr, x_train, y_train, x_test, y_test)

add_to_table(table, name_lr, details_lr, k_lr)

print('MODEL: ', name_lr, details_lr)

print_quality(k_lr)

##############################################################################################



# Статистически значимые величины

name_lr_st = 'Multiple'

details_lr_st = 'stat. sign. coef.'

lr_st = LinearRegression()

lr_st.fit(x_train_st, y_train)

k_lr_st = model_quality(lr_st, x_train_st, y_train, x_test_st, y_test)

add_to_table(table, name_lr_st, details_lr_st, k_lr_st)

print('\nMODEL: ', name_lr_st, details_lr_st)

print_quality(k_lr_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_lr_lg = 'Multiple'

details_lr_lg = '-'

lr_lg = LinearRegression()

lr_lg.fit(x_train_lg, y_train_lg)

k_lr_lg = model_quality_lg(lr_lg, x_train_lg, y_train_lg, x_test_lg, y_test_lg)

add_to_table(table_lg, name_lr_lg, details_lr_lg, k_lr_lg)

print('\nMODEL: ', name_lr_lg, details_lr_lg)

print_quality(k_lr_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_lr_lg_st = 'Multiple'

details_lr_lg_st = 'stat. sign. coef.'

lr_lg_st = LinearRegression()

lr_lg_st.fit(x_train_lg_st, y_train_lg)

k_lr_lg_st = model_quality_lg(lr_lg_st, x_train_lg_st, y_train_lg, x_test_lg_st, y_test_lg)

add_to_table(table_lg, name_lr_lg_st, details_lr_lg_st, k_lr_lg_st)

print('\nMODEL: ', name_lr_lg_st, details_lr_lg_st)

print_quality(k_lr_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg
from sklearn.preprocessing import PolynomialFeatures
# Исходный набор

name_poly2 = 'Polynomial 2'

details_poly2 = '-'

#############

polyfeat_p2 = PolynomialFeatures(degree=2)

x_train_p2 = polyfeat_p2.fit_transform(x_train)

x_test_p2 = polyfeat_p2.fit_transform(x_test)

#############

poly2 = LinearRegression()

poly2.fit(x_train_p2, y_train)

k_poly2 = model_quality(poly2, x_train_p2, y_train, x_test_p2, y_test)

add_to_table(table, name_poly2, details_poly2, k_poly2)

print('MODEL: ', name_poly2, details_poly2)

print_quality(k_poly2)

##############################################################################################



# Статистически значимые величины

name_poly2_st = 'Polynomial 2'

details_poly2_st = 'stat. sign. coef.'

#############

polyfeat_p2_st = PolynomialFeatures(degree=2)

x_train_p2_st = polyfeat_p2.fit_transform(x_train_st)

x_test_p2_st = polyfeat_p2.fit_transform(x_test_st)

#############

poly2_st = LinearRegression()

poly2_st.fit(x_train_p2_st, y_train)

k_poly2_st = model_quality(poly2_st, x_train_p2_st, y_train, x_test_p2_st, y_test)

add_to_table(table, name_poly2_st, details_poly2_st, k_poly2_st)

print('\nMODEL: ', name_poly2_st, details_poly2_st)

print_quality(k_poly2_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_poly2_lg = 'Polynomial 2'

details_poly2_lg = '-'

#############

polyfeat_p2_lg = PolynomialFeatures(degree=2)

x_train_p2_lg = polyfeat_p2.fit_transform(x_train_lg)

x_test_p2_lg = polyfeat_p2.fit_transform(x_test_lg)

#############

poly2_lg = LinearRegression()

poly2_lg.fit(x_train_p2_lg, y_train_lg)

k_poly2_lg = model_quality_lg(poly2_lg, x_train_p2_lg, y_train_lg, x_test_p2_lg, y_test_lg)

add_to_table(table_lg, name_poly2_lg, details_poly2_lg, k_poly2_lg)

print('\nMODEL: ', name_poly2_lg, details_poly2_lg)

print_quality(k_poly2_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_poly2_lg_st = 'Polynomial 2'

details_poly2_lg_st = 'stat. sign. coef.'

#############

polyfeat_p2_lg_st = PolynomialFeatures(degree=2)

x_train_p2_lg_st = polyfeat_p2_lg_st.fit_transform(x_train_lg_st)

x_test_p2_lg_st = polyfeat_p2_lg_st.fit_transform(x_test_lg_st)

#############

poly2_lg_st = LinearRegression()

poly2_lg_st.fit(x_train_p2_lg_st, y_train_lg)

k_poly2_lg_st = model_quality_lg(poly2_lg_st, x_train_p2_lg_st, y_train_lg, x_test_p2_lg_st, y_test_lg)

add_to_table(table_lg, name_poly2_lg_st, details_poly2_lg_st, k_poly2_lg_st)

print('\nMODEL: ', name_poly2_lg_st, details_poly2_lg_st)

print_quality(k_poly2_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg
# Исходный набор

name_poly3 = 'Polynomial 3'

details_poly3 = '-'

#############

polyfeat_p3 = PolynomialFeatures(degree=3)

x_train_p3 = polyfeat_p3.fit_transform(x_train)

x_test_p3 = polyfeat_p3.fit_transform(x_test)

#############

poly3 = LinearRegression()

poly3.fit(x_train_p3, y_train)

k_poly3 = model_quality(poly3, x_train_p3, y_train, x_test_p3, y_test)

add_to_table(table, name_poly3, details_poly3, k_poly3)

print('MODEL: ', name_poly3, details_poly3)

print_quality(k_poly3)

##############################################################################################



# Статистически значимые величины

name_poly3_st = 'Polynomial 3'

details_poly3_st = 'stat. sign. coef.'

#############

polyfeat_p3_st = PolynomialFeatures(degree=3)

x_train_p3_st = polyfeat_p3.fit_transform(x_train_st)

x_test_p3_st = polyfeat_p3.fit_transform(x_test_st)

#############

poly3_st = LinearRegression()

poly3_st.fit(x_train_p3_st, y_train)

k_poly3_st = model_quality(poly3_st, x_train_p3_st, y_train, x_test_p3_st, y_test)

add_to_table(table, name_poly3_st, details_poly3_st, k_poly3_st)

print('\nMODEL: ', name_poly3_st, details_poly3_st)

print_quality(k_poly3_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_poly3_lg = 'Polynomial 3'

details_poly3_lg = '-'

#############

polyfeat_p3_lg = PolynomialFeatures(degree=3)

x_train_p3_lg = polyfeat_p3.fit_transform(x_train_lg)

x_test_p3_lg = polyfeat_p3.fit_transform(x_test_lg)

#############

poly3_lg = LinearRegression()

poly3_lg.fit(x_train_p3_lg, y_train_lg)

k_poly3_lg = model_quality_lg(poly3_lg, x_train_p3_lg, y_train_lg, x_test_p3_lg, y_test_lg)

add_to_table(table_lg, name_poly3_lg, details_poly3_lg, k_poly3_lg)

print('\nMODEL: ', name_poly3_lg, details_poly3_lg)

print_quality(k_poly3_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_poly3_lg_st = 'Polynomial 3'

details_poly3_lg_st = 'stat. sign. coef.'

#############

polyfeat_p3_lg_st = PolynomialFeatures(degree=3)

x_train_p3_lg_st = polyfeat_p3_lg_st.fit_transform(x_train_lg_st)

x_test_p3_lg_st = polyfeat_p3_lg_st.fit_transform(x_test_lg_st)

#############

poly3_lg_st = LinearRegression()

poly3_lg_st.fit(x_train_p3_lg_st, y_train_lg)

k_poly3_lg_st = model_quality_lg(poly3_lg_st, x_train_p3_lg_st, y_train_lg, x_test_p3_lg_st, y_test_lg)

add_to_table(table_lg, name_poly3_lg_st, details_poly3_lg_st, k_poly3_lg_st)

print('\nMODEL: ', name_poly3_lg_st, details_poly3_lg_st)

print_quality(k_poly3_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg
from sklearn.neighbors import KNeighborsRegressor
# Исходный набор

name_knnr = 'k-NN'

details_knnr = 'k=12'

knnr = KNeighborsRegressor(n_neighbors=12)

knnr.fit(x_train, y_train)

k_knnr = model_quality(knnr, x_train, y_train, x_test, y_test)

add_to_table(table, name_knnr, details_knnr, k_knnr)

print('MODEL: ', name_knnr, details_knnr)

print_quality(k_knnr)

##############################################################################################



# Статистически значимые величины

name_knnr_st = 'k-NN'

details_knnr_st = 'stat. sign. coef., k=12'

knnr_st = KNeighborsRegressor(n_neighbors=12)

knnr_st.fit(x_train_st, y_train)

k_knnr_st = model_quality(knnr_st, x_train_st, y_train, x_test_st, y_test)

add_to_table(table, name_knnr_st, details_knnr_st, k_knnr_st)

print('\nMODEL: ', name_knnr_st, details_knnr_st)

print_quality(k_knnr_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_knnr_lg = 'k-NN'

details_knnr_lg = 'k=12'

knnr_lg = KNeighborsRegressor(n_neighbors=12)

knnr_lg.fit(x_train_lg, y_train_lg)

k_knnr_lg = model_quality_lg(knnr_lg, x_train_lg, y_train_lg, x_test_lg, y_test_lg)

add_to_table(table_lg, name_knnr_lg, details_knnr_lg, k_knnr_lg)

print('\nMODEL: ', name_knnr_lg, details_knnr_lg)

print_quality(k_knnr_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_knnr_lg_st = 'k-NN'

details_knnr_lg_st = 'stat. sign. coef., k=12'

knnr_lg_st = KNeighborsRegressor(n_neighbors=12)

knnr_lg_st.fit(x_train_lg_st, y_train_lg)

k_knnr_lg_st = model_quality_lg(knnr_lg_st, x_train_lg_st, y_train_lg, x_test_lg_st, y_test_lg)

add_to_table(table_lg, name_knnr_lg_st, details_knnr_lg_st, k_knnr_lg_st)

print('\nMODEL: ', name_knnr_lg_st, details_knnr_lg_st)

print_quality(k_knnr_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg
from sklearn.ensemble import RandomForestRegressor
# Исходный набор

name_rf = 'Random Forest'

details_rf = 'n_jobs=-1, n_estimators=55'

rf = RandomForestRegressor(n_jobs=-1, n_estimators=55)

rf.fit(x_train, y_train)

k_rf = model_quality(rf, x_train, y_train, x_test, y_test)

add_to_table(table, name_rf, details_rf, k_rf)

print('MODEL: ', name_rf, details_rf)

print_quality(k_rf)

##############################################################################################



# Статистически значимые величины

name_rf_st = 'Random Forest'

details_rf_st = 'stat. sign. coef., n_jobs=-1, n_estimators=55'

rf_st = RandomForestRegressor(n_jobs=-1, n_estimators=55)

rf_st.fit(x_train_st, y_train)

k_rf_st = model_quality(rf_st, x_train_st, y_train, x_test_st, y_test)

add_to_table(table, name_rf_st, details_rf_st, k_rf_st)

print('\nMODEL: ', name_rf_st, details_rf_st)

print_quality(k_rf_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_rf_lg = 'Random Forest'

details_rf_lg = 'n_jobs=-1, n_estimators=55'

rf_lg = RandomForestRegressor(n_jobs=-1, n_estimators=55)

rf_lg.fit(x_train_lg, y_train_lg)

k_rf_lg = model_quality_lg(rf_lg, x_train_lg, y_train_lg, x_test_lg, y_test_lg)

add_to_table(table_lg, name_rf_lg, details_rf_lg, k_rf_lg)

print('\nMODEL: ', name_rf_lg, details_rf_lg)

print_quality(k_rf_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_rf_lg_st = 'Random Forest'

details_rf_lg_st = 'stat. sign. coef., n_jobs=-1, n_estimators=55'

rf_lg_st = RandomForestRegressor(n_jobs=-1, n_estimators=55)

rf_lg_st.fit(x_train_lg_st, y_train_lg)

k_rf_lg_st = model_quality_lg(rf_lg_st, x_train_lg_st, y_train_lg, x_test_lg_st, y_test_lg)

add_to_table(table_lg, name_rf_lg_st, details_rf_lg_st, k_rf_lg_st)

print('\nMODEL: ', name_rf_lg_st, details_rf_lg_st)

print_quality(k_rf_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic sets")

table_lg
from xgboost import XGBRegressor
# Исходный набор

name_xgbr = 'XGboost'

details_xgbr = 'max_depth=4'

xgbr = XGBRegressor(max_depth=4)

xgbr.fit(x_train, y_train)

k_xgbr = model_quality(xgbr, x_train, y_train, x_test, y_test)

add_to_table(table, name_xgbr, details_xgbr, k_xgbr)

print('MODEL: ', name_xgbr, details_xgbr)

print_quality(k_xgbr)

##############################################################################################



# Статистически значимые величины

name_xgbr_st = 'XGboost'

details_xgbr_st = 'stat. sign. coef., max_depth=4'

xgbr_st = XGBRegressor(max_depth=4)

xgbr_st.fit(x_train_st, y_train)

k_xgbr_st = model_quality(xgbr_st, x_train_st, y_train, x_test_st, y_test)

add_to_table(table, name_xgbr_st, details_xgbr_st, k_xgbr_st)

print('\nMODEL: ', name_xgbr_st, details_xgbr_st)

print_quality(k_xgbr_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_xgbr_lg = 'XGboost'

details_xgbr_lg = 'max_depth=4'

xgbr_lg = XGBRegressor(max_depth=4)

xgbr_lg.fit(x_train_lg, y_train_lg)

k_xgbr_lg = model_quality_lg(xgbr_lg, x_train_lg, y_train_lg, x_test_lg, y_test_lg)

add_to_table(table_lg, name_xgbr_lg, details_xgbr_lg, k_xgbr_lg)

print('\nMODEL: ', name_xgbr_lg, details_xgbr_lg)

print_quality(k_xgbr_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_xgbr_lg_st = 'XGboost'

details_xgbr_lg_st = 'stat. sign. coef., max_depth=4'

xgbr_lg_st = XGBRegressor(max_depth=4)

xgbr_lg_st.fit(x_train_lg_st, y_train_lg)

k_xgbr_lg_st = model_quality_lg(xgbr_lg_st, x_train_lg_st, y_train_lg, x_test_lg_st, y_test_lg)

add_to_table(table_lg, name_xgbr_lg_st, details_xgbr_lg_st, k_xgbr_lg_st)

print('\nMODEL: ', name_xgbr_lg_st, details_xgbr_lg_st)

print_quality(k_xgbr_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg
from lightgbm import LGBMRegressor
# Исходный набор

name_lgbmr = 'LightGBM'

details_lgbmr = '-'

lgbmr = LGBMRegressor()

lgbmr.fit(x_train, y_train)

k_lgbmr = model_quality(lgbmr, x_train, y_train, x_test, y_test)

add_to_table(table, name_lgbmr, details_lgbmr, k_lgbmr)

print('MODEL: ', name_lgbmr, details_lgbmr)

print_quality(k_lgbmr)

##############################################################################################



# Статистически значимые величины

name_lgbmr_st = 'LightGBM'

details_lgbmr_st = 'stat. sign. coef.'

lgbmr_st = LGBMRegressor()

lgbmr_st.fit(x_train_st, y_train)

k_lgbmr_st = model_quality(lgbmr_st, x_train_st, y_train, x_test_st, y_test)

add_to_table(table, name_lgbmr_st, details_lgbmr_st, k_lgbmr_st)

print('\nMODEL: ', name_lgbmr_st, details_lgbmr_st)

print_quality(k_lgbmr_st)

##############################################################################################





# Regression for logarithmic sets



# Исходный набор с логарифмированными величинами

name_lgbmr_lg = 'LightGBM'

details_lgbmr_lg = '-'

lgbmr_lg = LGBMRegressor()

lgbmr_lg.fit(x_train_lg, y_train_lg)

k_lgbmr_lg = model_quality_lg(lgbmr_lg, x_train_lg, y_train_lg, x_test_lg, y_test_lg)

add_to_table(table_lg, name_lgbmr_lg, details_lgbmr_lg, k_lgbmr_lg)

print('\nMODEL: ', name_lgbmr_lg, details_lgbmr_lg)

print_quality(k_lgbmr_lg)

##############################################################################################



# Логарифмированные статистически значимые величины

name_lgbmr_lg_st = 'LightGBM'

details_lgbmr_lg_st = 'stat. sign. coef.'

lgbmr_lg_st = LGBMRegressor()

lgbmr_lg_st.fit(x_train_lg_st, y_train_lg)

k_lgbmr_lg_st = model_quality_lg(lgbmr_lg_st, x_train_lg_st, y_train_lg, x_test_lg_st, y_test_lg)

add_to_table(table_lg, name_lgbmr_lg_st, details_lgbmr_lg_st, k_lgbmr_lg_st)

print('\nMODEL: ', name_lgbmr_lg_st, details_lgbmr_lg_st)

print_quality(k_lgbmr_lg_st)

##############################################################################################
print("For original data:")

table
print("For logarithmic data")

table_lg