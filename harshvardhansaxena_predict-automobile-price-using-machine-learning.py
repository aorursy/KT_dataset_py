import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
df = pd.read_csv("AutoData.csv")
df
df.describe()
df.info()
sns.heatmap(df.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')
sns.countplot(x= 'fueltype', data = df)
sns.set_style('whitegrid')
sns.countplot(x= 'fueltype',hue = 'carbody', data = df, palette = 'rainbow')
sns.set_style('whitegrid')
sns.countplot(x= 'fueltype',hue = 'drivewheel', data = df, palette = 'rainbow')
sns.set_style('whitegrid')
sns.countplot(x= 'fueltype',hue = 'enginelocation', data = df, palette = 'rainbow')
sns.set_style('whitegrid')
sns.countplot(x= 'fueltype',hue = 'cylindernumber', data = df, palette = 'rainbow')
df.price.describe()
sns.distplot(df['price'],kde = False ,color = 'blue', bins = 40)
barG = df[['symboling','price']].groupby("symboling").mean().plot(kind='bar',legend = False,color = 'green')
barG.set_xlabel("Symbol")
barG.set_ylabel("Price")
plt.show()
df.make.values[0:10]
df['company'] = df.make.str.split(' ').str.get(0).str.upper()
df['company'].unique() 
df['company'] = df['company'].replace(['MAXDA'], 'MAZDA')
df['company'] = df['company'].replace(['PORCSHCE'], 'PORSCHE')
df['company'] = df['company'].replace(['TOYOUTA'], 'TOYOTA')
df['company'] = df['company'].replace(['VW', 'VOKSWAGEN'], 'VOLKSWAGEN')
df_avg = df[['company','price']].groupby("company", as_index = False).mean().rename(columns={'price':'Avgprice'})
barG = df_avg.plot(x = 'company', kind='bar',legend = False, sort_columns = True, figsize = (15,5), color= 'purple')
barG.set_xlabel("Company")
barG.set_ylabel("Avg Price")
plt.show()
df = df.merge(df_avg, on = 'company')
df['Car_cat'] = df['Avgprice'].apply(lambda x : "Budget" if x < 12000 else ("Mid_Range" if 12000 <= x < 24000 else "Luxury"))

df['mileage'] = df['citympg']*0.6 + df['highwaympg']*0.4
auto = df.copy()
auto.drop(['make','symboling','doornumber','enginelocation','carheight','fuelsystem','stroke','compressionratio','peakrpm','citympg','highwaympg','company','Avgprice'], axis = 1, inplace = True)
auto
auto = pd.get_dummies(auto, drop_first = True)
auto
plt.figure(figsize=(30,30))
ax = sns.heatmap(auto.corr(), annot = True, linewidth = 3)
ax.tick_params(size = 10, labelsize = 10)
plt.title("Automobile industrie sale", fontsize = 25)
plt.show()
from sklearn.linear_model import LinearRegression
X = auto['enginesize'].values.reshape(-1, 1)
Y = auto['price'].values.reshape(-1, 1)
linear_regressor = LinearRegression() 
linear_regressor.fit(X, Y)
pre = linear_regressor.predict(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=51)
 
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)
Y_pred = linear_regressor.predict(X_test)
Y_pred.shape
from sklearn.metrics import r2_score
r2_score(y_test,Y_pred)
plt.scatter(X, Y)
plt.plot(X, pre, color='green')
plt.show()
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(auto, test_size=0.2, random_state=51)
 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import warnings
warnings.filterwarnings('ignore')

num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']

df_train[num_vars] = sc.fit_transform(df_train[num_vars])
X_train = df_train.drop('price', axis=1)
y_train = df_train['price']
 
print('Shape of X = ', X_train.shape)
print('Shape of y = ', y_train.shape) 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
X_train
linear_regressor.fit(X_train, y_train)

rfe = RFE(linear_regressor, 10)  
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train_rfe = X_train[col]
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
linear_regressor = sm.OLS(y_train,X_train_rfe).fit()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new1 = X_train_rfe.drop(["cylindernumber_twelve"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new1)
linear_regressor = sm.OLS(y_train,X_train_lm).fit()  
print(linear_regressor.summary())
vif = pd.DataFrame()
X = X_train_new1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new2 = X_train_new1.drop(["carbody_sedan"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new2)
lm = sm.OLS(y_train,X_train_lm).fit()
print(lm.summary())
X_train_new3 = X_train_new2.drop(["carbody_hardtop"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new3)
linear_regressor = sm.OLS(y_train,X_train_lm).fit()
print(linear_regressor.summary())
vif = pd.DataFrame()
X = X_train_new3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new4 = X_train_new3.drop(["curbweight"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new4)
linear_regressor = sm.OLS(y_train,X_train_lm).fit()
print(linear_regressor.summary())
X_train_new5 = X_train_new4.drop(["carbody_wagon"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new5)
linear_regressor = sm.OLS(y_train,X_train_lm).fit()
print(linear_regressor.summary())
vif = pd.DataFrame()
X = X_train_new5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new6 = X_train_new5.drop(["enginetype_dohcv"], axis = 1)
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new6)
linear_regressor = sm.OLS(y_train,X_train_lm).fit()
print(linear_regressor.summary())
num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']

df_test[num_vars] = sc.fit_transform(df_test[num_vars])
X_test = df_test.drop('price', axis=1)
y_test = df_test['price']
 
print('Shape of X = ', X_test.shape)
print('Shape of y = ', y_test.shape) 
X_test_new = X_test[['carwidth', 'horsepower', 'Car_cat_Luxury', 'carbody_hatchback']]

import statsmodels.api as sm
X_test_new = sm.add_constant(X_test_new)
X_test_new.head()
X_test_new
X_test
y_pred = linear_regressor.predict(X_test_new)
y_pred
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('Test Vs Prediction', fontsize=15)           
plt.xlabel('Test', fontsize=12)                          
plt.ylabel('Prediction', fontsize=12)               