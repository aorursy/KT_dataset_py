import warnings

warnings.filterwarnings('ignore')
#import all libraries



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
cars_read = pd.read_csv('../input/carpriceprediction/CarPrice_Assignment.csv')

cars_read.head()
#shape of the data

cars_read.shape
#info the dataframe

cars_read.info()
#describe the data

cars_read.describe(percentiles = [0.10,0.25,0.50,0.75,0.90,0.99])
cars_clean = cars_read

cars_clean.duplicated(subset = ['car_ID']).sum()
cars_clean.drop(['car_ID'], axis =1)
cars_clean.isnull().sum()
#symboling column- Its assigned insurance risk rating, 

#A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.

cars_clean['symboling'].value_counts()
sns.pairplot(y_vars = 'symboling', x_vars = 'price' ,data = cars_clean)
#Column CarName

cars_clean['CarName'].value_counts()
cars_clean['car_company'] = cars_clean['CarName'].apply(lambda x:x.split(' ')[0])
#rechecking

cars_clean.head()
#deleting the original column

cars_clean = cars_clean.drop(['CarName'], axis =1)
cars_clean['car_company'].value_counts()
cars_clean['car_company'].replace('toyouta', 'toyota',inplace=True)

cars_clean['car_company'].replace('Nissan', 'nissan',inplace=True)

cars_clean['car_company'].replace('maxda', 'mazda',inplace=True)

cars_clean['car_company'].replace('vokswagen', 'volkswagen',inplace=True)

cars_clean['car_company'].replace('vw', 'volkswagen',inplace=True)

cars_clean['car_company'].replace('porcshce', 'porsche',inplace=True)
#rechecking the data:

cars_clean['car_company'].value_counts()
# fueltype - Car fuel type i.e gas or diesel

cars_clean['fueltype'].value_counts()
#aspiration - Aspiration used in a car

cars_clean['aspiration'].value_counts()
#doornumber - Number of doors in a car

cars_clean['doornumber'].value_counts()
def number_(x):

    return x.map({'four':4, 'two': 2})

    

cars_clean['doornumber'] = cars_clean[['doornumber']].apply(number_)
#rechecking

cars_clean['doornumber'].value_counts()
#carbody- body of car

cars_clean['carbody'].value_counts()
#drivewheel - type of drive wheel

cars_clean['drivewheel'].value_counts()
#enginelocation - Location of car engine

cars_clean['enginelocation'].value_counts()
#wheelbase - Weelbase of car 

cars_clean['wheelbase'].value_counts().head()
sns.distplot(cars_clean['wheelbase'])

plt.show()
#carlength - Length of car

cars_clean['carlength'].value_counts().head()
sns.distplot(cars_clean['carlength'])

plt.show()
#enginetype - Type of engine.

cars_clean['enginetype'].value_counts()
#cylindernumber- cylinder placed in the car

cars_clean['cylindernumber'].value_counts()
def convert_number(x):

    return x.map({'two':2, 'three':3, 'four':4,'five':5, 'six':6,'eight':8,'twelve':12})



cars_clean['cylindernumber'] = cars_clean[['cylindernumber']].apply(convert_number)
#re-checking

cars_clean['cylindernumber'].value_counts()
#fuelsystem - Fuel system of car

cars_clean['fuelsystem'].value_counts()
cars_visual = cars_clean.select_dtypes(include =['int64','float64'])

cars_visual.head()
plt.figure(figsize = (30,30))

sns.pairplot(cars_visual)

plt.show()
plt.figure(figsize = (20,20))

sns.heatmap(cars_clean.corr(), annot = True ,cmap = 'YlGnBu')

plt.show()
cars_categ = cars_clean.select_dtypes(include = ['object'])

cars_categ.head()
plt.figure(figsize = (20,12))

plt.subplot(3,3,1)

sns.boxplot(x = 'fueltype', y = 'price', data = cars_clean)

plt.subplot(3,3,2)

sns.boxplot(x = 'aspiration', y = 'price', data = cars_clean)

plt.subplot(3,3,3)

sns.boxplot(x = 'carbody', y = 'price', data = cars_clean)

plt.subplot(3,3,4)

sns.boxplot(x = 'drivewheel', y = 'price', data = cars_clean)

plt.subplot(3,3,5)

sns.boxplot(x = 'enginelocation', y = 'price', data = cars_clean)

plt.subplot(3,3,6)

sns.boxplot(x = 'enginetype', y = 'price', data = cars_clean)

plt.subplot(3,3,7)

sns.boxplot(x = 'fuelsystem', y = 'price', data = cars_clean)
plt.figure(figsize = (20,12))

sns.boxplot(x = 'car_company', y = 'price', data = cars_clean)
#creating dummies

cars_prep = pd.get_dummies(cars_categ, drop_first = True)

cars_prep.head()
cars_df  = pd.concat([cars_clean, cars_prep], axis =1)
cars_df = cars_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',

       'enginetype', 'fuelsystem', 'car_company'], axis =1)
cars_df.info()
df_train, df_test = train_test_split(cars_df, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
cars_visual.columns
col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',

            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
scaler = StandardScaler()
df_train[col_list] = scaler.fit_transform(df_train[col_list])
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
lr = LinearRegression()

lr.fit(X_train,y_train)



# Subsetting training data for 15 selected columns

rfe = RFE(lr,15)

rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
cols = X_train.columns[rfe.support_]

cols
X1 = X_train[cols]

X1_sm = sm.add_constant(X1)



lr_1 = sm.OLS(y_train,X1_sm).fit()
print(lr_1.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X1.columns

vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
lr2 = LinearRegression()



rfe2 = RFE(lr2,10)

rfe2.fit(X_train,y_train)
list(zip(X_train.columns,rfe2.support_,rfe2.ranking_))
supported_cols = X_train.columns[rfe2.support_]

supported_cols 
X2 = X_train[supported_cols]

X2_sm = sm.add_constant(X2)



model_2 = sm.OLS(y_train,X2_sm).fit()
print(model_2.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X2.columns

vif['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
X3 = X2.drop(['car_company_subaru'], axis =1)

X3_sm = sm.add_constant(X3)



Model_3 = sm.OLS(y_train,X3_sm).fit()
print(Model_3.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X3.columns

vif['VIF'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
X4 = X3.drop(['enginetype_ohcf'], axis =1)

X4_sm = sm.add_constant(X4)



Model_4 = sm.OLS(y_train,X4_sm).fit()
print(Model_4.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X4.columns

vif['VIF'] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
X5 = X4.drop(['car_company_peugeot'], axis =1)

X5_sm = sm.add_constant(X5)



Model_5 = sm.OLS(y_train,X5_sm).fit()
print(Model_5.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X5.columns

vif['VIF'] = [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
X6 = X5.drop(['enginetype_l'], axis =1)

X6_sm = sm.add_constant(X6)



Model_6 = sm.OLS(y_train,X6_sm).fit()
print(Model_6.summary())
#VIF

vif = pd.DataFrame()

vif['Features'] = X6.columns

vif['VIF'] = [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
y_train_pred = Model_6.predict(X6_sm)

y_train_pred.head()
Residual = y_train- y_train_pred
sns.distplot(Residual, bins =15)
df_test[col_list] = scaler.transform(df_test[col_list])
y_test = df_test.pop('price')

X_test = df_test
final_cols = X6.columns
X_test_model6= X_test[final_cols]

X_test_model6.head()
X_test_sm = sm.add_constant(X_test_model6)
y_pred = Model_6.predict(X_test_sm)
y_pred.head()
c = [i for i in range(1,63,1)]

plt.plot(c, y_test,color = 'Blue')

plt.plot(c, y_pred,color = 'red')

plt.xlabel("X")

plt.ylabel("Y")
plt.scatter(y_test, y_pred)

plt.xlabel('y_test')

plt.ylabel('y_pred')
r_squ = r2_score(y_test,y_pred)

r_squ