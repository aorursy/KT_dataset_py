# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
# Display all columns and rows

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Load data from CarPrice_Assignment.csv dataset

car_df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv", engine='python')
# Check the head of the dataset

car_df.head()
# Check for null values

car_df.isnull().sum()
# Dataset dimensions

car_df.shape
# Dataset information

car_df.info()
# More understanding about the dataset

car_df.describe()
import statsmodels.api as sm
# Function to get VIF (Variation Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_VIF(X_train):
    # A dataframe that will contain the names of all the feature variables and their respective VIFs
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
# Creating a derived column for company name of cars from the column CarName

car_df.loc[:,'company'] = car_df.CarName.str.split(' ').str[0]
car_df.company = car_df.company.apply(lambda x: str(x).lower())
car_df.company.unique()
car_df['company'].replace('maxda','mazda',inplace=True)
car_df['company'].replace('porcshce','porsche',inplace=True)
car_df['company'].replace('toyouta','toyota',inplace=True)
car_df['company'].replace(['vokswagen','vw'],'volkswagen',inplace=True)
# Dropping the CarName column

car_df.drop(columns = 'CarName', inplace=True)
car_df.fuelsystem.unique()
car_df['fuelsystem'].replace('mfi','mpfi',inplace=True)
car_df.enginetype.unique()
car_df['enginetype'].replace('dohcv','dohc',inplace = True)
car_df['enginetype'].replace('ohcv','ohc',inplace = True)
car_df.drivewheel.unique()
car_df['drivewheel'].replace('4wd', 'fwd', inplace = True)
import matplotlib.pyplot as plt
import seaborn as sns
# Plotting a paiplot for the continuous variables

sns.pairplot(car_df, diag_kind="kde")
plt.show()
plt.figure(figsize=(20,12))
sns.heatmap(car_df.corr(), linewidths=.5, annot=True, cmap="YlGnBu")
# curbweight/enginesize

car_df.loc[:,'curbweight/enginesize'] = car_df.curbweight/car_df.enginesize
# enginesize/horsepower

car_df.loc[:,'enginesize/horsepower'] = car_df.enginesize/car_df.horsepower
# carwidth/carlength

car_df.loc[:,'carwidth/carlength'] = car_df.carwidth/car_df.carlength
# highwaympg/citympg

car_df.loc[:,'highway/city'] = car_df.highwaympg/car_df.citympg
# We can now drop the corresponding columns as we have taken a ratio.

car_df.drop(columns = ['enginesize','carwidth', 'carlength', 'highwaympg', 'citympg'], inplace = True)
# Checking the dataset once more

car_df.head()
# Dropping car_ID column as it is not useful

car_df.drop(columns = 'car_ID', inplace=True)
car_df.symboling = car_df.symboling.map({-3: 'safe', -2: 'safe',-1: 'safe',0: 'moderate',1: 'moderate',2: 'risky',3:'risky'})
# Visualizing categorical data via boxplots

plt.figure(figsize=(20, 16))
plt.subplot(3,3,1)
sns.boxplot(x = 'symboling', y = 'price', data = car_df)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = car_df)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = car_df)
plt.subplot(3,3,4)
sns.boxplot(x = 'doornumber', y = 'price', data = car_df)
plt.subplot(3,3,5)
sns.boxplot(x = 'carbody', y = 'price', data = car_df)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = car_df)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginelocation', y = 'price', data = car_df)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = car_df)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = car_df)
plt.show()
# Plotting company vs price

plt.figure(figsize=(20, 16))
sns.boxplot(x = 'company', y = 'price', data = car_df, palette="Reds")
median_dict = car_df.groupby(['company'])[['price']].median().to_dict()
median_dict = median_dict['price']
median_dict
dict_keys = list(median_dict.keys())

# Median price of category below 10000 is low, between 10000 and 20000 is med and above 20000 is high
for i in dict_keys:
    if median_dict[i] < 10000:
        median_dict[i] = 'low'
    elif median_dict[i] >= 10000 and median_dict[i] <= 20000:
        median_dict[i] = 'med'
    else:
        median_dict[i] = 'high'

median_dict
car_df.company = car_df.company.map(median_dict)
car_df.company.unique()
car_df = pd.get_dummies(car_df, drop_first=True)
# Checking dataframe after dummy variable creation

car_df.head()
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)
print("Train data shape: ", df_train.shape)
print("Test data shape: ", df_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
conti_vars = ['wheelbase', 'carheight', 'boreratio', 'stroke', 'compressionratio', 'peakrpm', 'horsepower', 'curbweight', 'price', 'curbweight/enginesize', 'carwidth/carlength', 'highway/city', 'enginesize/horsepower']
df_train[conti_vars] = scaler.fit_transform(df_train[conti_vars])

df_train.describe()
# X and y division

y_train = df_train.pop('price')
X_train = df_train
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE to select 10 best features
rfe = rfe.fit(X_train, y_train)
# Checking the statistics of the model using statsmodel library

col_rfe = X_train.columns[rfe.support_]
X_train = X_train[col_rfe]

X_train_sm = sm.add_constant(X_train)
lm_1 = sm.OLS(y_train, X_train_sm).fit()
print(lm_1.summary()) #stats
get_VIF(X_train_sm) #VIF
X_train.drop(columns='carbody_hardtop', inplace=True)
X_train_sm = sm.add_constant(X_train)
lm_2 = sm.OLS(y_train, X_train_sm).fit()
print(lm_2.summary()) #stats
get_VIF(X_train_sm) #VIF
X_train.drop(columns='wheelbase', inplace=True)
X_train_sm = sm.add_constant(X_train)
lm_3 = sm.OLS(y_train, X_train_sm).fit()
print(lm_3.summary()) #stats
get_VIF(X_train_sm) #VIF
X_train.drop(columns='carbody_sedan', inplace=True)
X_train_sm = sm.add_constant(X_train)
lm_4 = sm.OLS(y_train, X_train_sm).fit()
print(lm_4.summary()) #stats
get_VIF(X_train_sm) #VIF
y_train_price = lm_4.predict(X_train_sm)
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Residual Error Distribution', fontsize = 20)
# We are scaling the testing set with the already existing scaler object which has been fitted on the train dataset

df_test[conti_vars] = scaler.transform(df_test[conti_vars])

df_test.describe()
# X and y division

y_test = df_test.pop('price')
X_test = df_test
X_test = X_test[col_rfe]
X_test.drop(columns=['carbody_sedan', 'wheelbase', 'carbody_hardtop'], inplace=True) # Dropping columns which we dropped while building the model after RFE
X_test_sm = sm.add_constant(X_test)
y_pred = lm_4.predict(X_test_sm)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel('y_test_price', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Model RMSE:',rmse)

r2=r2_score(y_test, y_pred)
print('Model r2_score:',r2)
c = [i for i in range(1,63)]

fig = plt.figure()
plt.plot(c,y_test,color="blue",linewidth=3,linestyle='-')
plt.plot(c,y_pred,color="red",linewidth=3,linestyle='-')
plt.ylabel('Car Price')
plt.xlabel('Index')
