# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker



# import all libraries and dependencies for machine learning

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score
# Local file path. Please change the file path accordingly



path = '../input/car-price-prediction/'

file = path + 'CarPrice_Assignment.csv'

file1 = path+ 'Data Dictionary - carprices.xlsx'
# Reading the automobile consulting company file on which analysis needs to be done



df_auto = pd.read_csv(file)



df_auto.head()
# Reading the data dictionary file



df_stru = pd.read_excel(file1)

df_stru.head(2)
# shape of the data

df_auto.shape
# information of the data

df_auto.info()
# description of the data

df_auto.describe()
# dropping car_ID based on business knowledge



df_auto = df_auto.drop('car_ID',axis=1)
# Calculating the Missing Values % contribution in DF



df_null = df_auto.isna().mean().round(4) * 100



df_null.sort_values(ascending=False).head()
# Datatypes

df_auto.dtypes
# Outlier Analysis of target variable with maximum amount of Inconsistency



outliers = ['price']

plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data=df_auto[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Price Range", fontweight = 'bold')

plt.xlabel("Continuous Variable", fontweight = 'bold')

df_auto.shape
# Extracting Car Company from the CarName as per direction in Problem 



df_auto['CarName'] = df_auto['CarName'].str.split(' ',expand=True)
# Unique Car company



df_auto['CarName'].unique()
# Renaming the typo errors in Car Company names



df_auto['CarName'] = df_auto['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 

                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})
# changing the datatype of symboling as it is categorical variable as per dictionary file



df_auto['symboling'] = df_auto['symboling'].astype(str)
# checking for duplicates



df_auto.loc[df_auto.duplicated()]
# Segregation of Numerical and Categorical Variables/Columns



cat_col = df_auto.select_dtypes(include=['object']).columns

num_col = df_auto.select_dtypes(exclude=['object']).columns

df_cat = df_auto[cat_col]

df_num = df_auto[num_col]
# Visualizing the different car names available



plt.rcParams['figure.figsize'] = [15,8]

ax=df_auto['CarName'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1')

ax.title.set_text('CarName')

plt.xlabel("Names of the Car",fontweight = 'bold')

plt.ylabel("Count of Cars",fontweight = 'bold')
plt.figure(figsize=(8,8))



plt.title('Car Price Distribution Plot')

sns.distplot(df_auto['price'])
ax = sns.pairplot(df_auto[num_col])
plt.figure(figsize=(20, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'doornumber', y = 'price', data = df_auto)

plt.subplot(3,3,2)

sns.boxplot(x = 'fueltype', y = 'price', data = df_auto)

plt.subplot(3,3,3)

sns.boxplot(x = 'aspiration', y = 'price', data = df_auto)

plt.subplot(3,3,4)

sns.boxplot(x = 'carbody', y = 'price', data = df_auto)

plt.subplot(3,3,5)

sns.boxplot(x = 'enginelocation', y = 'price', data = df_auto)

plt.subplot(3,3,6)

sns.boxplot(x = 'drivewheel', y = 'price', data = df_auto)

plt.subplot(3,3,7)

sns.boxplot(x = 'enginetype', y = 'price', data = df_auto)

plt.subplot(3,3,8)

sns.boxplot(x = 'cylindernumber', y = 'price', data = df_auto)

plt.subplot(3,3,9)

sns.boxplot(x = 'fuelsystem', y = 'price', data = df_auto)

plt.show()
plt.figure(figsize=(25, 6))



plt.subplot(1,3,1)

plt1 = df_auto['cylindernumber'].value_counts().plot('bar')

plt.title('Number of cylinders')

plt1.set(xlabel = 'Number of cylinders', ylabel='Frequency of Number of cylinders')



plt.subplot(1,3,2)

plt1 = df_auto['fueltype'].value_counts().plot('bar')

plt.title('Fuel Type')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of Fuel type')



plt.subplot(1,3,3)

plt1 = df_auto['carbody'].value_counts().plot('bar')

plt.title('Car body')

plt1.set(xlabel = 'Car Body', ylabel='Frequency of Car Body')
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'fuelsystem', y = 'price', hue = 'fueltype', data = df_auto)

plt.show()
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'carbody', y = 'price', hue = 'enginelocation', data = df_auto)

plt.show()
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'cylindernumber', y = 'price', hue = 'fueltype', data = df_auto)

plt.show()
plt.figure(figsize=(20, 6))



df_autox = pd.DataFrame(df_auto.groupby(['CarName'])['price'].mean().sort_values(ascending = False))

df_autox.plot.bar()

plt.title('Car Company Name vs Average Price')

plt.show()
plt.figure(figsize=(20, 6))



df_autoy = pd.DataFrame(df_auto.groupby(['carbody'])['price'].mean().sort_values(ascending = False))

df_autoy.plot.bar()

plt.title('Car Company Name vs Average Price')

plt.show()
#Binning the Car Companies based on avg prices of each car Company.



df_auto['price'] = df_auto['price'].astype('int')

df_auto_temp = df_auto.copy()

t = df_auto_temp.groupby(['CarName'])['price'].mean()

df_auto_temp = df_auto_temp.merge(t.reset_index(), how='left',on='CarName')

bins = [0,10000,20000,40000]

label =['Budget_Friendly','Medium_Range','TopNotch_Cars']

df_auto['Cars_Category'] = pd.cut(df_auto_temp['price_y'],bins,right=False,labels=label)

df_auto.head()
sig_col = ['price','Cars_Category','enginetype','fueltype', 'aspiration','carbody','cylindernumber', 'drivewheel',

            'wheelbase','curbweight', 'enginesize', 'boreratio','horsepower', 

                    'citympg','highwaympg', 'carlength','carwidth']
df_auto = df_auto[sig_col]
sig_cat_col = ['Cars_Category','fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber']
# Get the dummy variables for the categorical feature and store it in a new variable - 'dummies'



dummies = pd.get_dummies(df_auto[sig_cat_col])

dummies.shape
dummies = pd.get_dummies(df_auto[sig_cat_col], drop_first = True)

dummies.shape
# Add the results to the original dataframe



df_auto = pd.concat([df_auto, dummies], axis = 1)
# Drop the original cat variables as dummies are already created



df_auto.drop( sig_cat_col, axis = 1, inplace = True)

df_auto.shape
df_auto
# We specify this so that the train and test data set always have the same rows, respectively

# We divide the df into 70/30 ratio



np.random.seed(0)

df_train, df_test = train_test_split(df_auto, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.head()
scaler = preprocessing.StandardScaler()
sig_num_col = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']
# Apply scaler() to all the columns except the 'dummy' variables

import warnings

warnings.filterwarnings("ignore")



df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])
df_train.head()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (20, 20))

sns.heatmap(df_train.corr(), cmap="RdYlGn")

plt.show()
col = ['highwaympg','citympg','horsepower','enginesize','curbweight','carwidth']
# Scatter Plot of independent variables vs dependent variables



fig,axes = plt.subplots(2,3,figsize=(18,15))

for seg,col in enumerate(col):

    x,y = seg//3,seg%3

    an=sns.scatterplot(x=col, y='price' ,data=df_auto, ax=axes[x,y])

    plt.setp(an.get_xticklabels(), rotation=45)

   

plt.subplots_adjust(hspace=0.5)
y_train = df_train.pop('price')

X_train = df_train
X_train_1 = X_train['horsepower']
# Add a constant

X_train_1c = sm.add_constant(X_train_1)



# Create a first fitted model

lr_1 = sm.OLS(y_train, X_train_1c).fit()
# Check parameters created



lr_1.params
# Let's visualise the data with a scatter plot and the fitted regression line



plt.scatter(X_train_1c.iloc[:, 1], y_train)

plt.plot(X_train_1c.iloc[:, 1], 0.8062*X_train_1c.iloc[:, 1], 'r')

plt.show()
# Print a summary of the linear regression model obtained

print(lr_1.summary())
X_train_2 = X_train[['horsepower', 'curbweight']]
# Add a constant

X_train_2c = sm.add_constant(X_train_2)



# Create a second fitted model

lr_2 = sm.OLS(y_train, X_train_2c).fit()
lr_2.params
print(lr_2.summary())
X_train_3 = X_train[['horsepower', 'curbweight', 'enginesize']]
# Add a constant

X_train_3c = sm.add_constant(X_train_3)



# Create a third fitted model

lr_3 = sm.OLS(y_train, X_train_3c).fit()
lr_3.params
print(lr_3.summary())
# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Selecting the variables which are in support



col_sup = X_train.columns[rfe.support_]

col_sup
# Creating X_train dataframe with RFE selected variables



X_train_rfe = X_train[col_sup]
# Adding a constant variable and Build a first fitted model

import statsmodels.api as sm  

X_train_rfec = sm.add_constant(X_train_rfe)

lm_rfe = sm.OLS(y_train,X_train_rfec).fit()



#Summary of linear model

print(lm_rfe.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe1 = X_train_rfe.drop('cylindernumber_twelve', 1,)



# Adding a constant variable and Build a second fitted model



X_train_rfe1c = sm.add_constant(X_train_rfe1)

lm_rfe1 = sm.OLS(y_train, X_train_rfe1c).fit()



#Summary of linear model

print(lm_rfe1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe1.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe1.values, i) for i in range(X_train_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe2 = X_train_rfe1.drop('cylindernumber_six', 1,)



# Adding a constant variable and Build a third fitted model



X_train_rfe2c = sm.add_constant(X_train_rfe2)

lm_rfe2 = sm.OLS(y_train, X_train_rfe2c).fit()



#Summary of linear model

print(lm_rfe2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe3 = X_train_rfe2.drop('carbody_hardtop', 1,)



# Adding a constant variable and Build a fourth fitted model

X_train_rfe3c = sm.add_constant(X_train_rfe3)

lm_rfe3 = sm.OLS(y_train, X_train_rfe3c).fit()



#Summary of linear model

print(lm_rfe3.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe3.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe3.values, i) for i in range(X_train_rfe3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe4 = X_train_rfe3.drop('enginetype_ohc', 1,)



# Adding a constant variable and Build a fifth fitted model

X_train_rfe4c = sm.add_constant(X_train_rfe4)

lm_rfe4 = sm.OLS(y_train, X_train_rfe4c).fit()



#Summary of linear model

print(lm_rfe4.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe4.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe4.values, i) for i in range(X_train_rfe4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe5 = X_train_rfe4.drop('cylindernumber_five', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe5c = sm.add_constant(X_train_rfe5)

lm_rfe5 = sm.OLS(y_train, X_train_rfe5c).fit()



#Summary of linear model

print(lm_rfe5.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe5.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe5.values, i) for i in range(X_train_rfe5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe6 = X_train_rfe5.drop('enginetype_ohcv', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe6c = sm.add_constant(X_train_rfe6)

lm_rfe6 = sm.OLS(y_train, X_train_rfe6c).fit()



#Summary of linear model

print(lm_rfe6.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe6.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe6.values, i) for i in range(X_train_rfe6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe7 = X_train_rfe6.drop('curbweight', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe7c = sm.add_constant(X_train_rfe7)

lm_rfe7 = sm.OLS(y_train, X_train_rfe7c).fit()



#Summary of linear model

print(lm_rfe7.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe7.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe7.values, i) for i in range(X_train_rfe7.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe8 = X_train_rfe7.drop('cylindernumber_four', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe8c = sm.add_constant(X_train_rfe8)

lm_rfe8 = sm.OLS(y_train, X_train_rfe8c).fit()



#Summary of linear model

print(lm_rfe8.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe8.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe8.values, i) for i in range(X_train_rfe8.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe9 = X_train_rfe8.drop('carbody_sedan', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe9c = sm.add_constant(X_train_rfe9)

lm_rfe9 = sm.OLS(y_train, X_train_rfe9c).fit()



#Summary of linear model

print(lm_rfe9.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe9.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe9.values, i) for i in range(X_train_rfe9.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X_train_rfe10 = X_train_rfe9.drop('carbody_wagon', 1,)



# Adding a constant variable and Build a sixth fitted model

X_train_rfe10c = sm.add_constant(X_train_rfe10)

lm_rfe10 = sm.OLS(y_train, X_train_rfe10c).fit()



#Summary of linear model

print(lm_rfe10.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe10.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe10.values, i) for i in range(X_train_rfe10.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Predicting the price of training set.

y_train_price = lm_rfe10.predict(X_train_rfe10c)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms Analysis', fontsize = 20)                   

plt.xlabel('Errors', fontsize = 18)
import warnings

warnings.filterwarnings("ignore")



df_test[sig_num_col] = scaler.transform(df_test[sig_num_col])

df_test.shape
y_test = df_test.pop('price')

X_test = df_test
# Adding constant

X_test_1 = sm.add_constant(X_test)



X_test_new = X_test_1[X_train_rfe10c.columns]
# Making predictions using the final model

y_pred = lm_rfe10.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred', fontsize=16)    
r2_score(y_test, y_pred)
# Predicting the price of training set.

y_train_price2 = lm_rfe8.predict(X_train_rfe8c)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price2), bins = 20)

fig.suptitle('Error Terms Analysis', fontsize = 20)                   

plt.xlabel('Errors', fontsize = 18)
X_test_2 = X_test_1[X_train_rfe8c.columns]
# Making predictions using the final model

y_pred2 = lm_rfe8.predict(X_test_2)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred2)

fig.suptitle('y_test vs y_pred2', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred2', fontsize=16)    
r2_score(y_test, y_pred2)