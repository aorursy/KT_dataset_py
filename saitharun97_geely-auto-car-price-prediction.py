# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
geely_car = pd.read_csv("../input/geely-auto-car-price-dataset/CarPrice_Assignment.csv")
# Check the head of the dataset

geely_car.head()
geely_car.shape
geely_car.describe()
geely_car.info()
# Lets look for missing values

geely_car.isnull().sum()
#Checking for duplicates

geely_car.loc[geely_car.duplicated()]
geely_car['symboling'].value_counts()
geely_car['symboling'].dtype
#convert some columns into categorical datatype



cat_col = ['symboling','CarName','fueltype','aspiration','doornumber','carbody',

       'drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']



for col in cat_col:

    geely_car[col] = geely_car[col].astype('category')
#lets drop id column as it is not useful

geely_car.drop('car_ID',axis=1,inplace=True)
num_col = geely_car.select_dtypes(exclude='category')
num_col.columns
import matplotlib.pyplot as plt

import seaborn as sns
# Lets look at the price column distribution

sns.boxplot(geely_car['price'])
sns.distplot(geely_car['price'])
geely_car.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1])
# Lets look at the probability plot

from scipy import stats

from scipy.stats import norm



probplot = stats.probplot(geely_car['price'],plot=plt)
sns.pairplot(num_col)

plt.show()
plt.figure(figsize=(20, 12))

sns.heatmap(geely_car.corr(),annot=True)
plt.figure(figsize=(40, 30))

plt.subplot(4,3,1)

sns.boxplot(x = 'symboling', y = 'price', data = geely_car)

plt.subplot(4,3,2)

sns.boxplot(x = 'fueltype', y = 'price', data =  geely_car)

plt.subplot(4,3,3)

sns.boxplot(x = 'aspiration', y = 'price',data = geely_car)

plt.subplot(4,3,4)

sns.boxplot(x = 'doornumber', y = 'price',data = geely_car)

plt.subplot(4,3,5)

sns.boxplot(x = 'carbody', y = 'price', data =   geely_car)

plt.subplot(4,3,6)

sns.boxplot(x = 'drivewheel', y = 'price', data =   geely_car)

plt.subplot(4,3,7)

sns.boxplot(x = 'enginelocation', y = 'price', data =   geely_car)

plt.subplot(4,3,8)

sns.boxplot(x = 'enginetype', y = 'price', data =   geely_car)

plt.subplot(4,3,9)

sns.boxplot(x = 'cylindernumber', y = 'price', data =   geely_car)

plt.subplot(4,3,10)

sns.boxplot(x = 'fuelsystem', y = 'price', data =   geely_car)

plt.show()
#Fuel economy

geely_car['fueleconomy'] = (0.55 * geely_car['citympg']) + (0.45 * geely_car['highwaympg'])
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'symboling', y = 'price', hue = 'drivewheel', data = geely_car)

plt.show()
geely_car.columns
# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df



# Applying the function to the geely_car

geely_car = dummies('fueltype',geely_car)

geely_car = dummies('aspiration',geely_car)

geely_car = dummies('carbody',geely_car)

geely_car = dummies('drivewheel',geely_car)

geely_car = dummies('enginetype',geely_car)

geely_car = dummies('cylindernumber',geely_car)

geely_car = dummies('enginelocation',geely_car)

geely_car = dummies('fuelsystem',geely_car)
geely_car.head()
geely_car.shape
geely_car.drop('CarName',axis=1,inplace=True)
geely_car.drop('symboling',axis=1,inplace=True)
geely_car.drop(['mfi','spfi','doornumber'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(geely_car, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train.head()
df_train.columns
#Scaling required for columns

num_col = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',

       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',

       'peakrpm', 'citympg', 'highwaympg','price']
df_train[num_col] = scaler.fit_transform(df_train[num_col])
df_train.head()
df_train.describe()
#Correlation using heatmap

plt.figure(figsize = (30, 25))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
y_train = df_train.pop('price')

X_train = df_train
import statsmodels.api as sm



# Add a constant

X_train_lm = sm.add_constant(X_train[['curbweight']])



# Create a first fitted model

lr = sm.OLS(y_train, X_train_lm).fit()
# Check the parameters obtained



lr.params
# Let's visualise the data with a scatter plot and the fitted regression line

plt.scatter(X_train_lm.iloc[:, 1], y_train)

plt.plot(X_train_lm.iloc[:, 1], -0.139568 + 0.879864*X_train_lm.iloc[:, 1], 'r')

plt.show()
# Print a summary of the linear regression model obtained

print(lr.summary())
# Assign all the feature variables to X

X_train_lm = X_train[['curbweight', 'enginesize']]
# Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)



lr = sm.OLS(y_train, X_train_lm).fit()



lr.params
# Check the summary

print(lr.summary())
# Assign all the feature variables to X

X_train_lm = X_train[['curbweight', 'enginesize','horsepower']]
# Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)



lr = sm.OLS(y_train, X_train_lm).fit()



lr.params
# Print the summary of the model

print(lr.summary())
# Assign all the feature variables to X

X_train_lm = X_train[['curbweight', 'enginesize','horsepower','carwidth']]
# Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)



lr = sm.OLS(y_train, X_train_lm).fit()



lr.params
# Print the summary of the model

print(lr.summary())
# Check all the columns of the dataframe



X_train = df_train.copy()
#Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train)



lr_1 = sm.OLS(y_train, X_train_lm).fit()



lr_1.params
print(lr_1.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X = X_train.drop('six', 1,)
X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model

print(lr_2.summary())
# Calculate the VIFs again for the new model



vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables

X = X.drop('compressionratio', 1)
# Build a second fitted model

X_train_lm = sm.add_constant(X)



lr_3 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model



print(lr_3.summary())
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train = df_train
#RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
def VIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
VIF(X_train_rfe)
# Scaling test columns

df_test[num_col] = scaler.transform(df_test[num_col])
#Dividing into X and y

y_test = df_test.pop('price')

X_test = df_test
from sklearn.metrics import r2_score 

def eval(x):

    # Now let's use our model to make predictions.

    X_train_new = X_train_rfe.drop('const',axis=1)

    # Creating X_test_new dataframe by dropping variables from X_test

    X_test_new = X_test[X_train_new.columns]



    # Adding a constant variable 

    X_test_new = sm.add_constant(X_test_new)



    # Making predictions

    y_pred = lm.predict(X_test_new)



    if x=='score':

        print(r2_score(y_test, y_pred))

        

    elif x=='TvsP':

        #EVALUATION OF THE MODEL

        # Plotting y_test and y_pred to understand the spread.

        fig = plt.figure()

        plt.scatter(y_test,y_pred)

        fig.suptitle('y_test vs y_pred', fontsize=20)             

        plt.xlabel('y_test', fontsize=18)                          

        plt.ylabel('y_pred', fontsize=16)   

    

    elif x=='AvsP':

        # Actual vs Predicted

        c = [i for i in range(1,63)]

        fig = plt.figure()

        plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-",label='Actual')     #Plotting Actual

        plt.plot(c,y_pred, color="red",  linewidth=3.5, linestyle="-",label='Predicted')  #Plotting predicted

        fig.suptitle('Actual and Predicted', fontsize=20)            

        plt.ylabel('Car Price', fontsize=16)  

        plt.legend()

    

    elif x=='error':

        # Error terms

        fig = plt.figure()

        c = [i for i in range(1,63,1)]

        plt.scatter(c,y_test-y_pred)

        fig.suptitle('Error Terms', fontsize=20)            

        plt.ylabel('predicted', fontsize=16)     

        

    elif x=='e_dist':

        # Plot the histogram of the error terms

        mean = np.array(y_pred - y_test).mean()

        fig = plt.figure()

        sns.distplot((y_pred - y_test), bins = 20)

        plt.axvline(mean,color='b', linestyle='--')

        fig.suptitle('Error Terms', fontsize = 20)                  

        plt.xlabel('Errors', fontsize = 18)     

        plt.legend({'Mean':mean})
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
X_train_rfe = X_train_rfe.drop('curbweight',axis=1)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model

print(lm.summary())
VIF(X_train_rfe)
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
X_train_rfe.drop('rotor',axis=1,inplace=True)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model

print(lm.summary())
VIF(X_train_rfe)
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
X_train_rfe.drop('horsepower',axis=1,inplace=True)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model

print(lm.summary())
VIF(X_train_rfe)
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
X_train_rfe.drop('three',axis=1,inplace=True)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model

print(lm.summary())
VIF(X_train_rfe)
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
X_train_rfe.drop('stroke',axis=1,inplace=True)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

#Let's see the summary of our linear model

print(lm.summary())
eval('score')
eval('TvsP')
eval('AvsP')
eval('error')
eval('e_dist')
y_train_price = lm.predict(X_train_rfe)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  

plt.xlabel('Errors', fontsize = 18)                     