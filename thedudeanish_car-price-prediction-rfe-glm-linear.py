#Importing the required modules and packages



import matplotlib.pyplot as plt

from numpy.random import randn

from numpy.random import seed

from numpy import percentile

from scipy import stats

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline

import warnings
# Removing the minimum display columns to 500

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



# Ignoring warnings

warnings.filterwarnings("ignore")
# Importing the required csv from the folder:

carData = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
# Sense check of the application data



carData.head()
# Checking the top 5 rows and headers of the data

carData.head()
# Looking at the type of the data frame, data types and the number of rows

carData.info()
# Checking the number of rows and columns present in the data

carData.shape
# Looking at the data types of the data

carData.dtypes
# Making a copy of the application in dataframe df (checkpoint!) 

df = carData.copy(deep=False)
# Calculating the percent of missing values in the dataframe

percentMissing = (df.isnull().sum() / len(df)) * 100



# Making a dataframe with the missing values % and columns into a dataframe (on account of large number of rows) 

missingValuesDf = pd.DataFrame({'columnName': df.columns,

                                 'percentMissing': percentMissing})
# Viewing the dataframe to ensure that the values have been populated correctly

missingValuesDf
# Selecting only the numeric columns to perform correlation analysis

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df_num = df.select_dtypes(include=numerics)
# Displaying the top 5 rows of only the numerical values

df_num.head()
# Removal of the categorical columns

symboling = df_num.pop('symboling')

car_ID = df_num.pop('car_ID')
# Making boxplots as sub-plots to understand the trend of the data 

plt.figure(figsize=(15, 6))

plt.subplot(2,3,1)

sns.boxplot(x = 'wheelbase', data = df_num)

plt.subplot(2,3,2)

sns.boxplot(x = 'carlength', data = df_num)

plt.subplot(2,3,3)

sns.boxplot(x = 'carwidth', data = df_num)

plt.subplot(2,3,4)

sns.boxplot(x = 'carheight', data = df_num)

plt.subplot(2,3,5)

sns.boxplot(x = 'curbweight', data = df_num)

plt.subplot(2,3,6)

sns.boxplot(x = 'enginesize', data = df_num)

plt.show()
# Making boxplots as sub-plots to understand the trend of the data 



plt.figure(figsize=(15, 6))

plt.subplot(2,3,1)

sns.boxplot(x = 'boreratio', data = df_num)

plt.subplot(2,3,2)

sns.boxplot(x = 'stroke', data = df_num)

plt.subplot(2,3,3)

sns.boxplot(x = 'compressionratio', data = df_num)

plt.subplot(2,3,4)

sns.boxplot(x = 'horsepower', data = df_num)

plt.subplot(2,3,5)

sns.boxplot(x = 'peakrpm', data = df_num)

plt.subplot(2,3,6)

sns.boxplot(x = 'citympg', data = df_num)

plt.show()
# Making boxplots as sub-plots to understand the trend of the data 

plt.figure(figsize=(15, 3))

plt.subplot(1,2,1)

sns.boxplot(x = 'highwaympg', data = df_num)

plt.subplot(1,2,2)

sns.boxplot(x = 'price', data = df_num)
# Function to plot histogram for numerical, univariate analysis

def plotHistogram(df, colName):

    '''

    This function is used to set the style of the plot, name the graph and plot the distribution for the specified column

    

    Inputs:

    @df (dataframe) - The dataframe for which histograms are to be plotted

    @colName (string) - The numeric column for which histograms is to be plotted

    

    Output:

    Titles distribution plot of specified colName

    '''

    sns.set(style="whitegrid")

    plt.figure(figsize=(20,5)) 

    plt.title(colName)

    plt.ylabel('Density', fontsize=14)

    sns.distplot(df[colName], kde=True)
# Making boxplots as sub-plots to understand the trend of the data 

plt.figure(figsize=(2, 20))

plotHistogram(df_num, 'wheelbase')

plotHistogram(df_num, 'carlength')

plotHistogram(df_num, 'carwidth')

plotHistogram(df_num, 'carheight')

plotHistogram(df_num, 'curbweight')

plotHistogram(df_num, 'enginesize')

plotHistogram(df_num, 'boreratio')

plotHistogram(df_num, 'stroke')

plotHistogram(df_num, 'compressionratio')

plotHistogram(df_num, 'horsepower')

plotHistogram(df_num, 'peakrpm')

plotHistogram(df_num, 'citympg')

plt.show()

# Defining a function to view the distribution of the categorical variables

def plotFrequencyTable(df, catColName):

    '''

    This function is used to plot the frequency table of the specified categorical variable

    @df (dataframe) - Dataframe for which frequency table is to be plotted

    @catColName (string) - Column name for which frequency table is to be plotted

    '''

    sns.countplot(x=catColName, data=df)

    plt.title(catColName)

    plt.xticks(rotation = 90)

    plt.show();
## Subsetting data to subset categorical variables

df_cat = df.select_dtypes(include='object')
# Viewing the head of the data for a sense-check

df_cat.head()
# Making boxplots as sub-plots to understand the trend of the data 



# plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)

plotFrequencyTable(df,'enginelocation')

plt.subplot(2,2,2)

plotFrequencyTable(df_cat,'fueltype')

plt.subplot(2,2,3)

plotFrequencyTable(df_cat,'aspiration')

plt.subplot(2,2,4)

plotFrequencyTable(df_cat,'doornumber')

plt.subplot(2,3,5)

plotFrequencyTable(df_cat,'carbody')

plt.subplot(2,3,6)

plotFrequencyTable(df_cat,'drivewheel')

plt.show()
# Plotting the correlation matrix of the data

cor = df_num.corr()
#Correlation with output variable

cor_target = abs(cor['price'])
#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

round(relevant_features.sort_values(ascending = True), 2)
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (16, 10))

sns.heatmap(df_num.corr(), annot = True, cmap="YlGnBu")

plt.show()
# Plotting pair plots

sns.pairplot(df)

plt.figure(figsize=(40, 40))

plt.show()
# Plotting the highly correlated variables with price to understand the trend

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.boxplot(x = 'enginesize', y = 'price', data = df)



plt.subplot(2,3,2)

sns.boxplot(x = 'curbweight', y = 'price', data = df)



plt.subplot(2,3,3)

sns.boxplot(x = 'horsepower', y = 'price', data = df)



plt.subplot(2,3,4)

sns.boxplot(x = 'carwidth', y = 'price', data = df)



plt.subplot(2,3,5)

sns.boxplot(x = 'highwaympg', y = 'price', data = df)



plt.subplot(2,3,6)

sns.boxplot(x = 'citympg', y = 'price', data = df)

plt.show()
# Removing the unique identifier of the data

df.pop('car_ID').head()
df.head()
# Getting the name of the brand

df['CarName'] = df['CarName'].str.split('-').str[0]

df['CarName'] = df['CarName'].str.split(' ').str[0]



# Converting to lowercase

df['CarName'] = df['CarName'].str.lower()



# Correcting the mistakes present in the CarName columns like vw to volkswagen, maxda to mazda etc.

df['CarName'] = df['CarName'].str.replace('vw','volkswagen')

df['CarName'] = df['CarName'].str.replace('maxda','mazda')

df['CarName'] = df['CarName'].str.replace('vokswagen','volkswagen')

df['CarName'] = df['CarName'].str.replace('toyouta','toyota')



# Replacing occurences of 4wd with fwd as they are the same thing

df['drivewheel'] = df['drivewheel'].str.replace('4wd','fwd')
# # Making a function to make dummy variables

# def makeDummyVariables(df, colName):

#     '''

#     This function is used to make dummy variables, concatenate it to original dataframe and remove the older categorical column.

    

#     Inputs:

#     @colName (string): Name of the categorical column that we wish to make dummy variables for.

#     @df (dataframe): Dataframe which we would like to make modifications in

    

#     Output:

#     Desired dataframe with dummy varibles with original categorical variable (colName) removed

    

#     '''

#     # Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

#     status = pd.get_dummies(df[colName], drop_first = True)

    

#     # Concatenating the dummy variables to the dataframe

#     df = pd.concat([df, status], axis = 1)

    

#     # Dropping the original categorical variable from the dataframe

#     df.drop([colName], axis = 1, inplace = True)





# ---------------- function did not work for some reason. However, an attempt was made------------------------------
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['CarName'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['CarName'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['fueltype'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['fueltype'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['aspiration'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['aspiration'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['symboling'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['symboling'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['doornumber'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['doornumber'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['carbody'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['carbody'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['drivewheel'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['drivewheel'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['enginelocation'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['enginelocation'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['enginetype'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['enginetype'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['cylindernumber'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['cylindernumber'], axis = 1, inplace = True)
# Making dummy variables and dropping the first dummy column as n values should have n-1 dummy columns

status = pd.get_dummies(df['fuelsystem'], drop_first = True)



# Concatenating the dummy variables to the dataframe

df = pd.concat([df, status], axis = 1)



# Dropping the original categorical variable from the dataframe

df.drop(['fuelsystem'], axis = 1, inplace = True)
# Sense check of the data

df.head()
# Shape of the data

df.shape
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Obtaining the numerical features and scaling them

num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 

            'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 

            'highwaympg']



# Scaling the numerical features

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



# Sense check of the data

df_train.head()
# Putting the dependent variable in 'y' 

y_train = df_train.pop('price')



# Putting the rest of the features in 'X'

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 20



# Making a linear regression model object

lm = LinearRegression()



# Fitting the model on the training dataset

lm.fit(X_train, y_train)



# Outputting the top 20 features

rfe = RFE(lm, 20)             

rfe = rfe.fit(X_train, y_train)
# listing the relevant features (obtained via Recursive Feature Elimination - RFE)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# The columns selected by RFE

col = X_train.columns[rfe.support_]

col
# The columns that were not selected by RFE

X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable as stats models does not have a constant variable

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model to understand the ordinary least squares

lm = sm.OLS(y_train,X_train_rfe).fit()   
#Let's see the summary of our linear model

print(lm.summary())
X_train_rfe = X_train_rfe.drop(["two"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
X_train_rfe = X_train_rfe.drop(["dohcv"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_rfe)



# Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()  



#Let's see the summary of our linear model

print(lm.summary())
X_train_rfe = X_train_rfe.drop(["peakrpm"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_rfe)



# Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()  



#Let's see the summary of our linear model

print(lm.summary())
X_train_rfe = X_train_rfe.drop(["horsepower"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_rfe)



# Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()  



#Let's see the summary of our linear model

print(lm.summary())
X_train_rfe.columns
X_train_rfe = X_train_rfe.drop(['const'], axis=1)
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Making a copy of the application in dataframe df (checkpoint!) 

X_train_new = X_train_rfe.copy(deep=False)
X_train_new = X_train_new.drop(["three"], axis = 1)
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



# Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()  



#Let's see the summary of our linear model

print(lm.summary())
# Viewing the final columns to be used in the model

X_train_new.columns
y_train_price = lm.predict(X_train_lm)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 

            'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 

            'highwaympg']



df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Adding a constant variable 

import statsmodels.api as sm  

X_test_lm = sm.add_constant(X_test_new)



# Running the linear model

lm = sm.OLS(y_test,X_test_lm).fit()  



#Let's see the summary of our linear model

print(lm.summary())
from sklearn.metrics import r2_score

print('R-Squared Score for the Car Price Prediction model with Linear Regression + RFE is:\n', round(r2_score(y_test, y_pred)*100, 2), '%')