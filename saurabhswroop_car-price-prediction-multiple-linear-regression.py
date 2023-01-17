# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
# Importing CarPrice_Assignment.csv
# Please make sure that the csv file is in the same folder as the python notebook otherwise this command wont work
carprice = pd.read_csv("../input/CarPrice_Assignment.csv")
# Check the head of the dataset
carprice.head()
carprice.shape
# Prining all the columns of the dataframe
carprice.columns.values
carprice.info()
carprice.describe()
carprice = carprice.drop('car_ID',axis=1)
carprice['symboling'] = carprice['symboling'].astype(str)
carprice.symboling.describe()
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = carprice.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (carprice.isnull().sum()/carprice.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
carprice_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
carprice_missing_data.head(15)
# Splitting the CarName column and only retaining the Car Company name
carprice["CarName"] = carprice["CarName"].str.split(" ", expand = True)
  
# df display 
carprice.head()
# Printing all the car names
carprice.CarName.unique()
# Renaming the typo errors in Car Names

carprice['CarName'] = carprice['CarName'].replace({'maxda': 'mazda','alfa-romero':'Alfa-Romeo', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})

carprice.CarName.unique()
#Checking for any duplicates in the data frame
carprice.loc[carprice.duplicated()]
import matplotlib.pyplot as plt
import seaborn as sns
# Initializing a figure
plt.figure(figsize=(20,8))

# Initializing a subplot
plt.subplot(1,2,1)
# setting the title of the plot
plt.title('Car Price Histogram')
# Plotting a Histogram for price column
sns.distplot(carprice.price)

# Initializing another subplot
plt.subplot(1,2,2)
# setting the title of the plot
plt.title('Car Price Box Plot')
# Plotting a boxplot for price column
sns.boxplot(y=carprice.price)

plt.show()
# Checking the various percentile values for the price column
print(carprice.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
# Finding all the numerical columns in the dataset. 
numCols = carprice.select_dtypes(include=['int64','float'])

# Sorting the columns
numCols = numCols[sorted(numCols.columns)]

# Printing the columns
print(numCols.columns)
# Initializing a figure
plt.figure(figsize=(15,30))

# Dropping the price column from the plot since we dont need to plot a scatter plot for price
numCols = numCols.drop('price',axis=1)

# running a for-loop to print the scatter plots for all numerical columns
for i in range(len(numCols.columns)):
    # Creating a sub plot
    plt.subplot(len(numCols.columns),2,i+1)
    # Creating a scatter plot
    plt.scatter(carprice[numCols.columns[i]],carprice['price'])
    # Assigning a title to the plot
    plt.title(numCols.columns[i]+' vs Price')
    # Setting the y label
    plt.ylabel('Price')
    # setting the x label
    plt.xlabel(numCols.columns[i])


# printing all the plots
plt.tight_layout()
#Binning the Car Names based on avg prices of each Car company:

carprice['price'] = carprice['price'].astype('int')
#creating a copy of the original df for manipulations
df = carprice.copy()
# grouping the CarName column w.r.t mean price
table = df.groupby(['CarName'])['price'].mean()
# merging the changes to the df dataframe
df = df.merge(table.reset_index(), how='left',on='CarName')
# creating the bins
bins = [0,10000,20000,46000]
# creating the labels for the bins
cars_bin=['LowEnd','MediumEnd','HighEnd']
# applying the changes to the original df using pd.cut function
carprice['carprice_category'] = pd.cut(df['price_y'],bins,right=False,labels=cars_bin)
# Printing the df
carprice.head()
# Finding the categorical columns and printing the same.
categCols = carprice.select_dtypes(exclude=['int64','float64'])
# Sorting the columns
categCols = categCols[sorted(categCols.columns)]
# printing the columns
print(categCols.columns)
# Initializing a figure
plt.figure(figsize=(15,50))

# Initializing a variable for plotting multiple sub plots
n=0

# running a for-loop to print the histogram and boxplots for all categorical columns
for i in range(len(categCols.columns)):
    # Increasing the count of the variable n
    n+=1
    # Creating a 1st sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Histogram as the 1st plot for the column
    sns.countplot(carprice[categCols.columns[i]])
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='CarName':
        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' Histogram')
    
    # Increasing the count of the variable n to plot the box plot for the same column
    n+=1
    
    # Creating a 2nd sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Boxplot as the 2nd plot for the column
    sns.boxplot(x=carprice[categCols.columns[i]], y=carprice.price)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' vs Price')
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='CarName':
        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
        

# printing all the plots
plt.tight_layout()
#Applying the calculation mentioned above and finding the average fuel economy of a car:
carprice['fuel_economy'] = (0.55 * carprice['citympg']) + (0.45 * carprice['highwaympg'])

# printing the df
carprice.head()
# Printing a scatter plot to check the fuel economy w.r.t price having the carprice category as the hue
plt1 = sns.scatterplot(x = 'fuel_economy', y = 'price' , hue = 'carprice_category', data = carprice)

# setting the x label
plt1.set_xlabel('Fuel Economy')

# setting the y label
plt1.set_ylabel('Car Price')

# printing the plot
plt.show()
# Printing a scatter plot to check the carlength w.r.t price having the carprice category as the hue
plt2 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'carprice_category', data = carprice)

# setting the x label
plt2.set_xlabel('Horsepower')

# setting the y label
plt2.set_ylabel('Car Price')

# printing the df
plt.show()
# Printing a scatter plot to check the carlength w.r.t price having the carprice category as the hue
plt3 = sns.scatterplot(x = 'carlength', y = 'price', hue = 'carprice_category', data = carprice)

# setting the x label
plt3.set_xlabel('carlength')

# setting the y label
plt3.set_ylabel('Car Price')

# printing the df
plt.show()
# Creating a new df
carprice_lr = carprice[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fuel_economy', 'carlength','carwidth', 'carprice_category','CarName']]
# Checking the new df
carprice_lr.head()
# This step will take time due to the size of the data
plt.figure(figsize=(15, 15))

# plotting the pairplot for the dataset
sns.pairplot(carprice_lr)

plt.show()
# Defining the map function
def dummies(x,df):
    # Get the dummy variables for the categorical feature and store it in a new variable - 'dummy'
    dummy = pd.get_dummies(df[x], drop_first = True)
    # Add the results to the original dataframe
    df = pd.concat([df, dummy], axis = 1)
    # Drop the original category variables as dummy are already created
    df.drop([x], axis = 1, inplace = True)
    # return the df
    return df

# Applying the function to the carprice_lr categorical columns
carprice_lr = dummies('fueltype',carprice_lr)
carprice_lr = dummies('aspiration',carprice_lr)
carprice_lr = dummies('carbody',carprice_lr)
carprice_lr = dummies('drivewheel',carprice_lr)
carprice_lr = dummies('enginetype',carprice_lr)
carprice_lr = dummies('cylindernumber',carprice_lr)
carprice_lr = dummies('carprice_category',carprice_lr)
carprice_lr = dummies('CarName',carprice_lr)
# print the df
carprice_lr.head()
# checking the columns of the final df
carprice_lr.columns
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
# splitting the df into train and test sets
df_train, df_test = train_test_split(carprice_lr, train_size = 0.7, test_size = 0.3, random_state = 100)
# Checking the number of columns and rows in the train dataset
df_train.shape
# Checking the number of columns and rows in the test dataset
df_test.shape
# Importing the MinMaxScaler from the sklearn.preprocessing library
from sklearn.preprocessing import MinMaxScaler
# Assigning the function to a variable
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'dummy' variables
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fuel_economy','carlength','carwidth','price']

# Applying the fit_transform() function to the df_train df
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
# checking the df
df_train.head()
# checking the df
df_train.describe()
plt.figure(figsize = (30, 30))

# command to plot the heat map
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

#Printing the plot
plt.show()
#Dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)
# RFE function will now determine the ranking of all the variables and rank them for our use.
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Here we are taking the top 10 columns which are recommended by the RFE function
col = X_train.columns[rfe.support_]
col
# Printing the columns which are being discarded from further analysis
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
# We will use the statsmodels library to build the models
import statsmodels.api as sm

# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model
lm = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_rfe.drop(["ohcf"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping the variable "subaru"
X_train_new = X_train_new.drop(["subaru"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["curbweight"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["wagon"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(["three"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_lm)
# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
res=y_train - y_train_price
res
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot(res, bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
# Selecting our variables again
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fuel_economy','carlength','carwidth','price']

# Applying the transform() function on the test set
df_test[num_vars] = scaler.transform(df_test[num_vars])

# checking the df shape
df_test.shape
# Checking more info about test df
df_test.describe()
#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test
# Now let's use our model to make predictions.

# Adding a constant variable using the X_test df
X_test_1 = sm.add_constant(X_test)

# Creating X_test_new dataframe by dropping variables from X_test_1
X_test_new = X_test_1[X_train_new.columns]
# Checking the X_test_new df
X_test_new
# Making predictions
y_pred = lm.predict(X_test_new)
# Calculating the RMSE Score
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   
print(lm.summary())