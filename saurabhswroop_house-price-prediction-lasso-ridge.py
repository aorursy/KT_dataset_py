import numpy as np
import pandas as pd
pd.set_option('max_columns', 200)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
# Importing train.csv
# Please make sure that the csv file is in the same folder as the python notebook otherwise this command wont work
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# Check the head of the dataset
df_train.head()
df_train.shape
# Prining all the columns of the dataframe
df_train.columns.values
print("{} Numerical columns, {} Categorial columns are part of the original dataset.".format(
    list(df_train.select_dtypes(include=[np.number]).shape)[1],
    list(df_train.select_dtypes(include = ['object']).shape)[1]))
df_train.info()
df_train.describe()
#Checking for any duplicates in the data frame
df_train.loc[df_train.duplicated()]
df_train = df_train.drop('Id',axis=1)
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(20)
# setting a grid for the pot
sns.set_style("whitegrid")
# finding the no of missing values 
missing = df_train.isnull().sum()
# filtering the columns with just missing values
missing = missing[missing > 0]
# sorting the values
missing.sort_values(inplace=True)
# plotting the bar chart
missing.plot.bar()
# setting the title of the plot
plt.title('Columns with Missing values', fontsize=15)
# setting the x label
plt.xlabel('Columns')
# setting the y label
plt.ylabel('No of missing values')


# removing any column which has more than 90% null values
df_train = df_train.loc[:,df_train.isnull().sum()/df_train.shape[0]*100<80]
# printing the df
print(df_train.shape)
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(16)
NA=df_train[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt','BsmtFinType2',
'BsmtFinType1','BsmtCond', 'BsmtQual','BsmtExposure', 'MasVnrArea','MasVnrType','Electrical','FireplaceQu',
             'LotFrontage']]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print('We have :',NAcat.shape[1],'categorical features with missing values')
print('We have :',NAnum.shape[1],'numerical features with missing values')
# columns where NA values have meaning e.g. no garage etc.
cols_fillna = ['MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']

# replace 'NA' with 'No' in these columns
for col in cols_fillna:
    df_train[col].fillna('No',inplace=True)
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(5)
# checking the count of different values within the column
df_train['LotFrontage'].value_counts()
# pulling the length of number of unique values in the column
num_unique_values = len(df_train['LotFrontage'].unique())
# Plotting a histogram for visualizing the data
df_train['LotFrontage'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is ",df_train['LotFrontage'].mean())
# checking the mode of the column
print("Mode is ",df_train['LotFrontage'].mode())
# checking the median of the column
print("Median is ",df_train['LotFrontage'].median())
# imputing the value of median to the null values
df_train.loc[pd.isnull(df_train['LotFrontage']),['LotFrontage']]=69
# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['LotFrontage'].unique())
# Plotting a histogram for visualizing the data
df_train['LotFrontage'].plot.hist(bins = num_unique_values)
# checking the count of different values within the column
df_train['GarageYrBlt'].value_counts()
# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['GarageYrBlt'].unique())
# Plotting a histogram for visualizing the data
df_train['GarageYrBlt'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is ",df_train['GarageYrBlt'].mean())
# checking the mode of the column
print("Mode is ",df_train['GarageYrBlt'].mode())
# checking the median of the column
print("Median is ",df_train['GarageYrBlt'].median())
# imputing the value of mode to the null values
df_train.loc[pd.isnull(df_train['GarageYrBlt']),['GarageYrBlt']]=1980
# pulling the length of number of unique values in the column
num_unique_values =  len(df_train['GarageYrBlt'].unique())
# Plotting a histogram for visualizing the data
df_train['GarageYrBlt'].plot.hist(bins = num_unique_values)
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data.head(5)
# checking the count of different values within the column
df_train['MasVnrArea'].value_counts()
# checking the mean of the column
print("Mean is ",df_train['MasVnrArea'].mean())
# checking the mode of the column
print("Mode is ",df_train['MasVnrArea'].mode())
# checking the median of the column
print("Median is ",df_train['MasVnrArea'].median())
# imputing the value of median to the null values
df_train.loc[pd.isnull(df_train['MasVnrArea']),['MasVnrArea']]=0
# checking the count of different values within the column
df_train['Electrical'].value_counts()
df_train['Electrical'] = df_train['Electrical'].fillna("SBrkr")
# finding all the missing data and summing them based on each column and storing it in a dataframe
total = df_train.isnull().sum().sort_values(ascending = False)
# Finding the percentage of the missing data by diving the number of missing values with total and  storing it in a dataframe
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
# Concatinating both the above df's
df_train_missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Printing the data
df_train_missing_data[df_train_missing_data.sum(axis=1)>0]
# before we move forward, lets create a copy of the existing df
df_train1=df_train.copy()
# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
# df_train = df_train.drop(['SalePrice'],axis=1)

sns.boxplot(data=df_train1)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()
df_train1.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])
# Finding the columns on which the outlier treatment will be performed
AllCols = df_train1.select_dtypes(exclude='object')
# Sorting the columns
AllCols = AllCols[sorted(AllCols.columns)]
# printing the columns
print(AllCols.columns)
# running a for loop to remove the outliers from each column
for i in AllCols.columns:
    # setting the lower whisker
    Q1 = df_train[i].quantile(0.05)
    # setting the upper whisker
    Q3 = df_train[i].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
    IQR = Q3 - Q1
    # performing the outlier analysis
    df_train = df_train[(df_train[i] >= Q1-1.5*IQR) & (df_train[i] <= Q3+1.5*IQR)]
# Checking the shape of the df now
df_train.shape
# checking the different percentiles now
df_train.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])
# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
# df_train1 = df_train.drop(['SalePrice','LotArea'],axis=1)

sns.boxplot(data=df_train)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()
# Let's look at the scarifice
print("Shape before outlier treatment: ",df_train1.shape)
print("Shape after outlier treatment: ",df_train.shape)

print("Percentage data removal is around {}%".format(round(100*(df_train1.shape[0]-df_train.shape[0])/df_train1.shape[0]),2))
# Initializing a figure
plt.figure(figsize=(20,8))

# Initializing a subplot
plt.subplot(1,2,1)
# setting the title of the plot
plt.title('House Price Histogram')
# Plotting a Histogram for price column
sns.distplot(df_train.SalePrice, kde=False, fit=stats.lognorm)
plt.ylabel('Frequency')

# Initializing another subplot
plt.subplot(1,2,2)
# setting the title of the plot
plt.title('House Price Box Plot')
# Plotting a boxplot for price column
sns.boxplot(y=df_train.SalePrice)

plt.show()
# Checking the various percentile values for the price column
print(df_train.SalePrice.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
#skewness
print("Skewness: " + str(df_train['SalePrice'].skew()))
# Finding all the numerical columns in the dataset. 
numCols = df_train.select_dtypes(include=['int64','float'])

# Sorting the columns
numCols = numCols[sorted(numCols.columns)]

# Printing the columns
print(numCols.columns)
print("Numerical features : " + str(len(numCols.columns)))

# Initializing a figure
plt.figure(figsize=(30,200))

# Dropping the price column from the plot since we dont need to plot a scatter plot for price
numCols = numCols.drop('SalePrice',axis=1)

# running a for-loop to print the scatter plots for all numerical columns
for i in range(len(numCols.columns)):
    # Creating a sub plot
    plt.subplot(len(numCols.columns),2,i+1)
    # Creating a scatter plot
    plt.scatter(df_train[numCols.columns[i]],df_train['SalePrice'])
    # Assigning a title to the plot
    plt.title(numCols.columns[i]+' vs Price')
    # Setting the y label
    plt.ylabel('Price')
    # setting the x label
    plt.xlabel(numCols.columns[i])


# printing all the plots
plt.tight_layout()
# Finding the categorical columns and printing the same.
categCols = df_train.select_dtypes(exclude=['int64','float64'])
# Sorting the columns
categCols = categCols[sorted(categCols.columns)]
# printing the columns
print(categCols.columns)
# Initializing a figure
plt.figure(figsize=(15,100))

# Initializing a variable for plotting multiple sub plots
n=0

# running a for-loop to print the histogram and boxplots for all categorical columns
for i in range(len(categCols.columns)):
    # Increasing the count of the variable n
    n+=1
    # Creating a 1st sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Histogram as the 1st plot for the column
    sns.countplot(df_train[categCols.columns[i]])
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='Exterior1st' or categCols.columns[i]=='Exterior2nd' or categCols.columns[i]=='Neighborhood':        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' Histogram')
    
    # Increasing the count of the variable n to plot the box plot for the same column
    n+=1
    
    # Creating a 2nd sub plot
    plt.subplot(len(categCols.columns),2,n)
    # Creating a Boxplot as the 2nd plot for the column
    sns.boxplot(x=df_train[categCols.columns[i]], y=df_train1.SalePrice)
    # Assigning a title to the plot
    plt.title(categCols.columns[i]+' vs Price')
    # assigning x label rotation for carName column for proper visibility
    if categCols.columns[i]=='Exterior1st' or categCols.columns[i]=='Exterior2nd' or categCols.columns[i]=='Neighborhood':
        plt.xticks(rotation=75)
    else:
        plt.xticks(rotation=0)
        

# printing all the plots
plt.tight_layout()
# pulling all the columns which can be deleted based on skewness
cols_to_drop = ['Utilities','3SsnPorch','LowQualFinSF','MiscVal','PoolArea','ScreenPorch','KitchenAbvGr','GarageQual'
               ,'GarageCond','Functional','Heating','LandContour','LandSlope','LotConfig','MSZoning','PavedDrive',
                'RoofMatl','RoofStyle','SaleCondition','SaleType','Street']

# running the for loop to print the value counts
for i in cols_to_drop:
    print(df_train[i].value_counts(normalize=True) * 100)
# dropping the columns 
df_train = df_train.drop(['Utilities','3SsnPorch','LowQualFinSF','MiscVal','PoolArea','ScreenPorch','KitchenAbvGr','GarageQual'
               ,'GarageCond','Functional','Heating','LandContour','MSZoning','PavedDrive',
                'RoofStyle','SaleCondition','SaleType','Street','BedroomAbvGr','MoSold'],axis=1)
# checking the shape of the df now
df_train.shape
# saleprice correlation matrix
corr_num = 15 #number of variables for heatmap
corrmat = df_train.corr()
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(df_train[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()
# Initializing a figure
plt.figure(figsize=(20,8))

# Initializing a subplot
plt.subplot(1,2,1)
# setting the title of the plot
plt.title('House Price Histogram')
# Plotting a Histogram for price column
sns.distplot(df_train.SalePrice, kde=False, fit=stats.lognorm)
plt.ylabel('Frequency')
# Checking if the log transformation normalizes the target variable
sns.distplot(np.log(df_train["SalePrice"]))
# Applying the log transformation to the target variable
df_train["SalePrice"] = np.log(df_train["SalePrice"])
# importing the skew library to check the skewness
from scipy.stats import skew  
# pulling the numeric columns from the dataset 
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats
# Lets combine the floors square feet and the basement square feet to create the total sq feet
df_train['Total_sq_feet'] = (df_train['BsmtFinSF1'] + df_train['BsmtFinSF2'] +
                                     df_train['1stFlrSF'] + df_train['2ndFlrSF'])

# Lets combine all the bathrooms square feet to create the total bathroom sq feet
df_train['Total_Bathrooms_sq_feet'] = (df_train['FullBath'] + (0.5 * df_train['HalfBath']) +
                                   df_train['BsmtFullBath'] + (0.5 * df_train['BsmtHalfBath']))

# Lets combine all the porch square feet to create the total porch sq feet
df_train['Total_porch_sq_feet'] = (df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + df_train['WoodDeckSF'])
# checking the shape of the df now
df_train.shape
# lets drop the columns which we used to create new features
df_train= df_train.drop(['BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath',
                           'BsmtFullBath','BsmtHalfBath','OpenPorchSF','EnclosedPorch','WoodDeckSF'],axis=1)
# checking the df now
df_train.head()
# checking the shape now
df_train.shape
# pulling the list of all the year columns from the dataset
Year_cols = df_train.filter(regex='Yr|Year').columns
# running a for loop to find the max year of each column 
for i in Year_cols:
    i = df_train[i].max()
    print(i)
# running a for loop to subtract the max year with all values
for i in Year_cols:
    df_train[i] = df_train[i].apply(lambda x: 2010 - x)
# Checking the dataset now
df_train.head()
# pulling all the categorical columns from the dataset.
categCols = df_train.select_dtypes(exclude=['int64','float64'])
# Sorting the columns
categCols = categCols[sorted(categCols.columns)]
# printing the categorical columns
print(categCols.columns)
# Defining the map function
def dummies(x,df):
    # Get the dummy variables for the categorical feature and store it in a new variable - 'dummy'
    dummy = pd.get_dummies(df[x], drop_first = True)
    for i in dummy.columns:
        dummy = dummy.rename(columns={i: x+"_"+i})
    # Add the results to the original dataframe
    df = pd.concat([df, dummy], axis = 1)
    # Drop the original category variables as dummy are already created
    df.drop([x], axis = 1, inplace = True)
    # return the df
    return df

#Applying the function to the df_train categorical columns
for i in categCols:
    df_train = dummies(i,df_train)
# checking the dataset now
df_train.head()
# Checking the shape of the new dataset which will be used for model building
df_train.shape
# importing the libraries
from sklearn.preprocessing import StandardScaler
# dropping the target variable and stroing the remaining in a new df
X = df_train.drop(['SalePrice'],axis=1)
# storing the target column in a new df
y = df_train['SalePrice']
# initializing the standard scalar
scaler = StandardScaler()
# scale the X df
scaler.fit(X)
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.loc[vif['VIF'] > 3000, :]
# Dropping the above columns
X=X.drop(['Exterior2nd_CBlock','BsmtUnfSF','BsmtQual_No','Total_sq_feet','Exterior1st_CBlock','BsmtCond_No'
         ,'TotalBsmtSF','GrLivArea','GarageFinish_No','GarageType_No','BsmtFinType1_No'],axis=1)
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
# X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.loc[vif['VIF'] > 100, :]
X=X.drop(['ExterCond_TA','Condition2_Norm','Exterior1st_VinylSd','Exterior2nd_VinylSd','Exterior1st_MetalSd','Exterior2nd_MetalSd'
         ,'Exterior1st_HdBoard'],axis=1)
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
# X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# importing the required libraries
from sklearn.model_selection import train_test_split

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)
# Checking the number of columns and rows in the train dataset
X_train.shape
# Checking the number of columns and rows in the test dataset
X_test.shape
# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)

# print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = lm.predict(X_test)
# print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))
# model coefficients
# liner regression model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

# Initializing the Ridge regression
ridge = Ridge()

# cross validation
# Setting the number of folds
folds = 5
# performing GridSearchCV on the ridge regression using the list of params
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
# Fitting the model on our Train sets
model_cv.fit(X_train, y_train) 
# Storing the results in a new df
cv_results = pd.DataFrame(model_cv.cv_results_)
# filtering out the alpha parameters which are less than 200
cv_results = cv_results[cv_results['param_alpha']<=200]
# checking the results
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()
# finding the best Alpha value
print ('The best value of Alpha for Ridge Regression is: ',model_cv.best_params_)
# setting the value of alpha as 7
alpha = 7
# initializing the ridge regression with the optimized alpha value
ridge = Ridge(alpha=alpha)

# running the ridge algo on the train datasets
ridge.fit(X_train, y_train)

# Lets predict
y_train_pred = ridge.predict(X_train)
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = ridge.predict(X_test)
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))

# checking the coefficient values of all the features.
ridge.coef_
# Assigning the columns to the respective coefficient values
# ridge model parameters
model_parameters = list(ridge.coef_)
model_parameters.insert(0, ridge.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))
# pulling the coefficients and index and creating a new df
coef = pd.Series(ridge.coef_, index = X.columns).sort_values()
# filtering the top 5 positive and negative features 
ridge_imp_coef = pd.concat([coef.head(10), coef.tail(10)])
# plotting the graph
ridge_imp_coef.plot(kind = "barh")
# setting the title of the plot
plt.title("Model Coefficients")
# Converting the important feature list into a df for better understanding
ridge_imp_coef = ridge_imp_coef.to_frame('Coeff_val').reset_index()
ridge_imp_coef.columns = ['Features', 'Coeff_val']
ridge_imp_coef['Coeff_val'] = ridge_imp_coef['Coeff_val'].abs()
ridge_imp_coef = ridge_imp_coef.sort_values(by=['Coeff_val'], ascending=False)
ridge_imp_coef.head(10)
p_pred = np.expm1(ridge.predict(X))
plt.scatter(p_pred, np.expm1(y))
plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="red")
# initializing the Lasso regression
lasso = Lasso()

# cross validation
# performing GridSearchCV on the lasso regression using the list of params
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
# Fitting the model on our Train sets
model_cv.fit(X_train, y_train) 
# Storing the results in a new df
cv_results = pd.DataFrame(model_cv.cv_results_)

# checking the results
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
# plotting the mean train scores
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
# plotting the mean test scores
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
# setting the x label
plt.xlabel('alpha')
# setting the y label
plt.ylabel('Negative Mean Absolute Error')
# setting the xscale into log
plt.xscale('log')
# setting the title
plt.title("Negative Mean Absolute Error and alpha")
# setting the legend
plt.legend(['train score', 'test score'], loc='upper left')
# showing the plot
plt.show()
print('The best value of Alpha for Lasso Regression is: ',model_cv.best_params_)
# initializing the ridge regression with the optimized alpha value
lm = Lasso(alpha=0.001)
# fitting the model on the train datasets
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print("RMSE Train {}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("R2 Score Train {}".format(r2_score(y_train, y_train_pred)))
y_test_pred = lm.predict(X_test)
print("RMSE Test {}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("R2 Score Test {}".format(r2_score(y_test, y_test_pred)))
# checking the coefficient values of all the features.
lm.coef_
# Assigning the columns to the respective coefficient values
# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))
# pulling the coefficients and index and creating a new df
coef = pd.Series(lm.coef_, index = X.columns).sort_values()
# filtering the top 5 positive and negative features 
lasso_imp_coef = pd.concat([coef.head(10), coef.tail(10)])
# plotting the graph
lasso_imp_coef.plot(kind = "barh")
# setting the title of the plot
plt.title("Model Coefficients")
# Converting the important feature list into a df for better understanding
lasso_imp_coef = lasso_imp_coef.to_frame('Coeff_val').reset_index()
lasso_imp_coef.columns = ['Features', 'Coeff_val']
lasso_imp_coef['Coeff_val'] = lasso_imp_coef['Coeff_val'].abs()
lasso_imp_coef = lasso_imp_coef.sort_values(by=['Coeff_val'], ascending=False)
lasso_imp_coef.head(10)
p_pred = np.expm1(lm.predict(X))
plt.scatter(p_pred, np.expm1(y))
plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="red")
# checking how many features were dropped by lasso during modelling
print("Lasso kept",sum(coef != 0), "important features and dropped the other", sum(coef == 0),"features")