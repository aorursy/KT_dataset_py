from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file

# CarPrice_Assignment.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

cars = pd.read_csv('/kaggle/input/CarPrice_Assignment.csv', delimiter=',', nrows = nRowsRead)

cars.dataframeName = 'CarPrice_Assignment.csv'

nRow, nCol = cars.shape

print(f'There are {nRow} rows and {nCol} columns')
# Let's take a look at the first few rows

cars.head()
# Summary of the dataset: 205 rows, 26 columns, no null values

print(cars.info())
# Symboling: -2 (least risky) to +3 most risky

# Most cars are 0,1,2

cars['symboling'].astype('category').value_counts()
# All numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float64', 'int64'])

cars_numeric.head()
# Dropping car_ID as it is of no use

cars_numeric = cars_numeric.drop(['car_ID'], axis=1)

cars_numeric.head()
# Pairwise scatter plot

plt.figure(figsize=(20, 10))

sns.pairplot(cars_numeric)

plt.show()
# Correlation matrix

cor = cars_numeric.corr()

cor
# Figure size

plt.figure(figsize=(16,8))



# Heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
# Checking the Variable formats

cars.info()
#Removing car_ID column as it is irrelavent

cars.drop('car_ID', axis=1, inplace=True)
#Taking out the manufacturer's name

cars['CarName'] = cars['CarName'].apply(lambda x: x.split()[0])

cars['CarName'].value_counts().index.sort_values()
# In the data there is different spellings for same Manufacturer, hence sorting the data accordingly 

cars['CarName'] = cars['CarName'].replace('porcshce','porsche')

cars['CarName'] = cars['CarName'].replace('vokswagen','volkswagen')

cars['CarName'] = cars['CarName'].replace('vw','volkswagen')

cars['CarName'] = cars['CarName'].replace('toyouta','toyota')

cars['CarName'] = cars['CarName'].replace('maxda','mazda')

cars['CarName'] = cars['CarName'].replace('Nissan','nissan')
#Renaming the car name to company name

cars = cars.rename(columns={'CarName':'Manufacturer'})
#Checking the car names again

cars['Manufacturer'].value_counts()
# Let's check for any outliers

cars.describe()
# Let's check for any outliers

cars.describe()
cars.head()
# Checking the different levels of 'cylindernumber'

cars['cylindernumber'].astype('category').value_counts()
# Checking the different levels of 'doornumber'

cars['doornumber'].astype('category').value_counts()
# A function to map the categorical levels to actual numbers. You can see the categorical levels above and use them for mapping.

def num_map(x):

    return x.map({'two': 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12})



# Applying the function to the two columns

cars[['cylindernumber', 'doornumber']] = cars[['cylindernumber', 'doornumber']].apply(num_map)
# Subset all categorical variables

cars_categorical = cars.select_dtypes(include=['object'])

cars_categorical.head()
# Convert into dummies

cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)

cars_dummies.head()
# Drop categorical variable columns

cars = cars.drop(list(cars_categorical.columns), axis=1)
# Concatenate dummy variables with X

cars = pd.concat([cars, cars_dummies], axis=1)
# Let's check the first few rows

cars.head()
# Split the datafram into train and test sets

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(cars, test_size=0.3, random_state=100)

print(df_train.shape)

print(df_test.shape)
cars_numeric.columns
# Import the StandardScaler()

from sklearn.preprocessing import StandardScaler



# Create a scaling object

scaler = StandardScaler()



# Create a list of the variables that you need to scale

varlist = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight', 'enginesize', 'boreratio', 'stroke', 

           'compressionratio','horsepower', 'peakrpm', 'citympg', 'highwaympg', 'doornumber', 'cylindernumber', 'price']



# Scale these variables using 'fit_transform'

df_train[varlist] = scaler.fit_transform(df_train[varlist])
# Let's take a look at the train dataframe now

df_train.head()
# Splitting the train dataset into X and y

y_train = df_train.pop('price')

X_train = df_train
# Instantiate

lm = LinearRegression()

# Fit a line

lm.fit(X_train, y_train)
# Print the coefficients and intercept

print(lm.coef_)

print(lm.intercept_)
# Import RFE

from sklearn.feature_selection import RFE



# RFE with 15 features

lm = LinearRegression()

rfe1 = RFE(lm, 15)



# Fit with 15 features

rfe1.fit(X_train, y_train)



# Print the boolean results

print(rfe1.support_)           

print(rfe1.ranking_) 
# Import statsmodels

import statsmodels.api as sm  



# Subset the features selected by rfe1

col1 = X_train.columns[rfe1.support_]



# Subsetting training data for 15 selected columns

X_train_rfe1 = X_train[col1]



# Add a constant to the model

X_train_rfe1 = sm.add_constant(X_train_rfe1)

X_train_rfe1.head()
# Fitting the model with 15 variables

lm1 = sm.OLS(y_train, X_train_rfe1).fit()   

print(lm1.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe1.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe1.values, i) for i in range(X_train_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# RFE with 10 features

lm = LinearRegression()

rfe2 = RFE(lm, 10)



# Fit with 10 features

rfe2.fit(X_train, y_train)
# Subset the features selected by rfe2

col2 = X_train.columns[rfe2.support_]



# Subsetting training data for 10 selected columns

X_train_rfe2 = X_train[col2]



# Add a constant to the model

X_train_rfe2 = sm.add_constant(X_train_rfe2)



# Fitting the model with 10 variables

lm2 = sm.OLS(y_train, X_train_rfe2).fit()   

print(lm2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_rfe2.drop('Manufacturer_subaru', axis = 1, inplace = True)
# Refitting with 9 variables

X_train_rfe2 = sm.add_constant(X_train_rfe2)

# Fitting the model with 9 variables

lm2 = sm.OLS(y_train, X_train_rfe2).fit()   

print(lm2.summary())
vif = pd.DataFrame()

vif['Features'] = X_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_rfe2.drop('Manufacturer_peugeot', axis = 1, inplace = True)
# Refitting with 8 variables

X_train_rfe2 = sm.add_constant(X_train_rfe2)



# Fitting the model with 8 variables

lm2 = sm.OLS(y_train, X_train_rfe2).fit()   

print(lm2.summary())
vif = pd.DataFrame()

vif['Features'] = X_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm2.predict(X_train_rfe2)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test[varlist] = scaler.transform(df_test[varlist])
# Split the 'df_test' set into X and y

y_test = df_test.pop('price')

X_test = df_test
col2
# Let's subset these columns and create a new dataframe 'X_test_rfe2'

X_test_rfe2 = X_test[col2]
# Let's now drop the variables we had manually eliminated as well

X_test_rfe2 = X_test_rfe2.drop(['Manufacturer_subaru', 'Manufacturer_peugeot'], axis = 1)
# Add a constant to the test set created

X_test_rfe2 = sm.add_constant(X_test_rfe2)

X_test_rfe2.info()
# Making predictions

y_pred = lm2.predict(X_test_rfe2)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# r2_score for 6 variables

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
print(lm2.summary())