# to handle datasets

import pandas as pd

import numpy as np



# for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# for tree binarisation

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score





# to build the models

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb



# to evaluate the models

from sklearn.metrics import mean_squared_error

from math import sqrt



pd.pandas.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# Read in the Dataset

data = pd.read_csv("../input/train.csv")


# Looking at the top 5 Rows

data[:5]


data.shape
data.info()
data.dtypes
print('Number of House Id labels:', len(data.Id.unique()))

print('Number of Housees in Dataset:', len(data))
categorical = [var for var in data.columns if data[var].dtypes == 'O']

print('There are {} categorical features'.format(len(categorical)))
data[categorical].head()


# Get the numerical

numerical = [var for var in data.columns if data[var].dtypes != 'O']
data[numerical].head()


# We are extracting the year features

years = [var for var in numerical if 'Yr' in var or 'Year' in var]

years
data[years].head()


# Looking at the month the house was sold

data.groupby('MoSold')['SalePrice'].median().plot()

plt.title('House price variation')

plt.ylabel('mean House price')


# Looking at the year the house was sold

data.groupby('YrSold')['SalePrice'].median().plot()

plt.title('House price variation')

plt.ylabel('mean House price')


discrete = []

for var in numerical:

    if len(data[var].unique()) < 20 and var not in years:

        print(var, 'values:', data[var].unique())

        discrete.append(var)

print()

print('There are {} discrete features'. format(len(discrete)))


# Making a list of all the numerical features



numerical = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice'] and var not in years]

print('There are {} continuos features', format(len(numerical)))
for var in data.columns:

    if data[var].isnull().sum() > 0:

        print(var, data[var].isnull().mean())
# Checking for missing values and calculating the percentages

# We are using a list comprehension to extract all the variables with NA's

vars_with_na = [var for var in data.columns if data[var].isnull().sum()>0]

print('Total features that contains missing:', len(vars_with_na))
# Inspecting features with a lot of missing

for var in data.columns:

    if data[var].isnull().mean() > 0.80:

        print(var, data[var].unique())
# Let us analyse the relationship between missing data and the Target variable (SalePrice)



def analyse_na_value(df, var):

    df = df.copy()

    df[var] = np.where(df[var].isnull(), 1, 0)

    df.groupby(var)['SalePrice'].median().plot.bar()

    plt.title(var)

    plt.show()

    

for var in vars_with_na:

    analyse_na_value(data, var)


# let's make boxplots to visualise outliers in the continuous variables 

# and histograms to get an idea of the distribution



for var in numerical:

    plt.figure(figsize=(15, 6))

    plt.subplot(1,2,1)

    fig = data.boxplot(column=var)

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1,2,2)

    fig = data[var].hist(bins = 20)

    fig.set_ylabel('Number of houses')

    fig.set_xlabel(var)

    plt.show()


for var in discrete:

    (data.groupby(var)[var].count()/np.float(len(data))).plot.bar()

    plt.ylabel('Percentage of observations')

    plt.title(var)

    plt.show()

    print()


num_labels_ls = []

for var in categorical:

    num_labels_ls.append(len(data[var].unique()))

    

tmp = pd.Series(num_labels_ls)

tmp.index = pd.Series(categorical)

tmp.plot.bar(figsize = (15,8))

plt.title('Number of labels in cat features')

plt.xlabel('Categorical Features')

plt.ylabel('Number of different features')
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.1,

                                                    random_state=0)

X_train.shape, X_test.shape
# function to calculate elapsed time



def time_elapsed(df, var):

    # capture difference between year variable and year the house was sold

    df[var] = df['YrSold'] - df[var]

    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    X_train = time_elapsed(X_train, var)

    X_test = time_elapsed(X_test, var)

  
X_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
# drop YrSold

X_train.drop('YrSold', axis=1, inplace=True)

X_test.drop('YrSold', axis=1, inplace=True)
# remove YrSold because it is no longer in our dataset

years.remove('YrSold')



# examine percentage of missing values

for col in numerical+years:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add median imputation

for df in [X_train, X_test]:

    for var in ['LotFrontage', 'GarageYrBlt']:

        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

        df[var].fillna(X_train[var].median(), inplace=True) 



for df in [X_train, X_test]:

    df['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace=True)
# print variables with missing data

for col in discrete:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# print variables with missing data

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add label indicating 'Missing' to categorical variables



for df in [X_train, X_test]:

    for var in categorical:

        df[var].fillna('Missing', inplace=True)
# check absence of null values

for var in X_train.columns:

    if X_train[var].isnull().sum()>0:

        print(var, X_train[var].isnull().sum())
# check absence of null values

for var in X_train.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
def tree_binariser(var):

    score_ls = [] # here I will store the mse



    for tree_depth in [1,2,3,4]:

        # call the model

        tree_model = DecisionTreeRegressor(max_depth=tree_depth)



        # train the model using 3 fold cross validation

        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')

        score_ls.append(np.mean(scores))



    # find depth with smallest mse

    depth = [1,2,3,4][np.argmin(score_ls)]

    #print(score_ls, np.argmin(score_ls), depth)



    # transform the variable using the tree

    tree_model = DecisionTreeRegressor(max_depth=depth)

    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)

    X_train[var] = tree_model.predict(X_train[var].to_frame())

    X_test[var] = tree_model.predict(X_test[var].to_frame())

for var in numerical:

    tree_binariser(var)
X_train[numerical].head()
# let's explore how many different buckets we have now among our engineered continuous variables

for var in numerical:

    print(var, len(X_train[var].unique()))
for var in numerical:

    X_train.groupby(var)['SalePrice'].mean().plot.bar()

    plt.title(var)

    plt.ylabel('Mean House Price')

    plt.xlabel('Discretised continuous variable')

    plt.show()
def rare_imputation(variable):

    # find frequent labels / discrete numbers

    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))

    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]

    

    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')

    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
# find infrequent labels in categorical variables and replace by Rare

for var in categorical:

    rare_imputation(var)

    

# find infrequent labels in categorical variables and replace by Rare

# remember that we are treating discrete variables as if they were categorical

for var in discrete:

    rare_imputation(var)
# let's check that it worked

for var in categorical:

    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
# let's check that it worked

for var in discrete:

    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
def encode_categorical_variables(var, target):

        # make label to price dictionary

        ordered_labels = X_train.groupby([var])[target].mean().to_dict()

        

        # encode variables

        X_train[var] = X_train[var].map(ordered_labels)

        X_test[var] = X_test[var].map(ordered_labels)

       

# encode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'SalePrice')

    

# encode labels in discrete vars

for var in discrete:

    encode_categorical_variables(var, 'SalePrice')
X_train.isnull().any()
#let's inspect the dataset

X_train.head()
X_train.describe()
# let's create a list of the training variables

training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]



print('total number of variables to use for training: ', len(training_vars))
training_vars
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set for later use
ln = LinearRegression(fit_intercept=True)

ln.fit(scaler.transform(X_train[training_vars]), np.log(y_train))
# Print intercept and slope



print('Coeff (M):', ln.coef_)

print('Coeff (b):', ln.intercept_)
# Make predictions



pred = ln.predict(scaler.transform(X_test[training_vars]))
# Plot and evaluate the model

plt.scatter(y_test, pred, color = 'red')

plt.ylabel('Predictions')

plt.xlabel('Actuals')

plt.title('Actual vs Predicted')
pred = ln.predict(scaler.transform(X_train[training_vars]))

print('Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = ln.predict(scaler.transform(X_test[training_vars]))

print('Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
# Train the model

lin_model = Lasso(random_state=2909, alpha=0.005)

lin_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))
# Make predictions

pred_1 = lin_model.predict(scaler.transform(X_train[training_vars]))

print('Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred_1))))

print('Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred_1)))))



print()

pred_2 = lin_model.predict(scaler.transform(X_test[training_vars]))

print('Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred_2))))

print('Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred_2)))))
# Visualise test sets

plt.scatter(y_test, pred_2, color = 'red')

plt.ylabel('Predictions')

plt.xlabel('Actuals')

plt.title('Actual vs Predicted')
plt.scatter(y_train, pred_1, color = 'red')

plt.ylabel('Predictions')

plt.xlabel('Actuals')

plt.title('Actual vs Predicted')