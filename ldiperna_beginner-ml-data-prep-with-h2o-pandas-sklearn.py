# required imports

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline 

import pandas as pd



# launch h2o cluster

h2o.init()



# import the train and test datasets with H2O

training = h2o.import_file("../input/train.csv")

test = h2o.import_file("../input/test.csv")



# set the predictors (remove the response and ID column)

predictors = training.columns[1:-1]

# set the response column

response = 'SalePrice'



###############################

# repeat the steps for pandas #

###############################



# import the train and test datasets with pandas, to illustrate encoding methods

# which H2O would handle automatically

pd_training = pd.read_csv("../input/train.csv")

pd_test = pd.read_csv("../input/test.csv")



# set the predictors for pandas(remove the response and ID column)

pd_predictors = training.columns[1:-1]

# set the response column

pd_response = 'SalePrice'
# pandas interpreted 'NA' in the categorical columns as missing, let's revert those missing values back to 

# the string 'NA' values since NA means "No Alley" or "No Garage" for example



# print the missing value count before making the replacement

print('missing in pandas train originally:', pd_training.isnull().sum().sum())

print('missing in pandas test originally:', pd_test.isnull().sum().sum())

print("")



# code below explained:

# `[pd_training.dtypes == "object"]` finds all the columns that have type object

# `[pd_training.dtypes == "object"].index` returns all the index names (column names) that have type object

# `pd_training[pd_training.dtypes[pd_training.dtypes == "object"].index]` selects all the categorical columns

# `.fillna('NA')` replaces the missing values with the string 'NA'



# replace missing values with 'NA' in the training set:

pd_training[pd_training.dtypes[pd_training.dtypes == "object"].index] = pd_training[pd_training.dtypes[pd_training.dtypes == "object"].index].fillna('NA')



# replace missing values with 'NA' in the test set:

pd_test[pd_test.dtypes[pd_test.dtypes == "object"].index] = pd_test[pd_test.dtypes[pd_test.dtypes == "object"].index].fillna('NA')



# comparing number of missing value in pandas dataframe to H2O dataframe

# H2O interprets missing values in categorical columns as their own level, because

# H2O assumes these values are 'missing for a reason' (which in this case applies 'NA' can mean 'No Alley' or 'No Basement')

print('missing in H2O train originally:', training.isna().sum())

print('missing in H2O test originally:', test.isna().sum())

print("")



print('missing in pandas train after munging:', pd_training.isnull().sum().sum())

print('missing in pandas test after munging:', pd_test.isnull().sum().sum())

print("")



# is there any missing data in the response column? ans: No

print("missing values in H2O train response:", training['SalePrice'].isna().sum())

print('missing in pandas train:', pd_training['SalePrice'].isnull().sum())
# get the list of categorical columns that now have 'NA' values



# we find that there are only missing values in there numeric columns

print("Categorical Column Names with Missing Value Count")

categorical_list = list(pd_training.dtypes[pd_training.dtypes == "object"].index)

print("")

mis_cat = []

# list which columns have missing values

for column in categorical_list:

    if (pd_training[column] == 'NA').any():

        mis_cat.append(column )

        print(column + ':' + str(pd_training[pd_training[column] == 'NA'].shape[0]))

print("")

print('mis_cat:', mis_cat)
# get the list of categorical columns that still have missing values

print("Categorical Column that Still Have Missing Values")



# we find that there are only missing values in there numeric columns

categorical_list = list(pd_training.dtypes[pd_training.dtypes == "object"].index)

print("")

mis_cat = []

# list which columns have missing values

for column in categorical_list:

    if (pd_training[column].isnull().sum() >0).all():

        mis_cat.append(column )

        print(column + ':' + str(pd_training[column].isnull().sum()))

print("")

print('mis_cat:', mis_cat)
# get the list of numeric columns that still have missing values

print("Numerical Column Names with Missing Value Count")

print("")

num_list = list(pd_training.columns.difference(categorical_list))

mis_col = []

# list which columns have missing values

for column in num_list:

    if (pd_training[column].isnull().sum() >0).all():

        mis_col.append(column )

        print(column + ':' + str(pd_training[column].isnull().sum()))

print("")

print('mis_col:', mis_col)
# let's plot some box plots for a quick overview of the numeric columns with missing values

fig = plt.figure(figsize=(15,5))

fig.add_subplot(131)

pd_training[['GarageYrBlt']].boxplot(return_type='axes')

plt.ylim((1850,2020))



fig.add_subplot(132)

pd_training[['MasVnrArea']].boxplot(return_type='axes')

plt.ylim((-100,1650))



fig.add_subplot(133)

pd_training[['LotFrontage']].boxplot(return_type='axes')

plt.ylim((20,140))

plt.show()
# let's explore the MasVnrArea feature and its corresponding MasVnrType feature

# except for two examples if MasVnrType = None, MasVnrArea = 0

# if MasVnrArea is missing then MasVnrType is missing as well

print ('MasVnrType has 5 unique levels:', pd_training['MasVnrType'].unique())

print("")

print("number of houses where MasVnrType is not None and MasVnrArea is missing",pd_training[(pd_training['MasVnrType'] == 'NA') & (pd_training['MasVnrArea'].isnull())][['MasVnrArea','MasVnrType']].shape[0])

print("number of houses where MasVnrType is NA and MasVnrArea is NA:", pd_training[(pd_training['MasVnrType']=='NA')][['MasVnrArea','MasVnrType']].shape[0])

print("number of houses where MasVnrType is None and MasVnrArea is missing:",pd_training[(pd_training['MasVnrType'] == 'None') & (pd_training['MasVnrArea'].isnull())][['MasVnrArea','MasVnrType']].shape[0])

print("number of houses where MasVnrType is None and MasVnrArea is zero:",pd_training[(pd_training['MasVnrType'] == 'None') & (pd_training['MasVnrArea'] == 0)][['MasVnrArea','MasVnrType']].shape[0])

print("number of houses where MasVnrType is not None (or NA) and MasVnrArea is zero:",pd_training[(pd_training['MasVnrType'] != 'None') & (pd_training['MasVnrArea'] == 0)][['MasVnrArea','MasVnrType']].shape[0])



# let's take a look at the distribution for MasVnrArea

plt.figure(figsize=(8,8))

plt.title('MasVnrArea')

pd_training['MasVnrArea'].hist(bins = 20);



# since the missing MasVnrArea correspond with missing MasVnrType, and there are so many MasVnrArea already equal to

# zero, let's use zero to filling in the 8 missing values of the MasVnrArea feature column

# we also do this because filling with zeros doesn't significantely change the features distribution



pd_training['MasVnrArea'] = pd_training['MasVnrArea'].fillna(0)

# check that there are no more missing values

print('any more missing values? No:', pd_training['MasVnrArea'].isnull().sum())
# Now let's take a look at the Garage Year Built feature Column:

print ('GarageType has 7 unique levels:', pd_training['GarageType'].unique())

print("")

print('number of houses with Garage Type equal to NA:', pd_training[pd_training['GarageType'] == 'NA'].shape[0])

print('number of houses with Garage Year Built equal to zero:', pd_training[pd_training['GarageYrBlt'].isnull()].shape[0])

print('number of houses with Garage Year Built equal to zero and Garage Type equal to NA:',

      pd_training[(pd_training['GarageYrBlt'].isnull()) & (pd_training['GarageType'] == 'NA')].shape[0])

# since Garage Year Built seems to correspond to no garage,

# we have two main options to deal with their missing values

# options:

# 1) impute with the mean/median depending on distribution shape so that these values don't have much 

#    impact on the target value

# 2) transform the features so that the missing value is changed to a value the algo can distinguish as a seperate case



# let's take a look at the distribution for GarageYrBlt

# since it is right skewed, let's take the median to impute the missing values

# then see how much it changes the distribution

fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.title('Original GarageYrBlt')

pd_training['GarageYrBlt'].hist(bins = 20);





fig.add_subplot(122)

garage_median = pd_training['GarageYrBlt'].median()

plt.title('GarageYrBlt with Missing Values Filled by Median')

pd_training['GarageYrBlt'].fillna(garage_median).hist(bins = 20);
# now let's the try the second option, where we apply (Garage Year Built - 2010) to get years since 2010

# and fill missing values with -10

# let's look at the distribution

pd_training['Garage_Age_since_2010'] = 2010 - pd_training['GarageYrBlt']

pd_training['Garage_Age_since_2010'] = pd_training['Garage_Age_since_2010'].fillna(-10)  

plt.figure(figsize=(5,5))

pd_training['Garage_Age_since_2010'].hist(bins = 21)

plt.xlim([-3, 115])

plt.axvline(0, color='k', linestyle='dashed', linewidth=.5);
# let's use the second option: keep the column pd_training['Garage_Age_since_2010'] and drop pd_training['GarageYrBlt']

print('shape with extra feature:', pd_training.shape)



# this drops the GarageYrBlt column 

pd_training = pd_training.drop('GarageYrBlt', 1)

print('shape with original feature removed:', pd_training.shape)
# Since the missing values for the LotFrontage feature don't have any obvious reason why they are missing

# like MasVnrArea and GarageYrBlt, let's plot LotFrontage histograms by Neighborhood to see if filling it

# with the median or mean might make sense

matplotlib.rcParams['figure.figsize'] = (16.0, 16.0)

pd_training['LotFrontage'].hist(by=pd_training['Neighborhood']);

print('before imputing missing values')

pd_training[['Neighborhood','LotFrontage']].head(8)
# looking at the histograms using the median seems to make the most sense, since the distributions aren't

# symetrical or nicely skewed (in a way that we could log transform them), the median in many cases would also

# capture the most common LotFrontage for each neighborhood



# replace the missing LotFrontage values with the median lot frontage for each neighborhood

# for more info on how transform and groupy work read pandas':

# http://pandas.pydata.org/pandas-docs/stable/groupby.html#transformation

# the transform method returns an object that is indexed the same size as the one being grouped

# so `transform(lambda x: x.fillna(x.median()))` replaces the missing values with the median of each 

# neighborhood group which you get from `pd_training.groupby("Neighborhood")`

pd_training['LotFrontage'] = pd_training.groupby("Neighborhood").transform(lambda x: x.fillna(x.median()))['LotFrontage']



# check if there are still missing values in the LotFrontage

print('missing values:', pd_training['LotFrontage'].isnull().sum())



print('after imputing missing values (compare index 8)')

pd_training[['Neighborhood','LotFrontage']].head(8)
# let's plot a scatter plot of the response column to see if we can identify outliers

plt.figure(figsize= (8,8))

plt.boxplot(pd_training['SalePrice'], sym='k.');

plt.show()



# let's look at the number of houses that are above $500,000 and $600,000

print('# of houses above $500,000:', pd_training[pd_training['SalePrice'] > 500000].shape[0])

print('# of houses above $600,000:', pd_training[pd_training['SalePrice'] > 600000].shape[0])
# let's see how the distribution changes if we remove some of the outliers

# (first if we were to remove all houses above $500,000 and second all houses above $600,000)

fig = plt.figure(figsize= (20,5))

fig.add_subplot(141)

plt.boxplot(pd_training[pd_training['SalePrice'] < 500000]['SalePrice'], sym='k.');



fig.add_subplot(142)

pd_training[pd_training['SalePrice'] < 500000]['SalePrice'].hist()



fig.add_subplot(143)

plt.boxplot(pd_training[pd_training['SalePrice'] < 600000]['SalePrice'], sym='k.');



fig.add_subplot(144)

pd_training[pd_training['SalePrice'] < 600000]['SalePrice'].hist()

plt.show()
# let's look at the distribution if we transform the response column from sales price per house to 

# sales price per square foot. 



# the response column is `SalePrice` (the property's sale price in dollars), 

# this may cause an unfair comparison between houses of different sizes, instead of house price

# let's compare price per square foot 

# Note: if using python 2.x please use `from __future__ import division` to get the correct type of division here



# pd_training['SalePrice'] = pd_training['SalePrice']/pd_training['LotArea']





figure = plt.figure(figsize = (15,5))

figure.add_subplot(121)

plt.title('Sales Price per House')

sns.distplot(pd_training['SalePrice'], hist=True, rug=True)



figure.add_subplot(122)

plt.title('Sales Price per Square Foot')

sns.distplot((pd_training['SalePrice']/pd_training['LotArea']), hist=True, rug=True)

plt.show()
# and let's take a peak at the boxplots for the Sales Price per House and Sales Price per Square Foot

figure = plt.figure(figsize = (15,5))

figure.add_subplot(121)

plt.title('Sales Price per House')

plt.boxplot(pd_training['SalePrice'], sym='k.');



figure.add_subplot(122)

plt.title('Sales Price per Square Foot')

plt.boxplot(pd_training['SalePrice']/pd_training['LotArea'], sym='k.');

plt.show()



# let's think about the outliers here as well

print('# of houses above $80/square foot:', (pd_training[pd_training['SalePrice']/pd_training['LotArea'] > 80]).shape[0] )
# Let's take a look at how the log transformation would affect the response columns distribution

figure = plt.figure(figsize = (15,15))

figure.add_subplot(221)

plt.title('Sales Price per House')

sns.distplot(pd_training['SalePrice'], hist=True, rug=True)



figure.add_subplot(222)

plt.title('Sales Price per House Transformed')

sns.distplot(np.log1p(pd_training['SalePrice']), hist=True, rug=True)



figure.add_subplot(223)

plt.title('Sales Price per Square Foot')

sns.distplot((pd_training['SalePrice']/pd_training['LotArea']), hist=True, rug=True)



figure.add_subplot(224)

plt.title('Sales Price per Square Foot Transformed')

sns.distplot(np.log1p(pd_training['SalePrice']/pd_training['LotArea']), hist=True, rug=True)

plt.show()
# for fun lets also look at what would happen if we removed the outliers from the 

# sales price per square foot

sales = pd_training[pd_training['SalePrice']/pd_training['LotArea'] < 80]["SalePrice"]

new_sales = sales /pd_training[pd_training['SalePrice']/pd_training['LotArea'] < 80]['LotArea']



figure = plt.figure(figsize = (15,15))

figure.add_subplot(221)

plt.title('Sales Price per Square Foot without Outliers')

sns.distplot(new_sales, hist=True, rug=True)



figure.add_subplot(222)

plt.title('Sales Price per Square Foot without Outliers Transformed')

sns.distplot(np.log1p(new_sales), hist=True, rug=True)



figure.add_subplot(223)

plt.title('Sales Price per Square Foot')

sns.distplot((pd_training['SalePrice']/pd_training['LotArea']), hist=True, rug=True)



figure.add_subplot(224)

plt.title('Sales Price per Square Foot Transformed')

sns.distplot(np.log1p(pd_training['SalePrice']/pd_training['LotArea']), hist=True, rug=True)

plt.show()
# let's remove the houses above $500,000 (which removes 9 observations), then transform the response 

# price per square foot, and finally take the log so that we evaluate our results like kaggle does



# remove all houses above $500,000

pd_training = pd_training[pd_training['SalePrice'] < 500000]

# convert response to price per square foot

pd_training['SalePrice'] = pd_training['SalePrice']/pd_training['LotArea']

# now take the log(x+1) transform of the response column

pd_training['SalePrice'] = np.log1p(pd_training['SalePrice'])



# here's what the final distribution looks like

plt.figure(figsize = (8,8))

plt.title('Log Transform of Sales Price per Square Foot')

sns.distplot(pd_training['SalePrice'], hist=True, rug=True)

plt.show()
# Note that we will need to take the exponential exp(x) - 1 and then multiply by the square footage to get the 

# predicted sale price in the original value range
# take a look at the data's structure (using H2O) to see how many categorical (also called factors or enums)

# exist in the dataset to get the number of categorical columns you can use

# `columns_by_type()`, which returns a list with the indices

# of each categorical column, and then get the length of that list:

print('# of categorical columns:', len(training.columns_by_type(coltype='categorical')))

print("")

# to see the strucutre of the dataset, with column type and number of factor levels use `structure()`

# the first line is the H2OFrame ID which you can reference if you wanted to see this in Flow (the H2O GUI 

# interface) as well

# the second line shows the number of observations (obs.) - # of rows - and the number of features (variables) 

# - # of columns.

# the following lines show the column title followed by the column type and a selection of the first values found in

# that column

print(training.structure())
# identify the categorical features that have an order and convert those to numerical values

# from the data_description.txt we see several categorical features that describe quality as

# 'Excellent', 'Good', 'Average', 'Fair', or 'Poor' could be replaced with

# 5,4,3,2,1 for example



# for each feature column that specifies quality, convert them to integers



# list of features that have 'Excellent', 'Good', 'Average', 'Fair', or 'Poor'

quality_list = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']



# list of features that have 'Excellent', 'Good', 'Average', 'Fair', 'Poor', or 'NA'

# for the features that include NA, since no basement could be better than a poor

# quality basement we should also keep these features an unordered categoricals

# using LabelEncoder and OneHotEncoder (shown later on)

quality_w_na_list = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 

                     'PoolQC']





# map to numeric for categorical feature that appear to have an inherent order

# keep these features also encoded features without an inherent order

to_numeric_list = ['Functional', 'Utilities', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

                  'GarageFinish', 'PavedDrive', 'Fence', 'CentralAir', 'Alley']





# dictionaries for quality lists

quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

quality_w_na_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}



# dictionaries for the `to_numeric_list` (each line corresponds to a feature's unique levels,

# except line 4 which corresponds to 'BsmtFinType1' and 'BsmtFinType2' - the feature order

# is the same as the list of feature in `to_numeric_list`)

to_num_dict = {'Typ':1, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0,

               'AllPub':3, 'NoSewr':2, 'NoSeWa':1, 'ELO':0,

               'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0,

               'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0,

               'Fin':3, 'RFn':2, 'Unf':1, 'NA':0,

               'Y':2, 'P':1, 'N':0,

               'GdPrv':4,'MnPrv':3, 'GdWo':2, 'MnWw':1, 'NA':0,

               'Y':1,'N':0,

               'Grvl':2, 'Pave': 1, 'NA':0}
# given a dictionary of mapped values, define a function to replace each categorical level

# with its corresponding number

def cat_2_num(column, dict_mapper):

    """

    converts a categorical feature level to corresponding integer with a dictionary

    column: a categorical feature column

    dict_mapper: a dictionary of the categorical columns unique levels mapped to integers

                 that have an inherent ordering to them

    returns: pandas series

    """

    # create a copy of the column

    converted_col = pd.Series(column, copy=True)

    # replace the dictionary keys with their values

    for key, value in dict_mapper.items():

        converted_col.replace(key, value, inplace=True)

    # return the converted numerical column

    return converted_col
# map the quality features that have no 'NA' level

pd_training[quality_list] = pd_training[quality_list].apply(

    lambda x: cat_2_num(x, quality_map), axis = 0)



# map the quality features that have an 'NA' level

pd_training[quality_w_na_list] = pd_training[quality_w_na_list].apply(

    lambda x: cat_2_num(x, quality_w_na_map), axis = 0)



# map the `to_numeric_list` to_num_dict

pd_training[to_numeric_list] = pd_training[to_numeric_list].apply(

    lambda x: cat_2_num(x, to_num_dict), axis = 0)
# repeat for the test set

# map the quality features that have no 'NA' level

pd_test[quality_list] = pd_test[quality_list].apply(

    lambda x: cat_2_num(x, quality_map), axis = 0)



# map the quality features that have an 'NA' level

pd_test[quality_w_na_list] = pd_test[quality_w_na_list].apply(

    lambda x: cat_2_num(x, quality_w_na_map), axis = 0)



# map the `to_numeric_list` to_num_dict

pd_test[to_numeric_list] = pd_test[to_numeric_list].apply(

    lambda x: cat_2_num(x, to_num_dict), axis = 0)
# get the list of remaining unconverted categoricals



# the list of categorical features names

categorical_list = list(pd_training.dtypes[pd_training.dtypes == "object"].index)

# the list of names of all the already converted categoricals

converted_categoricals = quality_list + quality_w_na_list + to_numeric_list                        



# create new list of remaing categoricals to be converted

unconverted_cats = [x for x in categorical_list if x not in converted_categoricals]
# convert the remaining categorical features that don't seem to have an inherent order

# to integers using sklearn's LabelEncoder



# to use sklearn's one hot encoder, we first have to convert all the categorical feature levels to integers

# Note: H2O does categorical encoding automatically - these steps are not required for H2O but they'll help you 

# understand what H2O can do behind the scenes



# first use the LabelEncoder to convert all the categorical levels into different numbers

# then use sklearn's One Hot Encoder to convert those numbers into a sparse matrix where each column corresponds 

# to one possible value of one feature

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



# fit the label encoder to the dataset and transform the categorical levels

# (labels) to normalized encoding

pd_training[unconverted_cats] = pd_training[unconverted_cats].apply(le.fit_transform)



# repeat for the test set

pd_test[unconverted_cats] = pd_test[unconverted_cats].apply(le.fit_transform)
# now that all the categorical features are represented by integers we can use sklearn's 

# One Hot Encoder, which expands each feature with n levels into n binary features  

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()



for col in categorical_list:

    # creating an exhaustive list of all possible categorical values

    data=pd_training[[col]]

    enc.fit(data)

    # Fitting One Hot Encoding on train data

    temp = enc.fit_transform(pd_training[[col]]).toarray()

    # Changing the encoded features into a data frame with new column names

    temp = pd.DataFrame(temp, columns=[(col+"_"+str(i)) for i in data[col]

        .value_counts().index])

    # in side by side concatenation index values should be same

    # setting the index values similar to the X_train data frame

    temp = temp.set_index(pd_training.index.values)

    # adding the new One Hot Encoded varibales to the train data frame

    pd_training = pd.concat([pd_training,temp],axis=1)



    # repeat for the test set

    data=pd_test[[col]]

    enc.fit(data)

    # Fitting One Hot Encoding on train data

    temp = enc.fit_transform(pd_test[[col]]).toarray()

    # Changing the encoded features into a data frame with new column names

    temp = pd.DataFrame(temp, columns=[(col+"_"+str(i)) for i in data[col]

        .value_counts().index])

    # In side by side concatenation index values should be same

    # Setting the index values similar to the X_train data frame

    temp = temp.set_index(pd_test.index.values)

    # adding the new One Hot Encoded varibales to the train data frame

    pd_test = pd.concat([pd_test,temp],axis=1)



# take a look at the shape of the new dataframe

pd_training.shape
# save your clean data set and run a baseline model

# uncomment the line below if  you want to save the dataframe as a csv, update the path 

# so that the file gets saved to the location you want it to

# pd_training.to_csv("cleaned_pandas_frame.csv")
# standardize the features 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# get the new selection of predictors

new_predictors = list(pd_training.columns[1:])

# remove the response column 

new_predictors.remove('SalePrice')



# fit the scaler to the data

scaler.fit(pd_training[new_predictors])

print(scaler)



# transform the data to be scaled appropriately

rescaled_training_features = scaler.transform(pd_training[new_predictors])

rescaled_training_features.shape
# Use a Sklearn Generalized Linear Model to build a linear regression model

# on your array.

from sklearn import linear_model

from sklearn.model_selection import cross_val_score



# split the training dataset into a new train and test dataset

# set your features and response arrays

X = rescaled_training_features

# y is your response column (use np.array() to convert into an array)

y = np.array(pd_training['SalePrice'])





# Build a model using Ridge regression with built in cross validation

# alphas is the array of alpha values to try

regr = linear_model.RidgeCV(alphas = [.01, .1, 10, 15])

# fit a Ridge regression model to the training dataset 

regr.fit(X, y)



# use 5 fold cross validation to calculate your model's rmse

# the first line: returns a list of rmse values one for each fold

# the second line: to get a single statistic take the average score of all the folds

# note that in sklearn mse is negative, see the following for more information: https://github.com/scikit-learn/scikit-learn/issues/2439

rmse_values = np.sqrt(-cross_val_score(regr, X, y, scoring="neg_mean_squared_error", cv = 5))

print('score for each fold:', rmse_values)

# get a single evaluation metric

print(rmse_values.mean())
# remember the dataset we imported with H2O that hasn't had any data munging applied to it?

# You can use that dataset as is and run the following H2O model. 

from h2o.estimators.glm import H2OGeneralizedLinearEstimator



# use 80% of the training dataset to train the model, and 20% to validate the model. 

# set a seed so the split is reproducible

train, valid = training.split_frame(ratios = [.8], seed = 1234)



# convert the response column to use the same metric as kaggle

training['SalePrice'] = training['SalePrice'].log1p()



# build a GLM: first initialize the estimator, then train the model

glm = H2OGeneralizedLinearEstimator()

glm.train(x = predictors, y = 'SalePrice', training_frame = train, validation_frame = valid)



# for GLM: 

print('GLM train rmse:', glm.rmse(train = True))

print('GLM valid rmse', glm.rmse(valid = True))



# this part can be used to create a submission file

# predict on the test data using the GLM 

# GLM_predictions = glm.predict(test)



# take the inverse of log(1+x) with exp(x) - 1 (`using expm1`), to get back the original range of sales price values

# GLM_predictions['predict'] = GLM_predictions['predict'].expm1()



# # create a new frame with the Id column and SalePrice prediction value

# # use .columns to reset the column names

# glm_submission = test['Id'].concat(GLM_predictions)

# glm_submission.columns = ['Id', 'SalePrice']



# export the csv to the folder where this notebook lives

# h2o.export_file(glm_submission, 'glm_submission2.csv')
# now that we have an array of the correct format, we need to figure out how many principal components

# we need.

# To start use all the features you have, then plot the cumulative variance to see how many features you need

# to explain what percentage of the variance.

# As a rule of thumb use the number of features that explain 95% of the variance



from sklearn.decomposition import PCA

# set number of components to the number of features 252, see the cell above where we pring pd_training.shape

pca = PCA(n_components=252)



# fit the numpy array with your data where you exclude the response column

pca.fit(rescaled_training_features)



# get the amount of variance explained by pca

var = pca.explained_variance_ratio_



# get the cumulative variace

var_cum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



fig = plt.figure(figsize = (15, 5))

fig.add_subplot(121)

plt.plot(var_cum, 'r--');



fig.add_subplot(122)

plt.plot(var, 'b--');



# TO DO fix this and cell below, var should have values above .95

"{} of principal components that explain 95% of the variance:".format((np.nonzero(var)[0] > 95).sum())
# we see that we need 156 components to explain 95% of the variance:



from sklearn.decomposition import PCA

# set number of components to the number of features 253, see the cell above where we pring pd_training.shape

pca = PCA(n_components=156)



# fit the numpy array with your data where you exclude the response column

pca.fit(rescaled_training_features)



# get the amount of variance explained by pca

var = pca.explained_variance_ratio_



# get the cumulative variace

var_cum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



fig = plt.figure(figsize = (15, 5))

fig.add_subplot(121)

plt.plot(var_cum, 'r--');



fig.add_subplot(122)

plt.plot(var, 'b--');
# transform the dataset

reduced_training_set = pca.transform(rescaled_training_features)

print('shape of transformed dataset:', reduced_training_set.shape)
# now let's use our new dimensional reduced frame to build a model

# Use a Sklearn Generalized Linear Model to build a linear regression model

# on your array.

# Use a Sklearn Generalized Linear Model to build a linear regression model

# on your array.



# split the training dataset into a new train and test dataset

# set your features and response arrays

X_pca = rescaled_training_features

y_pca = np.array(pd_training['SalePrice'])





# Build a model with Ridge regression:

regression = linear_model.RidgeCV(alphas = [.01, .1, 10, 15, 20])

# fit the training dataset

regression.fit(X_pca, y_pca)



# use 5 fold cross validation to calculate your model's rmse

# the first line returns a list of rmse values one for each fold

# to get a single statistic take the average score of all the folds

# note that in sklearn mse is negative, see the following for more information: https://github.com/scikit-learn/scikit-learn/issues/2439

rmse_values_pca = np.sqrt(-cross_val_score(regression, X_pca, y_pca, scoring="neg_mean_squared_error", cv = 5))

print('score for each fold after using pca:', rmse_values)

# get a single evaluation metric

print(rmse_values_pca.mean())