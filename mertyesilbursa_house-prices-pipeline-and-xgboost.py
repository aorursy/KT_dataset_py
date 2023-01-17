import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

def report_missing_unique(df):

    """

    Enter your data frame. Returns a DataFrame object with the following columns:

    

    MissingCount: number of missing values per column

    MissingPercent: percentage of missing values per column

    DataType: data type of column

    UniqueCount: number of unique values per column

    UniqueList: list of unique values per column

    """

    

    count_missing_list = []

    percent_missing_list = []

    datatypes_list = []

    count_unique_list = []

    unique_values_list = []

    features_list = []

    

    for col in df.columns:

        count_missing_list.append(df[col].isna().sum())

        percent_missing_list.append(round((df[col].isna().sum() / len(df[col]))*100, 3))

        datatypes_list.append(df[col].dtypes)

        count_unique_list.append(df[col].nunique())

        unique_values_list.append(df[col].unique())

        features_list.append(col)

    

    summary_df = pd.DataFrame({'MissingCount':count_missing_list, 

                               'MissingPercent':percent_missing_list, 

                               'DataType':datatypes_list, 

                               'UniqueCount':count_unique_list, 

                               'UniqueList':unique_values_list})

    

    summary_df.index = features_list

    

    return summary_df.sort_values(by="MissingCount",ascending=False)

# Read the data

train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Set the target to the 'SalePrice' column of train

y = train[['SalePrice']]

# Show the initial number of rows and columns of the data sets train and test

print("train: ", train.shape)

print("test: ", test.shape)

# concatenate the train and test features

X = pd.concat([train.drop("SalePrice", axis=1), test], axis=0)

print("X: ", X.shape)

# get a report on the missing and unique values of features with missing values

mu_df = report_missing_unique(X)

mu_df[mu_df.MissingCount > 0]

# 'MSSubClass' is actually a categorical column that is currently formatted as integer

X['MSSubClass'] = X['MSSubClass'].astype(str)

train['MSSubClass'] = train['MSSubClass'].astype(str)

test['MSSubClass'] = test['MSSubClass'].astype(str)

# Numerical features

df_num = X.select_dtypes(exclude=['object']).copy()

df_cat = X.select_dtypes(include=['object']).copy()



print(f"There are {df_num.shape[1]} numerical columns.")

print(f"There are {df_cat.shape[1]} categorical columns.")

# list discrete numerical variables

list_num_discrete = []

for i in df_num.columns:

    if len(df_num[i].unique()) < 20:

        list_num_discrete.append(i)



# PoolArea is not a discrete continuous variable, so removing that from the list

list_num_discrete.remove('PoolArea')

        

# list continuous numerical variables

list_num_continuous = []

for i in df_num.columns:

    if i not in list_num_discrete:

        list_num_continuous.append(i)

        

print(f"There are {len(list_num_discrete)} discrete numerical columns.")

print(f"There are {len(list_num_continuous)} continuous numerical columns.")

df_num_train = train.select_dtypes(exclude=['object']).copy()

df_cat_train = train.select_dtypes(include=['object']).copy()



# previous step excluded 'SalePrice' from df_cat_train as the target is numeric, bring it back

df_cat_train = pd.concat([df_cat_train, train['SalePrice']], axis=1)



df_num_disc_train = df_num_train.drop(list_num_continuous, axis=1)

df_num_cont_train = df_num_train.drop(list_num_discrete, axis=1)

fig = plt.figure(figsize=(20,20))

for index in range(len(df_num_cont_train.columns)-1):

    plt.subplot(6,4,index+1)

    sns.scatterplot(x=df_num_cont_train.iloc[:,index], y='SalePrice', data=df_num_cont_train)

fig.tight_layout(pad=1.0)

# get the Id of this strange data point (from Extra above)

X[X['GarageYrBlt'] >= 2020]['GarageYrBlt']

# Id is high (higher than 1460) so this is from test data

# this looks like a typo and the true value is likely 2007

test.loc[2593,'GarageYrBlt'] = 2007

X.loc[2593,'GarageYrBlt'] = 2007

fig = plt.figure(figsize=(16,10))

for index in range(len(df_num_disc_train.columns)-1):

    plt.subplot(4,4,index+1)

    sns.stripplot(x=df_num_disc_train.iloc[:,index], y='SalePrice', data=df_num_disc_train, jitter=True)

fig.tight_layout(pad=1.0)

fig = plt.figure(figsize=(18,30))

for index in range(len(df_cat_train.columns)-1):

    plt.subplot(11,4,index+1)

    sns.stripplot(x=df_cat_train.iloc[:,index], y='SalePrice', data=df_cat_train, jitter=True)

fig.tight_layout(pad=1.0)

# Correlation matrix

plt.figure(figsize=(14,12))

sns.heatmap(df_num.corr(), mask = df_num.corr() <0.66, linewidth=1.5, cmap='Reds')

k = 10 #number of variables for heatmap

plt.figure(figsize=(12,8))

corr_matrix = train.corr()

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, 

                square=True, fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values)

# Features with multicollinearity

X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)

# Features with a lot of missing values (>80% missing)

X.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

# Features unrelated to predict SalePrice

X.drop(['MoSold', 'YrSold'], axis=1, inplace=True)

# Features that have mostly just a single value (categorical)

X.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1, inplace=True)

# Features that have mostly just a single value (numerical)

X.drop(['LowQualFinSF', 'PoolArea', 'MiscVal'], axis=1, inplace=True)
##### NUMERIC FEATURES #####

# numeric columns where missing values are to be replaced with 0 (exception is 'LotFrontage' as below)

zero_list = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 

             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']

for col in zero_list:

    X[col] = X[col].fillna(0)

    

# Group by neighborhood and fill in missing value by the median LotFrontage of Neighborhood

X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



##### CATEGORICAL FEATURES #####

# Missing values in columns in the na list are to be filled in with the value "NA"

na_list = ['FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 

           'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 

           'BsmtFinType2', 'BsmtFinType1']

for col in na_list:

    X[col] = X[col].fillna('NA')



# Missing values in columns in the none list are to be filled in with the value "None"

none_list = ['MasVnrType']

for col in none_list:

    X[col] = X[col].fillna('None')



# Missing values in columns in the most frequent list are to be filled in with the most commonly occurring value

mf_list = ['MSZoning', 'Functional', 'Electrical', 'Exterior2nd', 

           'Exterior1st', 'SaleType', 'KitchenQual']

for col in mf_list:

    X[col] = X[col].fillna(X[col].mode()[0])
X['TotalWalledArea'] = X['TotalBsmtSF'] + X['GrLivArea']

X['TotalPorchArea'] = X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF']

X['TotalOccupiedArea'] = X['TotalWalledArea'] + X['TotalPorchArea']
map1 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}

set1 = ['FireplaceQu', 'GarageCond', 'GarageQual', 'ExterQual', 

        'ExterCond', 'BsmtCond', 'BsmtQual', 'HeatingQC', 'KitchenQual']

for col in set1:

    X[col] = X[col].replace(map1)



map2 = {'GLQ': 4,'ALQ': 3,'BLQ': 2,'Rec': 3,'LwQ': 2,'Unf': 1,'NA': 0}

set2 = ['BsmtFinType1', 'BsmtFinType2']

for col in set2:

    X[col] = X[col].replace(map2)    

    

map3 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

X['BsmtExposure'] = X['BsmtExposure'].replace(map3)



map4 = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}

X['GarageFinish'] = X['GarageFinish'].replace(map4)



map5 = {'Y': 1, 'N': 0}

X['CentralAir'] = X['CentralAir'].replace(map5)



map6 = {'Typ': 3.5, 'Min1': 3, 'Min2': 2.5, 'Mod': 2, 'Maj1': 1.5, 'Maj2': 1, 'Sev': 0.5, 'Sal': 0}

X['Functional'] = X['Functional'].replace(map6)



map7 = {'Y': 1, 'P': 0.5, 'N': 0}

X['PavedDrive'] = X['PavedDrive'].replace(map7)
# let's have quick look if there are any missing values

new_mu_df = report_missing_unique(X)

new_mu_df[new_mu_df.MissingCount > 0]
# Select categorical columns

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# let's have a quick look how many unique entries there are for our (remaining) categorical features, as OneHotEncoder will need that many new columns

# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X[col].nunique(), categorical_cols))

d = dict(zip(categorical_cols, object_nunique))

# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# define the transformers



numeric_transformer = Pipeline(steps=[

    ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))

])



categorical_transformer = Pipeline(steps=[

    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))

])

    

preprocessor = ColumnTransformer(transformers=[

        ('num', numeric_transformer, numeric_cols),

        ('cat', categorical_transformer, categorical_cols)], 

        remainder='passthrough')
# define the model



model = XGBRegressor(n_estimators=1000, 

                     learning_rate = 0.02, 

                     max_depth = 3,

                     subsample = 0.8,

                     gamma = 1,

                     random_state = 0)
# define the pipeline



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# splitting the train and test data

train = X[:train.shape[0]]

test = X[train.shape[0]:]
# cross-validation with 5-folds



scores = -cross_val_score(my_pipeline, train, y, cv=5, scoring='neg_mean_absolute_error')



print("MAE scores: ", scores)
print("Average MAE score (across experiments): {:.3f}".format(scores.mean()))
# fit model to the full train data

my_pipeline.fit(train, y)



# Generate test predictions

preds_test = my_pipeline.predict(test)



# Save test predictions to file

output = pd.DataFrame({'Id': test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

print ("Submission file is saved.")