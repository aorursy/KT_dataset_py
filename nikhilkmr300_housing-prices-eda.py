import numpy as np

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'
train = pd.read_csv(train_path, index_col='Id')

test_X = pd.read_csv(test_path, index_col='Id')    # test contains only inputs, no targets
target = 'SalePrice'

features = train.columns

features = features.drop(target)
valid_frac = 0.2

train_X, train_y = train[features], train[target]

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=valid_frac)
print(train_X.info())
plt.figure(figsize=(7, 7))

plt.title('Null values in the dataset are shown in white')

sns.heatmap(train_X.isnull(), cbar=False)
null_count_threshold = 0.1   # Columns with more than null_count_threshold percent nulls will be dropped

num_rows = train_X.shape[0]

for feature in features:

    null_count = train_X[feature].isna().sum()

    if(null_count > null_count_threshold * num_rows):

        train_X = train_X.drop(feature, axis=1)

        valid_X = valid_X.drop(feature, axis=1)

        test_X = test_X.drop(feature, axis=1)
categorical_columns = list(train_X.select_dtypes(include=['object']).columns)

numeric_columns = list(train_X.select_dtypes(include=['int', 'float']).columns)
print(len(categorical_columns))
print(len(numeric_columns))
imputer = SimpleImputer(strategy='most_frequent')

train_X[categorical_columns] = imputer.fit_transform(train_X[categorical_columns])

valid_X[categorical_columns] = imputer.transform(valid_X[categorical_columns])

test_X[categorical_columns] = imputer.transform(test_X[categorical_columns])
imputer = SimpleImputer(strategy='mean')

train_X[numeric_columns] = imputer.fit_transform(train_X[numeric_columns])

valid_X[numeric_columns] = imputer.transform(valid_X[numeric_columns])

test_X[numeric_columns] = imputer.transform(test_X[numeric_columns])
# There should now be no NaNs in train_X, valid_X and test_X.

print(f'train_X null count: {train_X.isna().sum().sum()}')

print(f'valid_X null count: {valid_X.isna().sum().sum()}')

print(f'test_X null count: {test_X.isna().sum().sum()}')
# Number of unique values in each categorical column

print('categorical_column'.ljust(25, ' ') + 'num_unique')

for categorical_column in categorical_columns:

    print(categorical_column.ljust(25, ' ') + str(train_X[categorical_column].nunique()))
num_unique_threshold = 5    # Categorical columns with more than num_unique_threshold unique values will be dropped

for categorical_column in categorical_columns:

    num_unique = train_X[categorical_column].nunique()

    if(num_unique > num_unique_threshold):

        train_X = train_X.drop(categorical_column, axis=1)

        valid_X = valid_X.drop(categorical_column, axis=1)

        test_X = test_X.drop(categorical_column, axis=1)

        categorical_columns.remove(categorical_column)
corrs = train_X.corrwith(train_y).abs()
plt.figure(figsize=(8, 7))

plt.xlabel('Correlation (absolute magnitude)')

plt.title('Correlation of numerical features with target')

sns.barplot(x=corrs, y=numeric_columns)
print('numeric_column'.ljust(25, ' ') + 'correlation')

for numeric_column, corr in zip(numeric_columns, corrs):

    print(numeric_column.ljust(25, ' ') + str(round(corr, 3)))
# Removing numeric variables with corr < corr_threshold

corr_threshold = 0.2

for numeric_column, stddev in zip(numeric_columns, corrs):

    if(corr < corr_threshold):

        train_X = train_X.drop(numeric_column, axis=1)

        numeric_columns.remove(numeric_column)
print(len(numeric_columns))
corr_matrix = train_X[numeric_columns].corr().abs()
plt.figure(figsize=(10, 8))

plt.title('Correlation map (absolute values)')

sns.heatmap(corr_matrix)
stddevs = train_X[numeric_columns].std(axis=0)
plt.figure(figsize=(8, 7))

plt.xlabel('Standard deviation (log scale)')

plt.title('Standard deviations of numerical variables')

sns.barplot(x=stddevs, y=numeric_columns, log=True)
print('numeric_column'.ljust(25, ' ') + 'stddev')

for numeric_column, stddev in zip(numeric_columns, stddevs):

    print(numeric_column.ljust(25, ' ') + str(round(stddev, 3)))
# Removing numeric variables with standard deviation < stddev_threshold

stddev_threshold = 1

for numeric_column, stddev in zip(numeric_columns, stddevs):

    if(stddev < stddev_threshold):

        train_X = train_X.drop(numeric_column, axis=1)

        numeric_columns.remove(numeric_column)
print(len(numeric_columns))
# Numerical columns

train_X_numerical = train_X[numeric_columns].values
# Ordinal encoded categorical columns

ordinal_encoder = OrdinalEncoder()

train_X_categorical_oe = ordinal_encoder.fit_transform(train_X[categorical_columns])
print(train_X_numerical)

print(f'train_X_numerical.shape: {train_X_numerical.shape}')

print(f'Number of numerical features = {len(numeric_columns)}')
print(train_X_categorical_oe)

print(f'train_X_categorical_oe.shape: {train_X_categorical_oe.shape}')

print(f'Number of categorical features = {len(categorical_columns)}')
alphas = np.linspace(0, 10000000, 50)    # alpha parameter for LASSO
# Rows --> Values of alpha, Columns --> Coefficients for a particular value of alpha

coef_trails = list()



# Keeps track of the number of alpha values for which the ith coefficient is non-zero

coef_lifetimes = np.zeros((train_X_numerical.shape[1]))



for alpha in alphas:

    regressor = Lasso(alpha=alpha)

    regressor.fit(train_X_numerical, train_y)

    coef_trails.append(regressor.coef_)

    

    for i, coef in enumerate(regressor.coef_):

        if(coef != 0):

            coef_lifetimes[i] += 1

            

coef_trails = np.array(coef_trails)
sns.set_palette(sns.color_palette('hls', coef_trails.shape[1]))

plt.figure(figsize=(8, 7))

plt.title('Coefficient trails for numerical variables')

plt.xlabel('alpha parameter for LASSO')

plt.ylabel('Coefficient value')

plt.ylim([None, 300])

for i in range(coef_trails.shape[1]):

    plt.plot(alphas, coef_trails[:, i], label=numeric_columns[i])

plt.legend()
plt.figure(figsize=(8, 7))

plt.xlabel('Coefficient lifetime')

plt.xticks([])    # Setting xticks to empty list as actual lifetime is irrelevant, only relative order of lifetime matters

sns.barplot(x=coef_lifetimes, y=numeric_columns)
alphas = np.linspace(0, 100000, 50)    # alpha parameter for LASSO
# Rows --> Values of alpha, Columns --> Coefficients for a particular value of alpha

coef_trails = list()



# Keeps track of the number of alpha values for which the ith coefficient is non-zero

coef_lifetimes = np.zeros((train_X_categorical_oe.shape[1]))



for alpha in alphas:

    regressor = Lasso(alpha=alpha)

    regressor.fit(train_X_categorical_oe, train_y)

    coef_trails.append(regressor.coef_)

    

    for i, coef in enumerate(regressor.coef_):

        if(coef != 0):

            coef_lifetimes[i] += 1

            

coef_trails = np.array(coef_trails)
sns.set_palette(sns.color_palette('hls', coef_trails.shape[1]))

plt.figure(figsize=(9, 8))

plt.title('Coefficient trails for categorical variables')

plt.xlabel('alpha parameter for LASSO')

plt.ylabel('Coefficient value')

for i in range(coef_trails.shape[1]):

    plt.plot(alphas, coef_trails[:, i], label=categorical_columns[i])

plt.legend()
plt.figure(figsize=(8, 7))

plt.xlabel('Coefficient lifetime')

plt.xticks([])    # Setting xticks to empty list as actual lifetime is irrelevant, only relative order of lifetime matters

sns.barplot(x=coef_lifetimes, y=categorical_columns)
selector = SelectKBest(f_regression, k=train_X_numerical.shape[1])

selector.fit(train_X_numerical, train_y)
# pvalues for each of the numerical variables in numeric_columns

pvalues = selector.pvalues_



# List of tuples of categorical variable to its pvalue

variable_pvalues = [(numeric_column, pvalues[i]) for i, numeric_column in enumerate(numeric_columns)]



# Sorting variable_pvalues in order of pvalue

variable_pvalues.sort(key=lambda x: x[1])



print('numeric_variable'.ljust(25, ' ') + 'pvalue (sorted)')

for variable, pvalue in variable_pvalues:

    print(variable.ljust(25, ' ') + str(pvalue))
selector = SelectKBest(f_regression, k=train_X_categorical_oe.shape[1])

selector.fit(train_X_categorical_oe, train_y)
# pvalues for each of the categorical variables in categorical_columns

pvalues = selector.pvalues_



# List of tuples of categorical variable to its pvalue

variable_pvalues = [(categorical_column, pvalues[i]) for i, categorical_column in enumerate(categorical_columns)]



# Sorting variable_pvalues in order of pvalue

variable_pvalues.sort(key=lambda x: x[1])



print('categorical_variable'.ljust(25, ' ') + 'pvalue (sorted)')

for variable, pvalue in variable_pvalues:

    print(variable.ljust(25, ' ') + str(pvalue))
final_numeric_columns = [    'LotArea', 

                             '1stFlrSF', 

                             'GarageArea', 

                             'BsmtFinSF1', 

                             'BsmtUnfSF', 

                             'FullBath', 

                             'TotRmsAbvGrd', 

                             'YearRemodAdd', 

                             'OverallCond']

final_categorical_columns = ['HeatingQC', 

                             'BsmtQual', 

                             'GarageType', 

                             'KitchenQual', 

                             'LotShape', 

                             'ExterQual', 

                             'BsmtExposure', 

                             'GarageFinish', 

                             'Electrical']



final_columns = final_numeric_columns + final_categorical_columns
mean = train_X[final_numeric_columns].mean(axis=0)

stddev = train_X[final_numeric_columns].std(axis=0)

train_X[final_numeric_columns] = (train_X[final_numeric_columns] - mean) / stddev

valid_X[final_numeric_columns] = (valid_X[final_numeric_columns] - mean) / stddev

test_X[final_numeric_columns] = (test_X[final_numeric_columns] - mean) / stddev
train_X_final = train_X[final_columns]

valid_X_final = valid_X[final_columns]

test_X_final = test_X[final_columns]
# One hot encoding categorical features in train, valid and test by fitting on train

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



# Categorical columns one hot encoded as NumPy arrays

train_X_one_hot = one_hot_encoder.fit_transform(train_X_final[final_categorical_columns])

valid_X_one_hot = one_hot_encoder.transform(valid_X_final[final_categorical_columns])

test_X_one_hot = one_hot_encoder.transform(test_X_final[final_categorical_columns])



# Converting one hot encoded NumPy arrays to Pandas DataFrame

train_X_one_hot = pd.DataFrame(train_X_one_hot)

valid_X_one_hot = pd.DataFrame(valid_X_one_hot)

test_X_one_hot = pd.DataFrame(test_X_one_hot)



# One hot encoding created NumPy arrays, so indexing was lost, reindexing

train_X_one_hot.index = train_X_final.index

valid_X_one_hot.index = valid_X_final.index

test_X_one_hot.index = test_X_final.index



# Removing the original categorical features columns as they have been one hot encoded

train_X_final = train_X_final.drop(final_categorical_columns, axis=1)

valid_X_final = valid_X_final.drop(final_categorical_columns, axis=1)

test_X_final = test_X_final.drop(final_categorical_columns, axis=1)



# Adding the one hot encoded columns to the final dataframes

train_X_final = pd.concat([train_X_final, train_X_one_hot], axis=1)

valid_X_final = pd.concat([valid_X_final, valid_X_one_hot], axis=1)

test_X_final = pd.concat([test_X_final, test_X_one_hot], axis=1)
print(f'Numeric column count = {len(final_numeric_columns)}')



# Number of one hot columns should equal the total number of unique classes in each categorical feature

one_hot_column_count = train_X[final_categorical_columns].nunique().sum()

print(f'One hot column count = {one_hot_column_count}')



print(f'Categorical column count = {len(final_categorical_columns)}')



# Total column count = Number of numeric columns + Number of one hot columns

print(f'Total column count must be = {len(final_numeric_columns) + one_hot_column_count}')
print(train_X_final.shape)

print(valid_X_final.shape)

print(test_X_final.shape)
categories = one_hot_encoder.categories_

print(categories)
i = 0

# Provides mapping from old one hot columns to new names for those columns

rename_dict = dict()

for category_name, category in zip(final_categorical_columns, categories):

    for category_class in category:

        rename_dict[i] = category_name + '_' + category_class

        i += 1

print(rename_dict)
train_X_final = train_X_final.rename(columns=rename_dict)

valid_X_final = valid_X_final.rename(columns=rename_dict)

test_X_final = test_X_final.rename(columns=rename_dict)
train_final = pd.concat([train_X_final, train_y], axis=1).sort_index()

valid_final = pd.concat([valid_X_final, valid_y], axis=1).sort_index()

"""

train_final.to_csv('data/train_final.csv')

valid_final.to_csv('data/valid_final.csv')

test_X_final.to_csv('data/test_X_final.csv')

"""