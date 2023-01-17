# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read both train and test datasets as a DataFrame

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
numerical_train = train.select_dtypes(include=['int', 'float'])

numerical_train.drop(['Id', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'], axis=1, inplace=True)
train.head()
sns.distplot(train['SalePrice']);
# Statistics about the data

train.describe()
train.info()
# Seperate out the numerical data

numerical_train = train.select_dtypes(include=['int', 'float'])

numerical_train.head()
# Plot the distributions of all numerical data. 

i = 1

fig = plt.figure(figsize=(40,50))

for item in numerical_train:

    axes = fig.add_subplot(8,5,i)

    axes = numerical_train[item].plot.hist(rot=0, subplots=True)

    plt.xticks(rotation=45)

    i += 1
# Seperate out the categorical data

categorical_train = train.select_dtypes(include=['object'])

categorical_train.head()
# Plot the counts of all categorical data. 

i = 1

fig = plt.figure(figsize=(40,50))

for item in categorical_train:

    axes = fig.add_subplot(9,5,i)

    axes = categorical_train[item].value_counts().plot.bar(rot=0, subplots=True)

    plt.xticks(rotation=45)

    i += 1
# Boxplot all categorical data with SalePrice

i = 1

fig = plt.figure(figsize=(40,50))

for item in categorical_train:

    data = pd.concat([train['SalePrice'], categorical_train[item]], axis=1)

    axes = fig.add_subplot(9,5,i)

    axes = sns.boxplot(x=item, y="SalePrice", data=data)

    plt.xticks(rotation=45)

    i += 1
# Correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
# Correlation matrix with strong correlations with SalePrice

sorted_corrs = train.corr()['SalePrice'].abs().sort_values()

strong_corrs = sorted_corrs[sorted_corrs > 0.5]

cols = strong_corrs.index

corrmat = train[strong_corrs.index].corr()

sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# The histogram on the diagonal is the distribution of a single variable 

# The scatter plots represent the relationships between two variables

sns.set()

cols = strong_corrs.index

sns.pairplot(numerical_train[cols], height=2.5)

plt.show()
df = train
def transform_features(df):

    # Count number of missing values in each numerical column

    num_missing = df.isnull().sum()

    # Drop the columns where at least 5% of the values are missing

    drop_missing_cols = num_missing[(num_missing > len(df)/20)].sort_values()

    df = df.drop(drop_missing_cols.index, axis=1)

    # Count number of missing values in each categorical column

    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    # Drop the columns where at least 1 missing value

    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]

    df = df.drop(drop_missing_cols_2.index, axis=1)

    # For numerical columns with missing values calcualate number of missing values

    num_missing = df.select_dtypes(include=['int', 'float']).isnull().sum()

    fixable_numeric_cols = num_missing[(num_missing <= len(df)/20) & (num_missing > 0)].sort_values()

    # Calcualte the most common value for each column

    replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]

    # For numerial columns with missing values fill with most common value in that column

    df = df.fillna(replacement_values_dict)

    # Compute two new columns by combining other columns which could be useful

    years_sold = df['YrSold'] - df['YearBuilt']

    years_since_remod = df['YrSold'] - df['YearRemodAdd']

    df['YearsBeforeSale'] = years_sold

    df['YearsSinceRemod'] = years_since_remod

    # Drop the no longer needed original year columns

    df = df.drop(["YearBuilt", "YearRemodAdd"], axis = 1)

    # Remove irrelevant data

    df = df.drop(["Id"], axis=1)

    return df
transform_features(df)
def select_features(df, uniq_threshold):

    # Create list of all column names that are supposed to be categorical

    nominal_features = ["Id", "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", 

                    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", 

                    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", 

                    "MiscFeature", "SaleType", "SaleCondition"]

    # Check which categorical columns we have carried with us

    transform_cat_cols = []

    for col in nominal_features:

        if col in df.columns:

            transform_cat_cols.append(col)

    # Check how many unique values in each categorical column

    uniqueness_counts = df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()

    # For each item that has more than the defined unique threshold values, create category 'Other'

    for item in uniqueness_counts.iteritems():

        if item[1] >= uniq_threshold:

            # Count unique values in the column

            unique_val = df[item[0]].value_counts()

            # Select the 10th least common index and the rest lower than that

            other_index = unique_val.loc[unique_val < unique_val.iloc[uniq_threshold - 2]].index

            df.loc[df[item[0]].isin(list(other_index)), item[0]] = 'Other'

    # Select the text columns and convert to categorical

    text_cols = df.select_dtypes(include=['object'])

    for col in text_cols:

        df[col] = df[col].astype('category')

    # Create dummy columns

    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include=['category']))], axis=1).drop(text_cols,axis=1)

    return df
def drop_features(df, coeff_threshold):

    # Select numerical columns

    numerical_df = df.select_dtypes(include=['int', 'float'])

    #print(numerical_df)

    # Compute the absolute correlation between the numerical columns and SalePrice

    abs_corr_coeffs = numerical_df.corr()['SalePrice'].abs().sort_values()

    #print(abs_corr_coeffs)

    # Drop the columns that have a coefficient lower than than the defined threshold

    df = df.drop(abs_corr_coeffs[abs_corr_coeffs < coeff_threshold].index, axis=1)

    return df
#split the data to train the model 

#y = train.SalePrice

#X_train,X_test,y_train,y_test = train_test_split(train_df.drop(['SalePrice'], axis=1) ,y ,test_size=0.2 , random_state=0)
# Scale the data

#scaler = StandardScaler().fit(train_df[features])

#rescaled_train_df = scaler.transform(train_df[features])

#rescaled_test_df = scaler.transform(test_df[features])



#model = linear_model.LinearRegression()

#model.fit(train_df[features], train["SalePrice"])

#predictions = model.predict(test_df[features])
transform_train_df = transform_features(train)

transform_test_df = transform_features(test)

train_df = select_features(transform_train_df, uniq_threshold=100)

test_df = select_features(transform_test_df, uniq_threshold=100)
train_features = drop_features(train_df, coeff_threshold=0.01)

test_features = test_df.columns

features = pd.Series(list(set(train_features) & set(test_features)))
X_train = train_df[features]

y_train = train["SalePrice"]

X_test = test_df[features]

X_train
#rfgs_parameters = {

#    'n_estimators': [50],

#    'max_depth'   : [n for n in range(2, 16)],

#    'max_features': [n for n in range(2, 16)],

#    "min_samples_split": [n for n in range(2, 8)],

#    "min_samples_leaf": [n for n in range(2, 8)],

#    "bootstrap": [True,False]

#}

#rfr_cv = GridSearchCV(RandomForestRegressor(), rfgs_parameters, cv=8, scoring='neg_mean_squared_log_error')
#rfr_cv.fit(X_train, y_train)
#predictions = rfr_cv.predict(X_test)
model_rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=42)
model_rf.fit(X_train, y_train)
predictions = model_rf.predict(X_test)
# Output the predictions into a csv

submission = pd.DataFrame(test.Id)

predictions = pd.DataFrame({'SalePrice': predictions})

output = pd.concat([submission,predictions],axis=1)

output.to_csv('submission.csv', index=False)