import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns

from tqdm import tqdm

from datetime import datetime

import json

import os

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Imputer

from scipy.stats import skew 

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

# models

from xgboost import XGBRegressor

import warnings

# Ignore useless warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# Avoid runtime error messages

pd.set_option('display.float_format', lambda x:'%f'%x)



# make notebook's output stable across runs

np.random.seed(42)
# Read CSVs

fetch_from = '../input/train.csv'

train = pd.read_csv(fetch_from)



fetch_from = '../input/test.csv'

test = pd.read_csv(fetch_from)
# How many datapoints in the training set?

train.shape
# How many datapoints in the test set?

test.shape
# Look at sample datapoints in the training set

train.sample(5)
# Look at sample datapoints in the test set

test.sample(5)
# How many missing values does the dataset have?

train.isnull().sum().sum()
# Which columns have the most missing values?

def missing_data(df):

    total = df.isnull().sum()

    percent = (df.isnull().sum()/train.isnull().count()*100)

    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        types.append(dtype)

    missing_values['Types'] = types

    missing_values.sort_values('Total',ascending=False,inplace=True)

    return(np.transpose(missing_values))

missing_data(train)
# Let's plot these missing values(%) vs column_names

missing_values_count = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)

plt.figure(figsize=(15,10))

base_color = sns.color_palette()[0]

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color = base_color)
train.describe()
test.describe()
train.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)

plt.show()
test.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)

plt.show()
train.describe(include='O')
test.describe(include='O')
train_eda = train.copy()

label_col = 'SalePrice'
col_name = 'HouseStyle'

freq_table = pd.crosstab(index=train_eda[col_name],  # Make a crosstab

                              columns="count")      # Name the count column

freq_table_per = pd.crosstab(index=train_eda[col_name],  # Make a crosstab

                              columns="percentage", normalize=True)

freq_table['percentage'] = freq_table_per['percentage']

freq_table.sort_values(by='count', ascending=False)
# First, we need to create a function to decide how many blocks to allocate to each category

def percentage_blocks(df, var):

    """

    Take as input a dataframe and variable, and return a Pandas series with

    approximate percentage values for filling out a waffle plot.

    """

    # compute base quotas

    percentages = 100 * df[var].value_counts() / df.shape[0]

    counts = np.floor(percentages).astype(int) # integer part = minimum quota

    decimal = (percentages - counts).sort_values(ascending = False)

    # add in additional counts to reach 100

    rem = 100 - counts.sum()

    for cat in decimal.index[:rem]:

        counts[cat] += 1

    return counts



# Second, plot those counts as boxes in the waffle plot form

waffle_counts = percentage_blocks(train_eda, col_name)

prev_count = 0

# for each category,

for cat in range(waffle_counts.shape[0]):

    # get the block indices

    blocks = np.arange(prev_count, prev_count + waffle_counts[cat])

    # and put a block at each index's location

    x = blocks % 10 # use mod operation to get ones digit

    y = blocks // 10 # use floor division to get tens digit

    plt.bar(x = x, height = 0.9, width = 0.9, bottom = y)

    prev_count += waffle_counts[cat]



# Third, we need to do involve aesthetic cleaning to polish it up for interpretability. We can take away the plot border and ticks, since they're arbitrary, but we should change the limits so that the boxes are square. We should also add a legend so that the mapping from colors to category levels is clear.

# aesthetic wrangling

plt.legend(waffle_counts.index, bbox_to_anchor = (1, 0.5), loc = 6)

plt.axis('off')

plt.axis('square')
# Area depicts the distribution of points. > width = > the number of points

base_color = sns.color_palette()[0]

plt.figure(figsize=(20,15))

plt.xticks(rotation=45)

sns.boxplot(data = train_eda, x = 'HouseStyle', y = 'SalePrice', color = base_color);
col_name = 'HouseStyle'

freq_table = pd.crosstab(index=train_eda[col_name],  # Make a crosstab

                              columns="count")      # Name the count column

freq_table_per = pd.crosstab(index=train_eda[col_name],  # Make a crosstab

                              columns="percentage", normalize=True)

freq_table['percentage'] = freq_table_per['percentage']

freq_table.sort_values(by='count', ascending=False)
# Relative frequency variation - Plotting absolute counts on axis and porportions on the bars

# Barchart sorted by frequency

base_color = sns.color_palette()[0]

cat_order = train_eda[col_name].value_counts().index

plt.figure(figsize=(15,10))

plt.xticks(rotation = 90)

sns.countplot(data = train_eda, x = col_name, order = cat_order, color = base_color);



# add annotations

n_points = train_eda.shape[0]

cat_counts = train_eda[col_name].value_counts()

locs, labels = plt.xticks() # get the current tick locations and labels



# loop through each pair of locations and labels

for loc, label in zip(locs, labels):



    # get the text property for the label to get the correct count

    count = cat_counts[label.get_text()]

    pct_string = '{:0.1f}%'.format(100*count/n_points)



    # print the annotation just below the top of the bar

    plt.text(loc, count+4, pct_string, ha = 'center', color = 'black')
# Area depicts the distribution of points. > width = > the number of points

base_color = sns.color_palette()[0]

plt.figure(figsize=(20,15))

plt.xticks(rotation=45)

sns.boxplot(data = train_eda, x = 'Neighborhood', y = 'SalePrice', color = base_color);
col_name = 'GrLivArea'

hist_kws={"alpha": 0.3}

plt.figure(figsize=(15,10))

# Trim long-tail/other values

# plt.xlim(0, 1200)

sns.distplot(train_eda[col_name], hist_kws=hist_kws);
col_name = 'OverallQual'

hist_kws={"alpha": 0.3}

plt.figure(figsize=(15,10))

# Trim long-tail/other values

# plt.xlim(0, 1200)

sns.distplot(train_eda[col_name], hist_kws=hist_kws);
col_name = 'OverallQual'

base_color = sns.color_palette()[0]

plt.figure(figsize=(20,15))

plt.xticks(rotation=45)

sns.boxplot(data = train_eda, x = col_name, y = 'SalePrice', color = base_color)
# Which features are the most correlated to our target variable, SalePrice?

corr_matrix = train_eda.corr()

plt.subplots(figsize=(15,10))

sns.heatmap(corr_matrix, vmax=1.0, square=True, cmap="Blues")
# Get the top 10 most correlated features

corr_matrix = train_eda.corr()

corr_matrix[label_col].sort_values(ascending=False)[:10]
from pandas.plotting import scatter_matrix



attributes = [label_col, "OverallQual", "LotArea", "BedroomAbvGr", "GrLivArea"]

scatter_matrix(train_eda[attributes], figsize=(15, 15));
col_name = 'GarageArea'

train_eda.plot(kind="scatter", x=label_col, y=col_name, alpha=0.2, figsize=(15,10))

# changing axis labels to only show part of the graph

plt.axis([0, 400000, 0, 1200])
train_eda.plot(kind="scatter", x=label_col, y="GarageArea", alpha=0.4,

             s=train_eda["BedroomAbvGr"], label="BedroomAbvGr", figsize=(20,15),

             c="YrSold", cmap=plt.get_cmap("jet"), colorbar=True,)

plt.axis([0, 400000, 0, 1200])

plt.legend();
train_fe = train.copy()

test_fe = test.copy()
train_fe = train_fe[train_fe.GrLivArea < 4300]

train_fe.reset_index(drop=True, inplace=True)



train_fe["SalePrice"] = np.log1p(train_fe["SalePrice"])

y = train_fe.SalePrice.reset_index(drop=True)

train_features = train_fe.drop(['SalePrice'], axis=1)

test_features = test.copy()



features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)
features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features["PoolQC"] = features["PoolQC"].fillna("None")

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

features.update(features[objects].fillna('None'))
# Convert categorical variables stored as numbers to strings

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
# Remove Id

train_ID = train_fe['Id']

test_ID = test_fe['Id']



train_fe.drop(['Id'], axis=1, inplace=True)

test_fe.drop(['Id'], axis=1, inplace=True)



# Create new features

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)



# Fix skewed features

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

final_features = pd.get_dummies(features).reset_index(drop=True)



X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(X):, :]



outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])



overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = X.drop(overfit, axis=1).copy()

X_sub = X_sub.drop(overfit, axis=1).copy()
X.head()
X.info()
features_train = train.copy()

features_train.dropna(axis=0, subset=['SalePrice'], inplace=True)

label = features_train.SalePrice

features_train = features_train.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(features_train.as_matrix(), label.as_matrix(), test_size=0.25)



my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
xgb = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006, random_state=42);



xgb.fit(train_X, train_y)

predictions = xgb.predict(test_X)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



def mae(y, y_pred):

    return mean_absolute_error(predictions, test_y)
print('MAE score on train data:')

print(mae(predictions, test_y))