# importing necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as pl 

import scipy.stats as st

from IPython.display import HTML, display

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace = True)

missing.plot.bar()

plt.show()
train.info()
# Finding Numerical and Categorical columns with missing values

cols_with_missing_num = [col for col in train.columns if train[col].isnull().any() and train[col].dtype in ['int64', 'float64']]

cols_with_missing_obj = [col for col in train.columns if train[col].isnull().any() and train[col].dtype == 'object']

print(cols_with_missing_num)

print(cols_with_missing_obj)
# Finding Numerical and Categorical columns with missing values

cols_with_missing_num_test = [col for col in test.columns if test[col].isnull().any() and test[col].dtype in ['int64', 'float64']]

cols_with_missing_obj_test = [col for col in test.columns if test[col].isnull().any() and test[col].dtype == 'object']

print(cols_with_missing_num_test)

print(cols_with_missing_obj_test)
# Identifying columns that have missing values in test dataset but not in train

col_drop_num = list(set(cols_with_missing_num_test) - set(cols_with_missing_num))

col_drop_obj = list(set(cols_with_missing_obj_test) - set(cols_with_missing_obj))

print(col_drop_num)

print(col_drop_obj)
test.shape[0]
test[col_drop_num].isnull().sum()
# Loading necessary packages for imputation



from sklearn.impute import SimpleImputer

from category_encoders.one_hot import OneHotEncoder

from sklearn.preprocessing import LabelEncoder
# Preprocessing numerical features with SimpleImputer

my_imputer = SimpleImputer(strategy='constant')

imputed_train = pd.DataFrame(my_imputer.fit_transform(train[cols_with_missing_num]))

imputed_val = pd.DataFrame(my_imputer.transform(test[cols_with_missing_num]))



imputed_train.columns = train[cols_with_missing_num].columns

imputed_val.columns = test[cols_with_missing_num].columns
imputed_train.head()
imputed_val.head()
train = train.drop(columns=cols_with_missing_num, axis = 1)

test = test.drop(columns=cols_with_missing_num, axis = 1)



imputed_train.index = train.index

imputed_val.index = test.index



train = pd.concat([train, imputed_train], axis = 1)

test = pd.concat([test, imputed_val], axis = 1)
train[cols_with_missing_num].isnull().sum()
test[cols_with_missing_num].isnull().sum()
#Imputing numerical columns in col_drop_num

imputer = SimpleImputer(strategy='mean')

imputer.fit(train[col_drop_num])

imputed_val = pd.DataFrame(imputer.transform(test[col_drop_num]))



imputed_val.columns = test[col_drop_num].columns
train[col_drop_num].isnull().sum()
imputed_val.isnull().sum()
test = test.drop(columns=col_drop_num, axis = 1)



imputed_val.index = test.index



test = pd.concat([test, imputed_val], axis = 1)
test[col_drop_num].isnull().sum()
# Imputing Categorical Columns

cat_imputer1 = SimpleImputer(strategy='most_frequent')

imputed_train = pd.DataFrame(cat_imputer1.fit_transform(train[cols_with_missing_obj]))

imputed_val = pd.DataFrame(cat_imputer1.transform(test[cols_with_missing_obj]))



imputed_train.columns = train[cols_with_missing_obj].columns

imputed_val.columns = test[cols_with_missing_obj].columns
imputed_train.head(10)
imputed_val.head(10)
train = train.drop(columns=cols_with_missing_obj, axis = 1)

test = test.drop(columns=cols_with_missing_obj, axis = 1)



imputed_train.index = train.index

imputed_val.index = test.index



train = pd.concat([train, imputed_train], axis = 1)

test = pd.concat([test, imputed_val], axis = 1)
train.head(10)
train.columns
test.head(10)
# Imputing Categorical Columns in col_drop_obj of test dataset

cat_imputer2 = SimpleImputer(strategy='most_frequent')

cat_imputer2.fit(train[col_drop_obj])

imputed_val = pd.DataFrame(cat_imputer2.transform(test[col_drop_obj]))



imputed_val.columns = test[col_drop_obj].columns
train[col_drop_obj].isnull().sum()
imputed_val[col_drop_obj].isnull().sum()
test = test.drop(columns=col_drop_obj, axis = 1)



imputed_val.index = test.index



test = pd.concat([test, imputed_val], axis = 1)
test.isnull().sum()
train.isnull().sum()
y = train['SalePrice']

box_cox = st.boxcox(y)

yeo = st.yeojohnson(y)

plt.figure(1); plt.title('box cox')

sns.distplot(box_cox[0])

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde = False, fit = st.norm)

plt.figure(3); plt.title('Log Transformation')

sns.distplot(y, kde = False, fit = st.lognorm)

plt.figure(4); plt.title('Johnson SU')

sns.distplot(y, kde = False, fit = st.johnsonsu)

plt.figure(5); plt.title('Yeo Johnson')

sns.distplot(yeo[0])

plt.show()
test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(train[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
f = pd.melt(train, value_vars=quantitative)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

g = g.map(sns.distplot, "value", kde_kws = {"bw" : 0.5})
for c in qualitative:

    train[c] = train[c].astype('category')

    if train[c].isnull().any():

        train[c] = train[c].cat.add_categories(['MISSING'])

        train[c] = train[c].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")
def anova(frame):

  anv = pd.DataFrame()

  anv['features'] = qualitative

  pvals = []

  for c in qualitative:

    samples = []

    for cls in frame[c].unique():

      s = frame[frame[c] == cls]['SalePrice'].values

      samples.append(s)

    pval = st.f_oneway(*samples)[1]

    pvals.append(pval)

  anv['pval'] = pvals

  return anv.sort_values('pval')
a = anova(train)

a.head(10)
a['disparity'] = np.log(1./a['pval'].values)

sns.barplot(data=a, x='features', y='disparity')

x=plt.xticks(rotation=90)
quant = [col for col in train.columns if col not in qualitative]

quant_train = train[quant]

quant_train.head()
corr_matrix = quant_train.corr()

print("\t  Heatmap Showing Correlation with SalePrice\n")

#saleprice correlation matrix

k = 11 #number of variables for heatmap

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(quant_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd"]

sns.pairplot(quant_train[cols], size = 2.5)

plt.show();
from category_encoders.one_hot import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
train.dtypes
# All categorical columns

object_cols = list(train.select_dtypes(include=['category']).columns)



# All Numerical Columns

num_cols = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]



print(num_cols)

print(object_cols)
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if train[col].nunique() < 7]



# Columns that will be label encoded

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be target encoded:', high_cardinality_cols)
train.isnull().sum()
test.isnull().sum()
y = train['SalePrice']

train = train.drop(columns=['SalePrice'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.1, random_state = 42)
X_train.isnull().sum()
X_test.isnull().sum()
y_train
# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', cols = low_cardinality_cols, use_cat_names = True)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_test))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_test.index

OH_cols_test.index = test.index
OH_cols_train.shape
OH_cols_valid.shape
X_test[X_test.index == 398]['Electrical']
OH_cols_valid['Electrical_SBrkr'].fillna(0, inplace=True)
OH_cols_valid['Electrical_FuseP'].fillna(0, inplace=True)
OH_cols_valid['Electrical_FuseF'].fillna(0, inplace = True)
OH_cols_valid['Electrical_FuseA'].fillna(0, inplace=True)
OH_cols_train.isnull().sum()
from category_encoders.target_encoder import TargetEncoder

tar_encoder = TargetEncoder(handle_unknown = 'value', cols = high_cardinality_cols)





num_X_train = pd.DataFrame(tar_encoder.fit_transform(OH_cols_train, y_train))

num_X_valid = pd.DataFrame(tar_encoder.transform(OH_cols_valid))

num_X_test = pd.DataFrame(tar_encoder.transform(OH_cols_test))



num_X_train.index = OH_cols_train.index

num_X_valid.index = OH_cols_valid.index

num_X_test.index = OH_cols_test.index
num_X_test.isnull().sum()
num_X_test.head()
y_train
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import mutual_info_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
# Creating a dummy RandomForestRegressor model

def train_model(X_train, X_valid, y_train, y_valid):

    bst = RandomForestRegressor()

    bst.fit(X_train, y_train)

    valid_score = bst.score(X_valid, y_valid)

    print(f"R-Square score: {valid_score:.4f}")

    return bst
import random
feature_cols = num_X_train.columns

np.random.seed(seed = 42) 

random.seed(42) # To always have reproducibe results

# Keep 182 features

selector = SelectKBest(mutual_info_regression, k=182)



X_new = selector.fit_transform(num_X_train, y_train)



# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=num_X_train.index, 

                                 columns=feature_cols)



# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns1 = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

num_X_valid[selected_columns1].head()
print(selector.pvalues_)
# Keep 10 features

selector = SelectKBest(mutual_info_regression, k=15)



X_new = selector.fit_transform(num_X_train, y_train)



# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=num_X_train.index, 

                                 columns=feature_cols)



# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns2 = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

num_X_valid[selected_columns2].head()
lasso = Lasso(alpha = 1, random_state=42, max_iter=3000)



model = SelectFromModel(lasso)



X_new = model.fit_transform(num_X_train, y_train)

X_new


# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=num_X_train.index,

                                 columns=num_X_train.columns)



# Dropped columns have values of all 0s, keep other columns 

selected_columns = selected_features.columns[selected_features.var() != 0]



num_X_valid[selected_columns].head()
num_X_valid.isnull().sum()
_ = train_model(num_X_train[selected_columns], num_X_valid[selected_columns], y_train, y_test) # Dataset having columns in list 'selected_columns'
_ = train_model(num_X_train[selected_columns1], num_X_valid[selected_columns1], y_train, y_test) # Dataset having columns in list 'selected_columns1'
_ = train_model(num_X_train[selected_columns2], num_X_valid[selected_columns2], y_train, y_test) # Dataset having columns in list 'selected_columns2'
selected_columns1
num_X_train[selected_columns1].isnull().sum()
model = train_model(num_X_train[selected_columns2], num_X_valid[selected_columns2], y_train, y_test)

y_pred = model.predict(num_X_test[selected_columns2])
sample = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
y_pred
sample['SalePrice'] = y_pred

sample.head()
sample.to_csv('Submission1.csv', index = False)