from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

from sklearn.preprocessing import StandardScaler

from scipy import stats

import os, sys


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pd.options.display.max_columns = 500

dataset = pd.read_csv("/kaggle/input/ameshousingdataset/AmesHousing.tsv", delimiter="\t")

dataset.shape
dataset.head()
missing_data = dataset.isnull().sum()

dropable_columns = missing_data[(missing_data> len(dataset)/20)].sort_values()

dropable_columns.index
dataset = dataset.drop(dropable_columns.index, axis=1)

dataset.shape
text_type_data = dataset.select_dtypes(include=['object']).isnull().sum()

dropable_text_columns = text_type_data[text_type_data>0]

dropable_text_columns
dataset = dataset.drop(dropable_text_columns.index, axis=1)

dataset.shape
num_missing_colmuns = dataset.select_dtypes(include=['int','float']).isnull().sum()

dropable_num_columns = num_missing_colmuns[num_missing_colmuns > 0]

dropable_num_columns
flexible_columns = num_missing_colmuns[(num_missing_colmuns < len(dataset)/20) & (num_missing_colmuns > 0)]

flexible_columns
replace_ment_dict = dataset[flexible_columns.index].mode().to_dict(orient='records')[0]

replace_ment_dict
dataset = dataset.fillna(replace_ment_dict)

dataset.shape
dataset.isnull().sum().value_counts()
years_sold = dataset['Yr Sold'] - dataset['Year Built']

years_sold[years_sold < 0]
years_since_remod = dataset['Yr Sold'] - dataset['Year Remod/Add']

years_since_remod[years_since_remod < 0]
dataset['Years Before Sale'] = years_sold

dataset['Years Since Remod'] = years_since_remod



dataset = dataset.drop([1702, 2180, 2181], axis=0)

dataset = dataset.drop(["Year Built", "Year Remod/Add"], axis = 1)

dataset = dataset.drop(["PID", "Order","Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
plt.figure(figsize=(10,8))

sns.distplot(dataset['SalePrice'])

plt.title("Sale price Frequency Graph")

plt.show()

# SalePrice is our target column
fig_per_time = 3

count=0

train_data = dataset[0:1460]

test_data = dataset[1460:]



for column in dataset.columns:

    plt.figure(count//fig_per_time,figsize=(25,5))

    plt.subplot(1, fig_per_time, np.mod(count,3)+1)

    plt.scatter(train_data[column],train_data['SalePrice'])

    plt.title("Model: {0}".format(column))

    count +=1 



# plt.scatter(train_data['MS SubClass'],train_data['SalePrice'])
numeric_data = dataset.select_dtypes(include=['int','float'])

numeric_data.head()
asb_corr_coff = numeric_data.corr()['SalePrice'].abs().sort_values()

asb_corr_coff
corrdata = numeric_data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(corrdata, vmax=1, square=True)

plt.title("Heatmap for data correlation")

plt.show()
transform_dataset = asb_corr_coff[asb_corr_coff >= 0.3]

transform_dataset
nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 

                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 

                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 

                    "Misc Feature", "Sale Type", "Sale Condition"]



transform_cat_cols = []

for col in nominal_features:

    if col in dataset.columns:

        transform_cat_cols.append(col)

unique_values = dataset[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()

unique_values
drop_nonuniq_cols = unique_values[unique_values>10].index

dataset = dataset.drop(drop_nonuniq_cols, axis=1)

dataset.shape
dataset.head()
text_cols = dataset.select_dtypes(include=['object'])

for col in text_cols:

    dataset[col] = dataset[col].astype('category')



text_cols.head()
categorical_features = text_cols.columns

feat_cat = dataset[categorical_features]



for feat in feat_cat:

    fig = plt.figure(figsize=(8,6))

    fig = sns.boxplot(x=feat, y='SalePrice', data=dataset)

    plt.show()
new_dataset = pd.concat([dataset, pd.get_dummies(dataset.select_dtypes(include=['category']))], axis=1)

new_dataset = new_dataset.drop(text_cols, axis=1)

new_dataset.head()
train_dataset = new_dataset[0:1460]

test_dataset = new_dataset[1460:]

numerical_data = new_dataset.select_dtypes(include=('int','float'))

features = numerical_data.columns.drop("SalePrice")
train_corrdata = train_dataset.select_dtypes(include=('int','float')).corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(train_corrdata, vmax=1, square=True)

plt.title("Heatmap for data train_data correlation")

plt.show()
corr_num = 15 #number of variables for heatmap

cols_corr = train_corrdata.nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(train_dataset[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.title("Heatmap for data train_data correlation")

plt.show()
lr = linear_model.LinearRegression()
kFolds = KFold(n_splits=5, shuffle=True)

rmse_values = []



for train_index, test_index in kFolds.split(new_dataset):

    train_data = new_dataset.iloc[train_index]

    test_data = new_dataset.iloc[test_index]

    

    lr.fit(train_data[features], train_data['SalePrice'])

    prediction = lr.predict(test_data[features])

    

    mse = mean_squared_error(prediction, test_data['SalePrice'])

    rmse = np.sqrt(mse)

    rmse_values.append(rmse)
print(rmse_values)

print("Avearge rmse mean: {0}".format(np.mean(rmse_values)))