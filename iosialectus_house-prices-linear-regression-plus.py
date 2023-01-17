# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from utility_scripts import estimate_entropy, estimate_mutual_info, estimate_mutual_info_multiple_columns

from sklearn.tree import DecisionTreeRegressor

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

print(df_train.head())
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(df_test.head())
testing_ids = df_test['Id']
numerical_columns = [col for col in df_train.describe().columns if len(df_train[col].unique())>25 and not col=="SalePrice"][1:]

for col in numerical_columns:

    print (col)
categorical_columns = [col for col in df_train.columns if not col in numerical_columns and not col=="SalePrice"][1:]

for col in categorical_columns:

    print (col)
df_train['logSalePrice'] = np.log(df_train['SalePrice'])
correls = df_train[numerical_columns + ['logSalePrice']].corr().sort_values(by=['logSalePrice'],ascending=False)

correls['logSalePrice']
df_train[numerical_columns].describe()
has_na = []

for col in numerical_columns:

    num_na_train = df_train[col].isna().sum()

    num_na_test = df_test[col].isna().sum()

    if num_na_train+num_na_test>0:

        has_na.append(col)

        print("{}: {}, {}".format(col, num_na_train, num_na_test))
for col in has_na:

    default = df_train[col].median()

    df_train[col] = df_train[col].fillna(default)

    df_test[col] = df_test[col].fillna(default)

    



# If the above worked should print nothing:

for col in numerical_columns:

    num_na_train = df_train[col].isna().sum()

    num_na_test = df_test[col].isna().sum()

    if num_na_train+num_na_test>0:

        print("{}: {}, {}".format(col, num_na_train, num_na_test))
numerical_columns_normed = [col+"_normed" for col in numerical_columns]

for col in numerical_columns:

    mean = df_train[col].mean()

    stdev = df_train[col].std()

    df_train[col+"_normed"] = df_train[col].apply(lambda x: (x-mean)/stdev)

    df_test[col+"_normed"] = df_test[col].apply(lambda x: (x-mean)/stdev)

    

df_train[numerical_columns_normed].describe()
pca = PCA(n_components=len(numerical_columns_normed))

pca.fit(df_train[numerical_columns_normed])
print(pca.explained_variance_ratio_)
pca_columns = ["pca_{}".format(i) for i in range(len(numerical_columns_normed))]

df_train = pd.concat([df_train, pd.DataFrame(data = pca.transform(df_train[numerical_columns_normed]),columns=pca_columns)],axis=1)

df_test = pd.concat([df_test, pd.DataFrame(data = pca.transform(df_test[numerical_columns_normed]),columns=pca_columns)],axis=1)
correls2 = df_train[pca_columns + ['logSalePrice']].corr().sort_values(by=['logSalePrice'],ascending=False)

correls2['logSalePrice']
correls2
important_principle_components = [pc for pc in pca_columns if abs(correls2.loc[pc]['logSalePrice'])>0.1 ]

print(important_principle_components)
df_train[important_principle_components + ['logSalePrice']].corr()
for c in important_principle_components:

    pyplot.scatter(df_train['logSalePrice'], df_train[c])

    pyplot.show()
df_train_filtered = df_train[df_train['pca_0']<=11][df_train['logSalePrice']>10.7]

pyplot.scatter(df_train_filtered['logSalePrice'], df_train_filtered['pca_0'])
target_filtered = df_train_filtered['logSalePrice']

target = df_train['logSalePrice']
training_data_linear_regression = df_train_filtered[important_principle_components]

testing_data_linear_regression = df_test[important_principle_components]
model = LinearRegression()

model.fit(training_data_linear_regression, target_filtered)
predictions_trainingset_lr = model.predict(df_train[important_principle_components])

predictions_lr = model.predict(testing_data_linear_regression)
pyplot.figure()

pyplot.scatter(df_train['logSalePrice'], predictions_trainingset_lr, alpha=0.5)

line_pts = np.linspace(10,15,1000)

pyplot.plot(line_pts,line_pts,color='r')

pyplot.xlim(10,15)

pyplot.ylim(10,15)

pyplot.show()
df_submission_lr = pd.DataFrame()

df_submission_lr['Id'] = df_test['Id']

df_submission_lr['SalePrice'] = np.exp(predictions_lr)

df_submission_lr.to_csv('submission_linear_regression.csv', index=False)
residue = df_train['logSalePrice'] - predictions_trainingset_lr

df_train['Residue'] = residue
residue.describe()
pyplot.scatter(df_train['logSalePrice'], residue)
df_train['SalePriceQuantiles20'] = pd.qcut(df_train['SalePrice'],20,range(20))

print(df_train[['Residue','SalePriceQuantiles20']].head(20))

print(df_train[['Residue','SalePriceQuantiles20']].tail(20))
mutual_infos = [(col, estimate_mutual_info(df_train, col, 'SalePriceQuantiles20')) for col in categorical_columns]

mutual_infos = sorted(mutual_infos, key=lambda x: -x[1])

for pair in mutual_infos:

    print(pair)
important_cat_cols = [pair[0] for pair in mutual_infos if pair[1]>0.1]

for col in important_cat_cols:

    print(col)
'''important_cat_cols2 = important_cat_cols[:1]

for i in range(5):

    this = important_cat_cols[-1]

    this_mi = estimate_mutual_info_multiple_columns(df_train, ['SalePriceQuantiles20'], important_cat_cols2 + [this])

    for col in important_cat_cols:

        if not col in important_cat_cols2:

            mi = estimate_mutual_info_multiple_columns(df_train, ['SalePriceQuantiles20'], important_cat_cols2 + [col])

            if mi > this_mi:

                this_mi = mi

                this = col

    important_cat_cols2.append(this)

    print("{}: {}".format(this, this_mi))'''
#df_train[important_cat_cols2].head(15)
needs_dummies = [c for c in important_cat_cols if not c in list(df_train[important_cat_cols].describe().columns)]

print(needs_dummies)
train_dummies = pd.get_dummies(df_train[needs_dummies])

test_dummies = pd.get_dummies(df_test[needs_dummies])
drop_list = [c for c in list(train_dummies.columns) if not c in list(test_dummies.columns)]
drop_list.extend([c for c in list(test_dummies.columns) if not c in list(train_dummies.columns)])
dummy_cols = [c for c in list(test_dummies.columns) if not c in drop_list]

print(dummy_cols)
cat_cols_to_use = dummy_cols + [c for c in important_cat_cols if not c in needs_dummies]

print(cat_cols_to_use)
df_train = pd.concat([df_train, train_dummies], axis=1)

df_test = pd.concat([df_test, train_dummies], axis=1)
has_na = []

for col in cat_cols_to_use:

    num_na_train = df_train[col].isna().astype('int32').sum()

    num_na_test = df_test[col].isna().astype('int32').sum()

    if num_na_train+num_na_test>0:

        has_na.append(col)

        print("{}: {}, {}".format(col, num_na_train, num_na_test))
for col in has_na:

    default = df_train[col].mode()[0]

    df_train[col] = df_train[col].fillna(default)

    df_test[col] = df_test[col].fillna(default)

    



# If the above worked should print nothing:

for col in cat_cols_to_use:

    num_na_train = df_train[col].isna().astype('int32').sum()

    num_na_test = df_test[col].isna().astype('int32').sum()

    if num_na_train+num_na_test>0:

        print("{}: {}, {}".format(col, num_na_train, num_na_test))
df_test.tail()
training2 = df_train[cat_cols_to_use + important_principle_components]

testing2 = df_test.iloc[:-1][cat_cols_to_use + important_principle_components]
#model2 = RandomForestRegressor(n_estimators=50, min_samples_split=10, max_samples=500)

model2 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=50, subsample=0.3, min_samples_split=50)

model2.fit(training2, residue)

predictions2 = model2.predict(testing2)

predictions2_training = model2.predict(training2)
pyplot.figure()

pyplot.scatter(residue, predictions2_training, alpha=0.5)

line_pts = np.linspace(-3,3,1000)

pyplot.plot(line_pts,line_pts,color='r')

pyplot.show()
df_submission = pd.DataFrame()

df_submission['Id'] = testing_ids

df_submission['SalePrice'] = np.exp(predictions2 + predictions_lr)

df_submission.to_csv('submission.csv', index=False)
pyplot.figure()

pyplot.scatter(df_train['logSalePrice'], predictions2_training + predictions_trainingset_lr, alpha=0.5)

line_pts = np.linspace(10,15,1000)

pyplot.plot(line_pts,line_pts,color='r')

pyplot.show()
training3 = df_train[cat_cols_to_use + important_principle_components]

testing3 = df_test[cat_cols_to_use + important_principle_components]
model3 = SVR()

model3.fit(training3, target)