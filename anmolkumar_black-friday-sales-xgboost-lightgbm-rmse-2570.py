# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import itertools

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-darkgrid')

np.random.seed(22)



from scipy import stats

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
train_data = pd.read_csv('/kaggle/input/black-friday-sales/train.csv')

test_data = pd.read_csv('/kaggle/input/black-friday-sales/test.csv')

train_data.columns = train_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print('Train Data shape: ', train_data.shape)

train_data.head()
train_data.dtypes
train_data.isnull().sum()
train_data['type'] = 'train'

test_data['type'] = 'test'



master_data = pd.concat([train_data, test_data])

master_data.head()
plt.figure(1)

plt.subplot(121)

sns.distplot(master_data["purchase"]);



plt.subplot(122)

master_data["purchase"].plot.box(figsize = (16, 6))

plt.show()
plt.figure(figsize = (10, 5))

print(master_data["gender"].value_counts())

master_data['gender'].value_counts(normalize = True).plot.bar(title = 'Gender')
plt.figure(figsize = (10, 5))

print(master_data["age"].value_counts())

master_data['age'].value_counts(normalize = True).plot.bar(title = 'Age')
plt.figure(figsize = (10, 5))

print(master_data["stay_in_current_city_years"].value_counts())

master_data['stay_in_current_city_years'].value_counts(normalize = True).plot.bar(title = 'Stay in current city - Years')
plt.figure(figsize = (10, 5))

print(master_data["occupation"].value_counts())

master_data['occupation'].value_counts(normalize = True).plot.bar(title = 'Occupation')
plt.figure(figsize = (10, 5))

print(master_data["city_category"].value_counts())

master_data['city_category'].value_counts(normalize = True).plot.bar(title = 'City Category')
plt.figure(figsize = (10, 5))

print(master_data["marital_status"].value_counts())

master_data['marital_status'].value_counts(normalize = True).plot.bar(title = 'Martial Status')
plt.figure(figsize = (10, 5))

print(master_data["product_category_1"].value_counts())

master_data['product_category_1'].value_counts(normalize = True).plot.bar(title = 'Product Category 1')
plt.figure(figsize = (10, 5))

print(master_data["product_category_2"].value_counts())

master_data['product_category_2'].value_counts(normalize = True).plot.bar(title = 'Product Category 2')
plt.figure(figsize = (10, 5))

print(master_data["product_category_3"].value_counts())

master_data['product_category_3'].value_counts(normalize = True).plot.bar(title = 'Product Category 3')
plt.figure(figsize = (12, 6))

prod_by_cat = master_data.groupby('product_category_1')['product_id'].nunique()



sns.barplot(x = prod_by_cat.index,y = prod_by_cat.values)

plt.title('Number of Unique Items per Category')

plt.show()
plt.figure(figsize = (10, 5))

sns.violinplot(x = 'city_category', y = 'purchase', hue = 'marital_status', data = master_data)
plt.figure(figsize = (10, 5))

sns.countplot(master_data["gender"], hue = master_data["age"]).set_title("Age & Gender")

sns.despine()
plt.figure(figsize = (10, 5))

x = master_data.groupby(["gender"]).mean()[["purchase"]].index

y = master_data.groupby(["gender"]).mean()[["purchase"]].values

plt.plot(x, y,"ro")

plt.xticks(x, ["male", "female"])

plt.title("Mean purchase of different gender")

sns.despine()
plt.figure(figsize = (10, 5))

master_data.groupby("city_category")["purchase"].sum().plot.pie(title = "City Categry", 

                                                               startangle = 90, explode = (0.1, 0, 0), 

                                                               autopct = "%1.1f%%", shadow = True)
plt.figure(figsize = (10, 5))

x = master_data.groupby(["city_category"]).mean()[["purchase"]].index

y = master_data.groupby(["city_category"]).mean()[["purchase"]].values

plt.plot(x, y,"ro")

plt.title("Mean purchase of different city categories")

sns.despine()
plt.figure(figsize = (10, 5))

master_data["stay_in_current_city_years"].value_counts().plot.pie(title = "Years of staying in the city", 

                                                                 explode = (0.1, 0, 0, 0, 0), 

                                                                 autopct = "%1.1f%%", shadow = True)
master_data['product_category_2'] = master_data['product_category_2'].fillna(master_data['product_category_3'] - 1)

master_data['product_category_2'] = master_data['product_category_2'].fillna(master_data['product_category_1'] + 1)

master_data['product_category_3'] = master_data['product_category_3'].fillna(1 + master_data['product_category_2'])
user_prod = master_data[['user_id', 'product_category_1', 'product_category_2', 'product_category_3']].drop_duplicates()
user_prod_1_dim = pd.DataFrame(master_data.groupby(['user_id', 'product_category_1'])['purchase'].count()).reset_index()

user_purchases = pd.DataFrame(master_data.groupby('user_id')['purchase'].count()).reset_index()

user_prod_1_dim = user_prod_1_dim.merge(user_purchases, on = 'user_id', how = 'left')

user_prod_1_dim['user_prod_1'] = user_prod_1_dim['purchase_x']/user_prod_1_dim['purchase_y']

user_prod_1_dim = user_prod_1_dim.drop(['purchase_x', 'purchase_y'], axis = 1)



user_prod_2_dim = pd.DataFrame(master_data.groupby(['user_id', 'product_category_1', 'product_category_2'])['purchase'].count()).reset_index()

user_prod_2_dim = user_prod_2_dim.merge(user_purchases, on = 'user_id', how = 'left')

user_prod_2_dim['user_prod_2'] = user_prod_2_dim['purchase_x']/user_prod_2_dim['purchase_y']

user_prod_2_dim = user_prod_2_dim.drop(['purchase_x', 'purchase_y'], axis = 1)



user_prod_3_dim = pd.DataFrame(master_data.groupby(['user_id', 'product_category_1', 'product_category_2', 'product_category_3'])['purchase'].count()).reset_index()

user_prod_3_dim = user_prod_3_dim.merge(user_purchases, on = 'user_id', how = 'left')

user_prod_3_dim['user_prod_3'] = user_prod_3_dim['purchase_x']/user_prod_3_dim['purchase_y']

user_prod_3_dim = user_prod_3_dim.drop(['purchase_x', 'purchase_y'], axis = 1)



user_prod = user_prod.merge(user_prod_1_dim, on = ['user_id', 'product_category_1'], how = 'left')

user_prod = user_prod.merge(user_prod_2_dim, on = ['user_id', 'product_category_1', 'product_category_2'], how = 'left')

user_prod = user_prod.merge(user_prod_3_dim, on = ['user_id', 'product_category_1', 'product_category_2', 'product_category_3'], how = 'left')

user_prod.head()
master_data = master_data.merge(user_prod, on = ['user_id', 'product_category_1', 'product_category_2', 'product_category_3'], how = 'left')

master_data.head()
# individual groupby dataframes for each gender

gender_prod_m = master_data[master_data['gender'] == 'M'][['product_category_1','gender']].groupby('product_category_1').count()

gender_prod_f = master_data[master_data['gender'] == 'F'][['product_category_1','gender']].groupby('product_category_1').count()



gender_prod = pd.concat([gender_prod_m, gender_prod_f],axis = 1)

gender_prod.columns = ['m_ratio','f_ratio']



# Adjust to reflect ratios

gender_prod['m_ratio'] = gender_prod['m_ratio'] / master_data[master_data['gender'] == 'M'].count()[0]

gender_prod['f_ratio'] = gender_prod['f_ratio'] / master_data[master_data['gender'] == 'F'].count()[0]



# Create likelihood of one gender to buy over the other

gender_prod['likely_ratio'] = gender_prod['m_ratio'] / gender_prod['f_ratio']



gender_prod['total_ratio'] = gender_prod['m_ratio'] + gender_prod['f_ratio']

gender_prod = gender_prod.reset_index()

gender_prod.head()
master_data = master_data.merge(gender_prod, on = 'product_category_1', how = 'left')

master_data.head()
# Unique values for all the columns

for col in master_data.columns[~(master_data.columns.isin(['user_id', 'product_id', 'user_id', 'purchase', 'type',

                                                           'm_ratio', 'f_ratio', 'likely_ratio', 'total_ratio']))].tolist():

    print(" Unique Values --> " + col, ':', len(master_data[col].unique()), ': ', master_data[col].unique())
testProdIDs = master_data.loc[(master_data['type'] == 'test'), 'product_id']
for column in ['product_category_2', 'product_category_3']:

    master_data[column] = master_data[column].astype('int8')



train_data, test_data = master_data.loc[(master_data['type'] == 'train')], master_data.loc[(master_data['type'] == 'test')]

train_data = train_data.drop(['user_id', 'type'], axis = 1)
gender = {'F': 1, 'M': 2}

age = {'0-17': 1, '55+': 2, '26-35': 7, '46-50': 4, '51-55': 3, '36-45': 6, '18-25': 5}

city_category = {'A':1,  'C':2, 'B': 3}

stay_in_current_city_years = {'2': 4, '4+': 2, '3': 3, '1': 5, '0': 1}



train_data['gender'] = train_data['gender'].map(gender)

test_data['gender'] = test_data['gender'].map(gender)



train_data['age'] = train_data['age'].map(age)

test_data['age'] = test_data['age'].map(age)



train_data['city_category'] = train_data['city_category'].map(city_category)

test_data['city_category'] = test_data['city_category'].map(city_category)



train_data['stay_in_current_city_years'] = train_data['stay_in_current_city_years'].map(stay_in_current_city_years)

test_data['stay_in_current_city_years'] = test_data['stay_in_current_city_years'].map(stay_in_current_city_years)



testRes = test_data[['user_id']]

testRes['Product_ID'] = testProdIDs

test_data = test_data.drop(['user_id', 'type', 'purchase'], axis = 1)



train_data.head()
# Label Encoding Product_IDs

new_product_ids = list(set(pd.unique(test_data['product_id'])) - set(pd.unique(train_data['product_id'])))



le = LabelEncoder()

train_data['product_id'] = le.fit_transform(train_data['product_id'])

test_data.loc[test_data['product_id'].isin(new_product_ids), 'product_id'] = -1

new_product_ids.append(-1)



test_data.loc[~test_data['product_id'].isin(new_product_ids), 'product_id'] = le.transform(test_data.loc[~test_data['product_id'].isin(new_product_ids), 'product_id'])



print(test_data.shape)
X = train_data[train_data.columns[~(train_data.columns.isin(['purchase']))].tolist()]



ss = StandardScaler()



X = ss.fit_transform(X)



y = train_data['purchase'].values



testData = test_data.copy()



test_data = ss.fit_transform(test_data)
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = XGBRegressor(random_state = 22, max_depth = 10, n_estimators = 600, objective = 'reg:squarederror')

    model.fit(X_train, y_train)#, cat_features = [0,1,5,7,8,9,10,11,12])

    preds = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, preds))

    scores.append(score)

    print('Validation RMSE:', score)

print("Average Validation RMSE: ", sum(scores)/len(scores))
yPreds = model.predict(test_data)

testRes['Purchase'] = yPreds

submission = testRes[['user_id', 'Product_ID', 'Purchase']]



submission.columns = ['User_ID', 'Product_ID', 'Purchase']

submission.to_csv('submission_XGB_v10.csv', index = False)

submission.head()
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = XGBRegressor(random_state = 22, max_depth = 10, n_estimators = 600, 

                         objective = 'reg:squarederror', booster = 'gbtree', eta = 0.1, nthread = 4,

                         subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 20,

                         max_delta_step = 0, gamma = 0)

    model.fit(X_train, y_train)#, cat_features = [0,1,5,7,8,9,10,11,12])

    preds = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, preds))

    scores.append(score)

    print('Validation RMSE:', score)

print("Average Validation RMSE: ", sum(scores)/len(scores))
yPreds = model.predict(test_data)

testRes['Purchase'] = yPreds

submission = testRes[['user_id', 'Product_ID', 'Purchase']]



submission.columns = ['User_ID', 'Product_ID', 'Purchase']

submission.to_csv('submission_XGB_v11.csv', index = False)

submission.head()
train_data = train_data.drop(['product_category_2', 'product_category_3'], axis = 1)

test_data = testData.drop(['product_category_2', 'product_category_3'], axis = 1)



X = train_data[train_data.columns[~(train_data.columns.isin(['purchase']))].tolist()]



ss = StandardScaler()



X = ss.fit_transform(X)



y = train_data['purchase'].values



test_data = ss.fit_transform(test_data)
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = XGBRegressor(random_state = 22, max_depth = 10, n_estimators = 600, 

                         objective = 'reg:squarederror', booster = 'gbtree', eta = 0.1, nthread = 4,

                         subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 20,

                         max_delta_step = 0, gamma = 0)

    model.fit(X_train, y_train)#, cat_features = [0,1,5,7,8,9,10,11,12])

    preds = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, preds))

    scores.append(score)

    print('Validation RMSE:', score)

print("Average Validation RMSE: ", sum(scores)/len(scores))
yPreds = model.predict(test_data)

testRes['Purchase'] = yPreds

submission = testRes[['user_id', 'Product_ID', 'Purchase']]



submission.columns = ['User_ID', 'Product_ID', 'Purchase']

submission.to_csv('submission_XGB_v12.csv', index = False)

submission.head()
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]

    

    model = LGBMRegressor(random_state = 22, max_depth = 10, n_estimators = 600)

    model.fit(X_train, y_train)#, cat_features = [0,1,5,7,8,9,10,11,12])

    preds = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, preds))

    scores.append(score)

    print('Validation RMSE:', score)

print("Average Validation RMSE: ", sum(scores)/len(scores))
yPreds = model.predict(test_data)

testRes['Purchase'] = yPreds

submission = testRes[['user_id', 'Product_ID', 'Purchase']]



submission.columns = ['User_ID', 'Product_ID', 'Purchase']

submission.to_csv('submission_LGB_v10.csv', index = False)

submission.head()