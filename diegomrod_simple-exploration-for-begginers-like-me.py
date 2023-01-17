import numpy as np # vector manipulation
import pandas as pd # dataframe manipulation

from sklearn.model_selection import train_test_split # spliting train and test dataset
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.tree import DecisionTreeRegressor # decision tree
from sklearn.ensemble import RandomForestRegressor # random forest

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt # plotting

%matplotlib inline

import warnings # just for ignoring annoying warnings
warnings.filterwarnings('ignore')
TARGET = 'SalePrice'
dataset = pd.read_csv('../input/train.csv')
dataset.head()
dataset.shape
desc = dataset.describe()
desc
count = desc.iloc[0]
count = count[count == dataset.shape[0]]
num_columns = count.index.values

num_columns
for column in num_columns:
    plt.title(column)
    plt.scatter(dataset[column], dataset[TARGET])
    plt.show()
selected = [
    'GarageArea',       # Garage
    'GarageCars', 
    'TotRmsAbvGrd',     # Above Ground
    'LotArea',          # Area
    'OverallQual',      # Quality
    'OverallCond',
    'TotalBsmtSF',      # Second Floor
    '1stFlrSF', 
    '2ndFlrSF',
    'YearBuilt',        # Year
    'YearRemodAdd'
]
corr = dataset[selected + [TARGET]].corr()
corr
corr_saleprice = corr['SalePrice']
corr_saleprice
corr_saleprice[corr_saleprice > 0.60]
linear_features = [
    'GarageArea',
    'GarageCars',
    'OverallQual',
    'TotalBsmtSF',
    '1stFlrSF'
]
linear_dataset = dataset[linear_features]
y = dataset[TARGET]
X_train, X_test, y_train, y_test = train_test_split(linear_dataset, y, test_size=0.30, random_state=42)
model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)
cat_dataset = dataset.drop(num_columns, axis=1)
cat_dataset.head()
count = cat_dataset.isna().sum()
count = count[count > 0]
count
cat_dataset = cat_dataset.drop(count.index.values, axis=1)
cat_dataset.head()
for col in cat_dataset.columns:
    print(cat_dataset[col].value_counts())
    print("\n")
cat_features = [
    'MSZoning',
    'LandContour',
    'LotShape',
    'LotConfig',
    'HouseStyle',
    'Foundation',
    'HeatingQC',
    'ExterQual',
    'KitchenQual',
    'SaleCondition'
]
sel_dataset = cat_dataset[cat_features]
sel_dataset.head()
# MSZoning : RL or not RL
sel_dataset['MSZoning'] = np.where(sel_dataset['MSZoning'] == 'RL', 1, 0)
sel_dataset.head()
sel_dataset['LandContour'] = np.where(sel_dataset['LandContour'] == 'Lvl', 1, 0)
sel_dataset.head()
sel_dataset['LotShape'] = np.where(sel_dataset['LotShape'] == 'Reg', 1, 0)
sel_dataset.head()
conditions = [
    sel_dataset['LotConfig'] == 'Inside',
    sel_dataset['LotConfig'] == 'Corner'
]

choices = [2, 1]

sel_dataset['LotConfig'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['HouseStyle'] == '2Story') | (sel_dataset['HouseStyle'] == '1Story') 
]

choices = [1]

sel_dataset['HouseStyle'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['Foundation'] == 'PConc'), 
    (sel_dataset['Foundation'] == 'CBlock') 
]

choices = [2, 1]

sel_dataset['Foundation'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['HeatingQC'] == 'Ex'), 
    (sel_dataset['HeatingQC'] == 'TA'),
    (sel_dataset['HeatingQC'] == 'Gd')
]

choices = [3, 2, 1]

sel_dataset['HeatingQC'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['ExterQual'] == 'TA'), 
    (sel_dataset['ExterQual'] == 'Gd')
]

choices = [2, 1]

sel_dataset['ExterQual'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['KitchenQual'] == 'TA'), 
    (sel_dataset['KitchenQual'] == 'Gd')
]

choices = [2, 1]

sel_dataset['KitchenQual'] = np.select(conditions, choices, default=0)
sel_dataset.head()
conditions = [
    (sel_dataset['SaleCondition'] == 'Normal')
]

choices = [1]

sel_dataset['SaleCondition'] = np.select(conditions, choices, default=0)
sel_dataset.head()
y = dataset[TARGET]
X_train, X_test, y_train, y_test = train_test_split(sel_dataset, y, test_size=0.30, random_state=42)
model = DecisionTreeRegressor().fit(X_train, y_train)
model.score(X_test, y_test)
both_dataset = pd.concat([linear_dataset, sel_dataset], axis=1, sort=False)
both_dataset.head()
X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)
one_hot = OneHotEncoder().fit(sel_dataset)
oh_dataset = one_hot.transform(sel_dataset)
oh_dataset = pd.DataFrame(oh_dataset.toarray())
both_dataset = pd.concat([linear_dataset, oh_dataset], axis=1, sort=False)
X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)
scaler = StandardScaler()
scaler.fit(linear_dataset)
scaled_dataset = scaler.transform(linear_dataset)
scaled_dataset = pd.DataFrame(scaled_dataset)
both_dataset = pd.concat([scaled_dataset, oh_dataset], axis=1, sort=False)
X_train, X_test, y_train, y_test = train_test_split(both_dataset, y, test_size=0.30, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)
y_log = np.log(y)
X_train, X_test, y_train, y_test = train_test_split(both_dataset, y_log, test_size=0.30, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)
model.score(X_test, y_test)
model.get_params()
params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}
f_model = RandomizedSearchCV(model, params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
f_model.fit(X_train, y_train)
f_model.score(X_test, y_test)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(f_model, file=f)
dataset = pd.read_csv('../input/test.csv')
dataset.head()
X_linear = dataset[linear_features] 
X_cat = dataset[cat_features]
X_cat['MSZoning'] = np.where(X_cat['MSZoning'] == 'RL', 1, 0)
X_cat['LandContour'] = np.where(X_cat['LandContour'] == 'Lvl', 1, 0)
X_cat['LotShape'] = np.where(X_cat['LotShape'] == 'Reg', 1, 0)

conditions = [
    X_cat['LotConfig'] == 'Inside',
    X_cat['LotConfig'] == 'Corner'
]

choices = [2, 1]

X_cat['LotConfig'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['HouseStyle'] == '2Story') | (X_cat['HouseStyle'] == '1Story') 
]

choices = [1]

X_cat['HouseStyle'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['Foundation'] == 'PConc'), 
    (X_cat['Foundation'] == 'CBlock') 
]

choices = [2, 1]

X_cat['Foundation'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['HeatingQC'] == 'Ex'), 
    (X_cat['HeatingQC'] == 'TA'),
    (X_cat['HeatingQC'] == 'Gd')
]

choices = [3, 2, 1]

X_cat['HeatingQC'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['ExterQual'] == 'TA'), 
    (X_cat['ExterQual'] == 'Gd')
]

choices = [2, 1]

X_cat['ExterQual'] = np.select(conditions, choices, default=0)


conditions = [
    (X_cat['KitchenQual'] == 'TA'), 
    (X_cat['KitchenQual'] == 'Gd')
]

choices = [2, 1]

X_cat['KitchenQual'] = np.select(conditions, choices, default=0)

conditions = [
    (X_cat['SaleCondition'] == 'Normal')
]

choices = [1]

X_cat['SaleCondition'] = np.select(conditions, choices, default=0)
X_cat.head()
X_cat = one_hot.transform(X_cat)
X_cat = pd.DataFrame(X_cat.toarray())
X_linear = scaler.transform(X_linear)
X_linear = pd.DataFrame(X_linear)
X = pd.concat([X_linear, X_cat], axis=1, sort=False)
X.head()
X = X.fillna(0)
y_final = f_model.predict(X)
y_final = np.exp(y_final)
y_final = pd.DataFrame({'SalePrice': y_final})
y_final.head()
y.head()
Ids = dataset['Id']
Ids.head()
submission = pd.concat([Ids, y_final], axis=1, sort=False)
submission.head()
submission.to_csv('./submission.csv', index=False)
