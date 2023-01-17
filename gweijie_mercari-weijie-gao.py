
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb

import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


%matplotlib inline
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
train.head()
test.head()
train.info()
train.corr()['price']
train['price'].describe()
train['price'].plot.hist(title="Histogram of item prices in Dataset")
train.shape
test.shape
len(train['category_name'].unique())
train['category_name'].value_counts()
train['brand_name'].describe()
train['brand_name'].value_counts()
train['brand_name'] = train['brand_name'].fillna("missing")
train['brand_name'].value_counts()
train['has_brand'] = np.where(train['brand_name'] == 'missing', 0, 1)
train.head()
train.corr()
len(train['brand_name'].unique())
train.groupby('shipping').mean()
train.groupby('shipping').mean()['price'].plot.bar(title="Average price of item vs. shipping")
train['item_description'].head()
train['item_description'].str.len().plot.hist(title="Histogram of length of item descriptions")
train['item_description'].str.len().describe()
train['name'].describe()
# checks null values, 
# a few in item_description
# a lot in brand_name 
# 6k + in category_name
train.isnull().sum()
train.groupby('item_condition_id').count()['price'].plot.bar(title="Distribution of items by item condition")
train['first_category_name'] = train['category_name'].str.split('/').str.get(0)
train['second_category_name'] = train['category_name'].str.split('/').str.get(1)
train['third_category_name'] = train['category_name'].str.split('/').str.get(2)
train.head()
train = train.drop('category_name', axis=1)
train.head()
train['first_category_name'] = train['first_category_name'].fillna('missing')
train['second_category_name'] = train['second_category_name'].fillna('missing')
train['third_category_name'] = train['third_category_name'].fillna('missing')
train['item_description'] = train['item_description'].fillna('missing')
train.isnull().sum()
train.head()
y = np.log1p(train["price"])

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(train["name"])

print("Category Encoders")
count_category_one = CountVectorizer()
X_category_one = count_category_one.fit_transform(train["first_category_name"])

count_category_two = CountVectorizer()
X_category_two = count_category_two.fit_transform(train["second_category_name"])

count_category_three = CountVectorizer()
X_category_three = count_category_three.fit_transform(train["third_category_name"])

print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(train["item_description"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(train["brand_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(train[[
    "item_condition_id", "shipping"]], sparse = True).values)
X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category_one,
                         X_category_two,
                         X_category_three,
                         X_name)).tocsr()
X_train, X_test, y_val_train, y_val_test = train_test_split( X, y, test_size=0.3, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train,y_val_train)
print(model)
from sklearn.metrics import mean_squared_error
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_val_test, predictions))
print(rmse)

xgb.plot_importance(model)
"""
to try gridsearching xgboost with gridsearchcv:

from sklearn.model_selection import GridSearchCV

model = xgb.XGBRegressor()

parameters = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.1, 0.15, 0.],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
    }

clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_val_train)
print(clf)
"""








