# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from collections import Counter

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.tsv", delimiter="\t")
df["name"] = df["name"].str.lower()

df["category_name"] = df["category_name"].str.lower()

df["brand_name"] = df["brand_name"].str.lower()

df["item_description"] = df["item_description"].str.lower()
df["name"] = df["name"].str.replace(',', '')

df["name"] = df["name"].str.replace('.', '')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = df.select_dtypes(include=[object])

X.head(3)
X.isnull().sum()
X["category_name"] = X["category_name"].fillna("-1")

X["brand_name"] = X["brand_name"].fillna("-1")

X = X.dropna()
df_enc = X.apply(le.fit_transform)



df_enc.head(3)
new_df = df
new_df["name"] = df_enc["name"]

new_df["category_name"] = df_enc["category_name"]

new_df["brand_name"] = df_enc["brand_name"]

new_df["item_description"] = df_enc["item_description"]
new_df.isnull().sum()
new_df = new_df.dropna()

new_df.isnull().sum()
price = new_df["price"].values
new_df = new_df.drop("price", axis=1)
X_train, X_test, y_train, y_test = train_test_split(new_df.values, price)
from sklearn import linear_model
clf = linear_model.Lasso()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)

pre
def rmsle(pre, act):

    assert len(pre) == len(y_test), "different size of the array {0},{1}".format(len(pre),len(y_test))

    return np.sqrt(np.mean((np.log(pre+1)-np.log(act+1))**2.0))
rmsle(pre, y_test)
from sklearn.model_selection import GridSearchCV
params = {

    "alpha":np.arange(0.1,1,0.05),

    "normalize": [True, False]

}
grid_search = GridSearchCV(clf, param_grid=params)
start = time.time()

grid_search.fit(X_train, y_train)

print("time", time.time()-start)
pred = grid_search.predict(X_test)
rmsle(pred, y_test)
test_df = pd.read_csv("../input/test.tsv", delimiter="\t")

test_df.head(3)
test_df["name"] = test_df["name"].str.lower()

test_df["category_name"] = test_df["category_name"].str.lower()

test_df["brand_name"] = test_df["brand_name"].str.lower()

test_df["item_description"] = test_df["item_description"].str.lower()
test_df["name"] = test_df["name"].str.replace(',', '')

test_df["name"] = test_df["name"].str.replace('.', '')
test_df.isnull().sum()
test_X = test_df.select_dtypes(include=[object])

test_X.head(3)
test_X["category_name"] = test_X["category_name"].fillna("-1")

test_X["brand_name"] = test_X["brand_name"].fillna("-1")

test_X = test_X.dropna()
test_df_enc = test_X.apply(le.fit_transform)



test_df_enc.head(3)
test_df["name"] = test_df_enc["name"]

test_df["category_name"] = test_df_enc["category_name"]

test_df["brand_name"] = test_df_enc["brand_name"]

test_df["item_description"] = test_df_enc["item_description"]
test_df.head()
test_pre = grid_search.predict(test_df)
test_pre
test = pd.DataFrame()
test["test_id"] = np.arange(len(test_pre))
test["price"] = test_pre
test.to_csv("sub1.csv",index=False)