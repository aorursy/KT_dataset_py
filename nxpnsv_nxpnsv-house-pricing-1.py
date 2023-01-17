import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn import linear_model
from sklearn import preprocessing
plt.style.use(style='seaborn')
plt.rcParams['figure.figsize'] = (10, 6)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
def clean_df(df):
    ddf = df.drop(columns=["Id", "SalePrice"], errors="ignore")
    clean = pd.DataFrame()
    for c in df.select_dtypes(include=["object"]):
        cats = df[c].astype('category').cat.codes
        clean[c] = cats.fillna(-1)
    for c in df.select_dtypes(include=["int64", "float64"]):
        clean[c] = c.fillna(-1)
    return clean
clean_train = clean_df(train)
clean_test = clean_df(test)
# Get object and numeric types separately for cleaning
t = train.drop(columns=["Id", "SalePrice"])
obj_train = t.select_dtypes(include=["object"])
num_train = t.select_dtypes(include=["int64", "float64"])
t = test.drop(columns=["Id"])
obj_test = t.select_dtypes(include=["object"])
num_test = t.select_dtypes(include=["int64", "float64"])
# Clean train
clean_train = pd.DataFrame()
clean_test = pd.DataFrame()
# Add numeric columns with nan as -1
for c in num_train.columns:
    clean_train[c] = num_train[c].fillna(-1)
    clean_test[c] = num_test[c].fillna(-1)
# Categorize non-numeric columns, and set nan to -1
for c in obj_train.columns:
    cats = obj_train[c].astype('category').cat.codes
    clean_train[c] = cats.fillna(-1)
    cats = obj_test[c].astype('category').cat.codes
    clean_test[c] = cats.fillna(-1)
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(clean_train.values)
norm_train = pd.DataFrame(scaled, columns=clean_train.columns)
norm_test = pd.DataFrame(min_max_scaler.transform(clean_test.values), columns=clean_test.columns)
norm_train
fig, ax = plt.subplots(figsize=(20,20))
corr = norm_train.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax);
tt = norm_train.copy()
X, y = norm_train, train["SalePrice"]
reg = linear_model.RidgeCV(alphas=[1E-6, 0.001, 0.01, 0.1, 1.0, 10.0, 15., 20., 100., 1000.])
reg.fit(X,y)
plt.scatter(y, reg.predict(X))
submission = pd.DataFrame()
submission["Id"]=test["Id"]
submission["SalePrice"] = reg.predict(norm_test)
submission.to_csv("submission.csv", index = False)
print(submission)
