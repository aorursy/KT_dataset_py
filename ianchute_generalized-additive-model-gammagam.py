!pip install pygam



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from pygam import GammaGAM, s, f, l

# from sklearn.linear_model import Ridge

from tqdm.notebook import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



missing_values = {

    "MSSubClass": {"150": "160"},

    "FullBath": {"4": "3"},

    "TotRmsAbvGrd": {"13": "14", "15": "14"},

    "Fireplaces": {"4": "3"},

    "GarageCars": {"5": "4"},

}



def geom_mean(x):

    return np.expm1(np.log1p(x).mean())



train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

X = train.drop(["SalePrice", "Id"], axis=1)

y = train["SalePrice"]



test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

X_test = test.drop("Id", axis=1)

ids_test = test["Id"]



cols_cat = X.select_dtypes("object").columns.tolist() + X.select_dtypes(["int", "float"]).nunique().pipe(lambda x: x[x <= 20].keys()).tolist()

cols_cat = [col for col in cols_cat if col not in {"3SsnPorch", "PoolArea"}]

X[cols_cat] = X[cols_cat].fillna("UNKNOWN").astype(str)

modes = X[cols_cat].mode().iloc[0]

for col in cols_cat:

    if "UNKNOWN" in X[col].unique():

        X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)

    else:

        X_test[col] = X_test[col].fillna(modes[col]).astype(str)

    if train[col].dtype != "object":

        X_test[col] = X_test[col].apply(lambda s: s.split(".")[0])

    diff = set(X_test[col].unique()).difference(set(X[col].unique()))

    if len(diff) > 0:

        print(col)

        for d in diff:

            print("\t", d, "->", missing_values[col][d], f"({(X_test[col] == d).mean()*100:.2f}%)")

            X_test.loc[X_test[col] == d, col] = missing_values[col][d]



encoders = {}

for col in cols_cat:

    encoders[col] = LabelEncoder().fit(X[col]) # pd.concat([X[col], y], axis=1).groupby(col)["SalePrice"].agg(geom_mean).to_dict()

    X[col] = encoders[col].transform(X[col]) # X[col].astype(str).map(encoders[col]) # .transform(X[col])

    X_test[col] = encoders[col].transform(X_test[col]) # X_test[col].astype(str).map(encoders[col])

        

cols_num = [col for col in X.columns if col not in cols_cat]

means = X[cols_num].agg(geom_mean)

X[cols_num] = X[cols_num].fillna(means)

X_test[cols_num] = X_test[cols_num].fillna(means)



def to_term(col):

    i = X.columns.tolist().index(col)

    if col in cols_num:

        return s(i, constraints="monotonic_inc") # assumption: more is generally better

    if col in cols_cat:

        return f(i)

    else:

        raise Exception(f"Unknown columns {col}")

        

terms = to_term(X.columns[0])

for col in X.columns[1:]:

    terms += to_term(col)

    

train_preds = []

test_preds = []



for i in tqdm(range(len(X))):

    try:

        model = GammaGAM(terms).fit(X.drop(i), y.drop(i))

        

        train_pred = model.predict(X.iloc[i].values.reshape(1, -1))[0]

        train_preds.append(train_pred)

        

        test_pred = model.predict(X_test)

        test_preds.append(test_pred)

    except Exception as e:

        print(e)

        train_preds.append(None)

        pass

    

test_preds = np.array(test_preds)

    

results = pd.DataFrame({

    "Id": ids_test,

    "SalePrice": np.exp(np.array(np.log(test_preds)).mean(axis=0))

})

results.to_csv("results.csv", index=False, float_format="%.2f")
from sklearn.metrics import mean_squared_log_error

np.sqrt(mean_squared_log_error(y, [

    yy if yy is not None else geom_mean(y) for yy in train_preds]))
results = pd.DataFrame({

    "Id": ids_test,

    "SalePrice": np.median(test_preds, axis=0)

})

results.to_csv("results.csv", index=False, float_format="%.2f")