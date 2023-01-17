import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_train = pd.read_csv("../input/train.csv", index_col='Id')
df_train.head()
df_train.dtypes
df_train.describe()
df_train.columns
sns.distplot(df_train.SalePrice)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df_train.select_dtypes(include=numerics).head()
f, ax = plt.subplots(figsize=(16, 8))
df_train.YearBuilt.value_counts().sort_index().plot.bar()
f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxenplot(x='YearBuilt', y='SalePrice', data=df_train)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
feature_cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']
sns.pairplot(df_train[feature_cols])
total_missing = df_train.isna().sum().sort_values(ascending=False)
percent_missing = total_missing * 100.0 / len(df_train) 
missing = pd.concat([total_missing, percent_missing], axis=1, keys=['Number', 'Percent'])
missing[missing.Number > 0]
feature_name = "GrLivArea"
plt.scatter(np.log(df_train["SalePrice"]), df_train[feature_name])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
class TypeSelector(TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self,X):
        return self
    def transform(self, X, Y=None):
        return X.select_dtypes(include=[self.dtype])
class MyEncoder(TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, X):
        return self
    def transform(self, X, Y=None):
        assert isinstance(X, pd.DataFrame)
        X_t = X.fillna("Other")
        for x in X_t:
            X_t[x] = X_t[x].astype("category")
        X_t = X_t.apply(lambda x : x.cat.codes.replace({-1: len(x.cat.categories)}))
        return X_t

X_train = df_train.drop(columns='SalePrice')
Y_train = df_train['SalePrice']

pipeline = FeatureUnion(transformer_list=[
    ("numbericals", Pipeline([
        ("selector", TypeSelector(np.number)),
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler())
    ])),
    ("categories", Pipeline([
        ("selector", TypeSelector("object")),
        ("encoder", MyEncoder())
    ]))
])

X_train = pipeline.fit_transform(X_train)
X_train.shape
X_train
Y_train = np.log(Y_train)
X, X_test, Y, Y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

preds = lin_reg.predict(X_test)
print("Root mean squared error: %0.3f" % (np.sqrt(mean_squared_error(Y_test, preds))))
from xgboost.sklearn import XGBRegressor
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'seed': 2018,
    'silent': True,
    'colsample_bylevel': 0.6,
    'n_estimators': 1000,
    'reg_alpha': 0, 'reg_lambda': 1,
    'nthread' : 4,
}

model = XGBRegressor(**params)
model.fit(X, Y)
preds = model.predict(X_test)
print("Root mean squared error : %0.6f" % (mean_squared_error(Y_test, preds)))
df_test = pd.read_csv("../input/test.csv", index_col='Id')
df_test.head()
X_submit = pipeline.fit_transform(df_test)
preds_submit = model.predict(X_submit)
preds_submit = np.exp(preds_submit)
df_result = pd.DataFrame(np.c_[df_test.reset_index().iloc[:, 0].values, preds_submit], columns=['Id', 'SalePrice'])
df_result['Id'] = df_result['Id'].astype(int)
df_result.to_csv("submission.csv", index=False)
preds_submit


