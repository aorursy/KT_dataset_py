import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
# Create a list of columns where the percentage of missing values is 30% or greater.

[col for col in train.columns if train[col].isnull().sum()/len(train.index) > .3]
num_cols = train.select_dtypes('number').columns
num_cols = num_cols.drop(['Id','MSSubClass'])
# for col in num_cols:
#     plt.hist(train[col])
#     plt.title(col)
#     plt.show()
import seaborn as sns

areas = [col for col in num_cols if train[col].map(lambda n:n==0).sum()/len(train.index) > .5]

for col in areas:
    sns.regplot(x=col,y='SalePrice',data=train)
    plt.show()

# train.groupby(train['FireplaceQu'].fillna('temp')).SalePrice.describe()
# train.groupby('Fireplaces').SalePrice.describe()
train.apply(lambda n: n.YearBuilt != n.GarageYrBlt if n.GarageYrBlt != 'NaN' else True, axis='columns').sum()
X = train.drop(['Alley','FireplaceQu','PoolArea','PoolQC','Fence','MiscFeature','SalePrice','PoolArea','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch','LowQualFinSF'], axis=1)
y = train['SalePrice']
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=1000, random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

cv = cross_val_score(clf, X, y, cv=5)
print(cv)
print(cv.mean())
test = test.drop(['Alley','FireplaceQu','PoolArea','PoolQC','Fence','MiscFeature','PoolArea','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch','LowQualFinSF'], axis=1)
clf.fit(X,y)
predict = clf.predict(test)

submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": predict
    })
submission.to_csv('submission.csv', index=False)
# from xgboost import XGBRegressor

# model_2 = XGBRegressor(n_estimators=1000)
# model_2_pl = Pipeline(steps=[('preprocessor', preprocessor),
#                             ('model_2',model_2)])

# model_2_pl.fit(train_X, y)
# prediction = model_2_pl.predict(X)

# print(mean_absolute_error(y, prediction))
# print(model_2_pl.score(X,y))
