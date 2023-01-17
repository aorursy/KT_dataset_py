import numpy as np

import pandas as pd
def tocsv(df, filename='tocsv.csv', index = False):

    df.to_csv(filename, index=index)
pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)
df_train = pd.read_csv('../input/train.csv') 

df_test = pd.read_csv('../input/test.csv')



id_col = df_test['Id']
print(f"Train shape: {df_train.shape}")

print(f"Test shape: {df_test.shape}")
%matplotlib inline

p = df_train.SalePrice.hist()

t = p.set_title("SalePrice distribution")
y_train = np.log10(df_train.SalePrice)

X_train = df_train.drop('SalePrice', axis=1)

X_test = df_test

X = pd.concat([X_train, X_test])
categoricals = X_train.select_dtypes(include='object').columns

numericals = X_train.select_dtypes(exclude='object').columns

print(f'{len(categoricals)} categorical features')

print(f'{len(numericals)} numerical features')
X[categoricals].isna().sum().sort_values(ascending=False)
X[categoricals] = X[categoricals].fillna("absent")
X[numericals].isna().sum().sort_values(ascending=False)
X = pd.get_dummies(X)

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy='mean')

imp_mean.fit(X)

X_imp = imp_mean.transform(X)



X_imp = pd.DataFrame(X_imp)
X_train_model = X_imp[0:1460]

X_test_model = X_imp[1460:]
from sklearn import linear_model

#regr = linear_model.Lasso(alpha=0.1)

regr = linear_model.LinearRegression()

regr = regr.fit(X_train_model, y_train)

out = regr.predict(X_test_model)



out = 10**out



out = pd.DataFrame(out,columns=['SalePrice'])



out.insert(0,"Id", id_col) 

out

tocsv(out)