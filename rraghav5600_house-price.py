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
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter



from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

combined = [train_df, test_df]

train_df.head()
print(train_df.info())

print('*'*30)

print(test_df.info())
def col_remover(col, ub):

    res = Counter(train_df[col])

    if res.most_common()[0][1]>ub: return True

    return False



cols = []

for col in train_df.columns:

    if col_remover(col, 1100): cols.append(col)

cols.append('Id')

print(len(cols))



for_submission = test_df['Id']

for i in combined:

    i.drop(cols, axis=1, inplace=True)
for df in combined:

    for col in train_df.columns:

        if col=='SalePrice': continue

        if Counter(df[col].isnull())[1]<300:

            if df[col].dtypes=='int64': df[col] = df[col].fillna(df[col].dropna().mean())

            elif df[col].dtypes=='float64': df[col] = df[col].fillna(df[col].dropna().mean())

            else: df[col] = df[col].fillna(df[col].dropna().mode()[0])



for df in combined:

        df.drop(['FireplaceQu'], axis=1, inplace=True)
s = 'MSSubClass MSZoning Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 BldgType HouseStyle OverallQual OverallCond RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 Heating HeatingQC CentralAir Electrical KitchenQual Functional FireplaceQu GarageType GarageFinish GarageQual GarageCond PavedDrive PoolQC Fence MiscFeature SaleType SaleCondition'

categorical = s.split()



def get_mapping(df, col):

    temp=dict()

    t=0

    for i in df[col].unique():

        temp[i]=t

        t=t+1

    return temp



def check_mapping(df, col, temp):

    size = len(temp)

    for i in df[col].unique():

        temp[i] = temp.get(i, size)

        if temp[i]==size: size=size+1

    return temp



for col in categorical:

    if col in train_df:

        if col=='SalePrice': continue

        ddict = get_mapping(train_df, col)

        train_df[col] = train_df[col].map(ddict).astype(int)

        ddict = check_mapping(test_df, col, ddict)

        test_df[col] = test_df[col].map(ddict).astype(int)
diff_dtype=[]

for col in train_df:

    if col=='SalePrice': continue

    if train_df[col].dtypes!=test_df[col].dtypes: diff_dtype.append(col)



for col in diff_dtype:

    print(col, train_df[col].dtypes, test_df[col].dtypes)

    

test_df = test_df.astype( {i:int for i in diff_dtype} )
for df in [train_df, test_df]:

    for col in df:

        if df[col].dtypes=='float64':

            df[col]=df[col].astype(int)
print(train_df.info())

print('*'*30)

print(test_df.info())

print('*'*30)

print([i.shape for i in combined])
cols = 'LotFrontage LotArea YearBuilt YearRemodAdd MasVnrArea BsmtFinSF1 BsmtUnfSF TotalBsmtSF 1stFlrSF 2ndFlrSF GrLivArea GarageYrBlt GarageArea WoodDeckSF OpenPorchSF YrSold'.split()



for df in [train_df, test_df]:

    for col in cols:

        df[col] = pd.cut(df[col], bins=5, labels=range(5))
for df in [train_df, test_df]:

    for col in df:

        df[col] = [int(i) for i in df[col].values]
X = train_df.drop(['SalePrice'], axis=1)

y = train_df['SalePrice']

train_X, val_X, train_y, val_y = train_test_split(X, y)

model_names, acc, err = [],[],[]
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeRegressor



rf = RandomForestClassifier(random_state=1)

xgb = XGBClassifier()

lr = LogisticRegression(solver='saga')

mlp = MLPClassifier()

dtc = DecisionTreeRegressor(random_state=0)



model_names = ['rf', 'xgb', 'lr', 'mlp', 'dtc']

models = [rf, xgb, lr, mlp, dtc]
for i in range(len(models)):

    models[i].fit(train_X, train_y)

    prediction = models[i].predict(val_X)

    acc.append(accuracy_score(val_y, prediction))

    err.append(mean_absolute_error(val_y, prediction))
models = pd.DataFrame({

    'Model': [i.upper() for i in model_names],

    'Accuracy': acc,

    'Mean Abs Error': err})

models.sort_values(by='Accuracy', ascending=False)
final_rf = RandomForestClassifier(random_state=1)

final_xgb = XGBClassifier()

final_lr = LogisticRegression(solver='saga')

final_mlp = MLPClassifier()

final_dtc = DecisionTreeRegressor(random_state=0)



models = [final_rf, final_xgb, final_lr, final_mlp, final_dtc]

for i in range(len(models)):

    models[i].fit(X,y)

    vals = models[i].predict(test_df)

    file = pd.DataFrame({'Id':for_submission, 'SalePrice':vals})

    file.to_csv('submission_'+model_names[i].upper()+'.csv', index = False)