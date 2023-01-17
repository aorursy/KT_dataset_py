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
path='/kaggle/input/home-data-for-ml-course/train.csv'
data=pd.read_csv(path)
pd.set_option("display.max_columns" , 100)
pd.set_option("display.max_rows" , 100)
data.describe()
data.head(15)
print(data.shape)
path='../input/home-data-for-ml-course/test.csv'
data_test=pd.read_csv(path)
pd.set_option("display.max_columns" , 100)
pd.set_option("display.max_rows" , 100)
data_test.describe()


data.isna().sum().sort_values(ascending=False)
id=data_test['Id']
data_test.isna().sum().sort_values(ascending=False)
y=data["SalePrice"]
merge=[data , data_test]
df=pd.concat(merge)
df=df.drop(["Id" ,"SalePrice"] , axis=1)
df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.columns
df['FireplaceQu'].value_counts()

df['FireplaceQu'].fillna('Gd',inplace=True)
df['LotFrontage'].value_counts()

df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)
df['GarageCond'].value_counts()
df['GarageCond'].fillna('TA',inplace=True)
df['GarageQual'].fillna('TA',inplace=True)

df['GarageFinish'].value_counts()

df['GarageFinish'].fillna('Unf',inplace=True)
df['GarageYrBlt'].value_counts()
df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(),inplace=True)
df['GarageType'].value_counts()
df['GarageType'].fillna('Attchd',inplace=True)
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0],inplace=True)
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0],inplace=True)
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0],inplace=True)
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0],inplace=True)
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0],inplace=True)
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0],inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(),inplace=True)
df['MSZoning'].fillna(df['MSZoning'].mode()[0],inplace=True)


df['BsmtHalfBath'].value_counts()


df['BsmtFullBath'].value_counts()
df['BsmtHalfBath'].fillna(round(df['BsmtHalfBath'].mean()),inplace=True)
df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0],inplace=True)
df['Utilities'].fillna(df['Utilities'].mode()[0],inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0],inplace=True)

df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mode()[0],inplace=True)
df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean(),inplace=True)
df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0],inplace=True)
df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mean(),inplace=True)
df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean(),inplace=True)
df['SaleType'].fillna(df['SaleType'].mode()[0],inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0],inplace=True)
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0],inplace=True)
df['GarageArea'].fillna(df['GarageArea'].mean(),inplace=True)
df['GarageCars'].fillna(df['GarageCars'].mode()[0],inplace=True)
df.isna().sum()
train=df.iloc[:1460]
test=df.iloc[1460:]

        
    
feature = train.select_dtypes(exclude=np.number)
feature

train = pd.get_dummies(train, columns=feature.columns)
train
test = pd.get_dummies(test, columns=feature.columns)
test
for col in train.columns:
    if col not in test.columns:
        train.drop([col],axis=1,inplace=True)
for col in test.columns:
    if col not in train.columns:
        test.drop([col],axis=1,inplace=True)
print(train.shape,test.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

model = RandomForestRegressor()
model.fit(train, y)
y_pred = model.predict(test)
output = pd.DataFrame({'Id': id,
                       'SalePrice': y_pred})
output.to_csv('submission.csv', index=False)

result_df1=pd.DataFrame(y_pred)
result=pd.concat([id ,result_df1] ,axis=1)
result
result.to_csv("sub.csv")
import base64
import pandas as pd
from IPython.display import HTML

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

file = 'sub.csv'
df = pd.read_csv(file, sep=",", header=None)
create_download_link(df, title = "Download CSV file", filename = "HP4.csv")
