import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt

print(os.listdir("../input"))
dfFull = pd.read_csv("../input/train.csv")
# Taking care of missing data
DropCpl = ["Street","Alley","Utilities","Condition2","LandSlope","RoofMatl","Heating","PoolArea","PoolQC","MiscFeature"
          ,"LandContour","Fence","FireplaceQu"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dfFull = dfFull.drop(DropCpl,1)
dfInt = dfFull.select_dtypes(include=numerics)
y = dfFull.iloc[:,dfFull.shape[1]-1].values
dfInt = dfInt.fillna(dfInt.mean())
dfFull = dfFull.drop(dfInt.columns,axis=1)
dfNaN = dfFull[dfFull.columns[dfFull.isna().any()].tolist()]
dfFull = dfFull.drop(dfNaN.columns,axis=1)
dfNaN = dfNaN.apply(lambda x:x.fillna(x.value_counts().index[0]))
dfNaN = pd.get_dummies(dfNaN)
dfFull = pd.get_dummies(dfFull)
dfFull = pd.concat([dfFull,dfNaN,dfInt], axis=1, sort=False)
#dfFull = dfFull.drop("SalePrice",1)
#dfFull["MoSold"] = (dfFull["YrSold"] - dfFull["YearRemodAdd"])*12 + dfFull["MoSold"]
dfFull = dfFull.drop(["YearRemodAdd","MoSold"],1)
X = dfFull.iloc[:,1:-1].values
df1Final = pd.DataFrame()
df1Full = pd.read_csv('../input/test.csv')
df1Full = df1Full.drop(DropCpl,1)
df1Int = df1Full.select_dtypes(include=numerics)
df1Int = df1Int.fillna(df1Int.mean())
df1Full = df1Full.drop(df1Int.columns,axis=1)
df1NaN = df1Full[df1Full.columns[df1Full.isna().any()].tolist()]
df1Full = df1Full.drop(df1NaN.columns,axis=1)
df1NaN = df1NaN.apply(lambda x:x.fillna(x.value_counts().index[0]))
df1NaN = pd.get_dummies(df1NaN)
df1Full = pd.get_dummies(df1Full)
df1Full = pd.concat([df1Full,df1NaN,df1Int], axis=1, sort=False)
#df1Full["MoSold"] = (df1Full["YrSold"] - df1Full["YearRemodAdd"])*12 + df1Full["MoSold"]
df1Full = df1Full.drop(["YearRemodAdd","MoSold"],1)
#for col in df.select_dtypes(include=['number']).columns:
for col in dfFull.columns:
    if col not in df1Full:
        df1Final[col] = dfFull[col]
    else:
        df1Final[col] = df1Full[col]

df1Final = df1Final.drop("SalePrice",1)
X1 = df1Final.iloc[:,1:].values
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, dfFull.drop(['SalePrice','Id'], axis = 1), dfFull['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
xgb_test.fit(dfFull.drop(['SalePrice','Id'], axis = 1), dfFull['SalePrice'])
yLinear_pred = xgb_test.predict(df1Final.drop(['Id'], axis = 1))
submission = pd.DataFrame(
    {'Id': df1Final.Id, 'SalePrice': yLinear_pred},
    columns = ['Id', 'SalePrice'])
submission.to_csv('submission.csv', index = False)
