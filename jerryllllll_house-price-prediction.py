import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
sns.set()

import os
print(os.listdir("../input"))
path = "../input"
train_file = pd.read_csv(path+"/train.csv")
test_file = pd.read_csv(path+"/test.csv")
SalePrice = train_file["SalePrice"]
test_id = test_file.Id

print(train_file.shape)
print(test_file.shape)




train_file.isnull().sum().sort_values(ascending=False)
#Remove the ID columns
train = train_file.drop(["Id"],axis=1)
test = test_file.drop(["Id"],axis=1)


print("train: ",train.shape)
print("test: ",test.shape)
num_to_object = ["MSSubClass","OverallQual","OverallCond","YrSold"]
for col in num_to_object:
    train[col] = train[col].astype("object")
    test[col] = test[col].astype("object")
sns.distplot(train["SalePrice"])
title = "Skewness: "+str(train["SalePrice"].skew())[:4]+" Kurtosis: "+str(train["SalePrice"].kurt())[:4]
plt.title(title)
train["SalePrice_log"] = np.log1p(train.SalePrice)
sns.distplot(train["SalePrice_log"])
title = "Skewness: "+str(train["SalePrice_log"].skew())[:4]+" Kurtosis: "+str(train["SalePrice_log"].kurt())[:4]
plt.title(title)
train.drop("SalePrice",axis=1,inplace=True)
numerical_columns = list(train.select_dtypes(["float64","int64"]).columns)
numerical_columns.remove("SalePrice_log")
print(numerical_columns)
train[numerical_columns].isnull().sum().sort_values(ascending=False)
#LotFrontage GarageYrBlt MasVnrArea
train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace=True)
train["GarageYrBlt"].fillna(train["GarageYrBlt"].mean(),inplace=True)
train["MasVnrArea"].fillna(train["MasVnrArea"].mean(),inplace=True)
print(len(numerical_columns))
nrows=11
ncols=3
fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols*3.5,nrows*3))
num_col_pear = []
for r in range(nrows):
    for c in range(ncols):
        idx = r*3+c
        if idx < len(numerical_columns):
            sns.regplot(train[numerical_columns[idx]],train["SalePrice_log"],ax=axes[r][c])
            pearson = stats.pearsonr(train[numerical_columns[idx]],train["SalePrice_log"])
            num_col_pear.append(abs(pearson[0]))
            axes[r][c].set_title("r:%2f  p-value:%2f"%(pearson[0],pearson[1]))
        
plt.tight_layout()
plt.show()
num_col_pear = pd.DataFrame({"Feature":numerical_columns,"Pearson":num_col_pear})
#check pearson correlation,choose a Threshold value(0.4 here) for filtering
num_col_pear.sort_values(by="Pearson",ascending=False)
weak_pear_col = list(num_col_pear[num_col_pear.Pearson<0.4]["Feature"])
strong_pear_col = list(num_col_pear[num_col_pear.Pearson>=0.4]["Feature"])

sns.heatmap(train[strong_pear_col+["SalePrice_log"]].corr(),cbar=True, square=True, fmt='.2f',annot=True,annot_kws={"size":5})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
train.drop(weak_pear_col,axis=1,inplace=True)
test.drop(weak_pear_col,axis=1,inplace=True)

train.drop(["TotRmsAbvGrd","GarageArea","1stFlrSF"],axis=1,inplace=True)
test.drop(["TotRmsAbvGrd","GarageArea","1stFlrSF"],axis=1,inplace=True)

object_columns = list(train.select_dtypes(["object"]).columns)
#plot: show boxplot of SalePrice_log and each object column
#will update later
unbalance_columns = [] 
for col in object_columns:
    temp = train.groupby(col)["SalePrice_log"].count()/train.shape[0]
    print(temp)
    if max(temp.values) > 0.90 or sum(temp.values)<0.10:
        unbalance_columns.append(col)
    print("#"*50)
print(unbalance_columns)
train.drop(unbalance_columns,axis=1,inplace=True)
test.drop(unbalance_columns,axis=1,inplace=True)


object_nunique = train.select_dtypes(["object"]).apply(pd.Series.nunique)
two_label_columns = list(object_nunique.index[object_nunique<=2])
multi_label_columns = list(object_nunique.index[object_nunique>2])

for col in two_label_columns:
    if train[col].isnull().sum()>0 or test[col].isnull().sum()>0:
        two_label_columns.remove(col)
        multi_label_columns.append(col)
def RMSE(pred,y):
    y = list(y)
    pred = list(pred)
    sum0 = 0.0
    for i in range(len(pred)):
        sum0 += (pred[i]-y[i])*(pred[i]-y[i])
    return math.sqrt(sum0/len(pred))
df_train = train.drop("SalePrice_log",axis=1)
df_test = test.copy()
print(two_label_columns)
for col in two_label_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

df_train = pd.get_dummies(df_train,columns=multi_label_columns)
df_test = pd.get_dummies(df_test,columns=multi_label_columns)

print("df_train:",df_train.shape)
df_train.head()
print("df_test:",df_test.shape)
df_test.head()
df_train,df_test = df_train.align(df_test,join="inner",axis=1)
train_x, test_x, train_y, test_y = train_test_split(df_train,train["SalePrice_log"])
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train_x,train_y)
pred_x = rf.predict(test_x)
print(RMSE(pred_x,test_y))

missing_num_cols = df_test.select_dtypes(["float64","int64"]).isnull().sum()
df_test_fill = df_test.copy()
na_col = missing_num_cols[missing_num_cols>0].index
for col in na_col:
    df_test_fill[col].fillna(df_test_fill[col].mean(),inplace=True)

pred_rf = np.expm1(rf.predict(df_test_fill))
xgb = XGBRegressor(objective="reg:linear",max_depth=10,learning_rate=0.1)
xgb.fit(train_x,train_y)
pred_x = xgb.predict(test_x)
print(RMSE(pred_x,test_y))
pred_xgb = np.expm1(xgb.predict(df_test))
