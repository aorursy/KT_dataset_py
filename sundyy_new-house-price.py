%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
train_df = pd.read_csv("../input/train.csv", index_col=0)
test_df = pd.read_csv("../input/test.csv", index_col=0)
train_df.head()
test_df.head()
train_df["SalePrice"].hist()
price = pd.DataFrame( { "log1p_SalePrice" : np.log1p(train_df["SalePrice"]) } )
norm_price = pd.DataFrame( { "SalePrice" : train_df["SalePrice"] } )
price.hist()
# 删除掉SalePrice列的值 保证test和train列数统一
train_df.pop("SalePrice")
all_df = pd.concat((train_df, test_df),axis=0)
all_df.shape
all_df.head()
all_df["MSSubClass"].dtype
all_df["MSSubClass"] = all_df["MSSubClass"].astype(str)
all_df["MSSubClass"][1]
# pd.get_dummies(all_df["MSSubClass"], prefix="MSSubClass").head()
all_dummies_df = pd.get_dummies(all_df)
all_dummies_df.head()
t = all_dummies_df.isnull().sum()
t[t!=0]
all_dummies_df = all_dummies_df.fillna(all_dummies_df.median())
t = all_dummies_df.isnull().sum()
t[t!=0]
x_train = all_dummies_df.loc[train_df.index]
x_train.head()
x_test = all_dummies_df.loc[test_df.index]
x_test.head()
# x_train.shape
# x_train["log1p_SalePrice"] = price
# x_train.head()
train_len = x_train.shape[0] * 0.8
train_len
x_train_train = x_train.loc[0:train_len]
x_train_train.shape
x_train_test = x_train.loc[train_len+1:]
x_train_test.shape
from sklearn.linear_model import Ridge
#x_train = x_train.drop("log1p_SalePrice", axis=1)
ridge = Ridge()
ridge.fit(x_train, price)
ans=ridge.predict(x_test)
ans=ans.reshape(1,len(ans))
ans = ans[0]
x_train.head()
ans = pd.Series(ans)
y = pd.DataFrame({"Id":x_test.index, "SalePrice":ans})
y.head()
y["SalePrice"] = np.expm1(y["SalePrice"])
y.head()
y.to_csv("anss.csv", index=False)