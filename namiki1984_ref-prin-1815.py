import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score
from math import sqrt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
# test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
tr_ex_obj = train.select_dtypes(exclude=object)
display(tr_ex_obj)
nnum = tr_ex_obj.isnull().sum()
nnum0 = nnum[nnum!=0]
nnum0
tr_ex_obj = tr_ex_obj.fillna({"LotFrontage":tr_ex_obj["LotFrontage"].median(),
                             "MasVnrArea":tr_ex_obj["MasVnrArea"].median(),
                             "GarageYrBlt":tr_ex_obj["GarageYrBlt"].median()})
# for i in range(len(nnum0.index)):
#     col = nnum0.index[i]
#     fn = tr_ex_obj.loc[:,col].median()
#     tr_ex_obj.loc[:,col] = tr_ex_obj.loc[:,col].fillna(fn)
#     tr_ex_obj.loc[:,col] = tr_ex_obj.loc[:,col].fillna(fn)

tr_ex_obj.isnull().sum()
X, y = tr_ex_obj.drop(['Id', 'SalePrice'], axis=1), tr_ex_obj['SalePrice']
sc = StandardScaler()
X1 = pd.DataFrame(sc.fit_transform(X),columns=X.columns)
display(X.head(), X1.head())
ncomp = 6
pca = PCA(n_components=ncomp)
pca.fit(X1)
feature = pca.transform(X1)
a = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(ncomp)], columns=["固有値"])
# a = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(X1.columns))], columns=["固有値"])
a["寄与率"] = a["固有値"] / a["固有値"].sum()
a["累積寄与率"] = ""
for i in range(len(a["寄与率"])):
    if i != 0:
        a.at[a.index[i],"累積寄与率"] = a.at[a.index[i], "寄与率"] + a.at[a.index[i-1], "累積寄与率"]
    elif i == 0:
        a.at[a.index[0], "累積寄与率"] = a.at[a.index[0], "寄与率"]
display(a)
plt.subplots(facecolor='w', figsize=(8,5)) 
plt.grid()
plt.plot(a.index, a["累積寄与率"])
plt.bar(a.index, a["寄与率"], alpha=0.4)
plt.show()
fuka = pd.DataFrame((-1)*pca.components_, columns=X.columns[:], index=["PC{}".format(x + 1) for x in range(ncomp)]).T
fuka
# tmp1 = pd.DataFrame((-1)*feature, columns=["PC{}".format(x + 1) for x in range(len(X.columns))]).iloc[:,0:17]
# tmp2 = pd.DataFrame((-1)*feature, columns=["PC{}".format(x + 1) for x in range(len(X.columns))]).iloc[:,0:24]
tr_pca = pd.DataFrame((-1)*feature, columns=["PC{}".format(x + 1) for x in range(ncomp)])
display(tr_pca)
X1_train, X1_valid, y1_train, y1_valid = train_test_split(X, y, test_size=0.3, random_state=42)
X2_train, X2_valid, y2_train, y2_valid = train_test_split(tr_pca, y, test_size=0.3, random_state=42)
rf_1 = RandomForestRegressor(random_state=42).fit(X1_train, y1_train)
rf_2 = RandomForestRegressor(random_state=42).fit(X2_train, y2_train)
pred1 = rf_1.predict(X1_valid)
pred2 = rf_2.predict(X2_valid)
rmse1 = round(sqrt(mse(y1_valid, pred1)), 4)
rmse2 = round(sqrt(mse(y2_valid, pred2)), 4)
rmsle1 = round(sqrt(msle(y1_valid, pred1)), 4)
rmsle2 = round(sqrt(msle(y2_valid, pred2)), 4)
r2_1 = round(r2_score(y1_valid, pred1), 4)
r2_2 = round(r2_score(y2_valid, pred2), 4)
print(f"オリジナルデータ rmse:{rmse1}   rmsle:{rmsle1}   R2:{r2_1}")
print(f"主成分　     　 rmse:{rmse2}   rmsle:{rmsle2}   R2:{r2_2}")
# pred_t1.to_csv("aaa.csv",index=False)
