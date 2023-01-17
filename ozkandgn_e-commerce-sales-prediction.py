import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/e-commerce-sales/data.csv")
data = data.drop(data.columns[0],axis=1)
data.sample(5)
data.columns = ["date","sessionid","locale","pagetype","itemid","cartamount",
               "productprice","oldproductprice","isinstock","issearched"]
data.sample(5)
data.info()
data["locale"].value_counts()
data["pagetype"].value_counts()
def to_timestamp(val):
    values = val.split(":")
    return int(values[0]) * 3600 + int(values[1]) * 60 + int(values[2])

data["datetm"] = data["date"].apply(lambda x: to_timestamp(x.split(" ")[1]))
data["datetm"].sample(5)
data["sessionid"].value_counts()
data["sessioncount"] = data["sessionid"].map(data["sessionid"].value_counts().to_dict())
data["sessioncount"].sample(5)
### Pagetype to binary success

data["success"] = data["pagetype"].apply(lambda x: 1 if x=="success" else 0)
data["success"].value_counts()
### Clearing itemid

filled_itemid = data["itemid"].fillna("0")

filled_itemid = filled_itemid.apply(lambda x: "0" if "[" in x else x)

filled_itemid = filled_itemid.apply(lambda x: x[1:-1] if "\"" in x else x)

filled_itemid = filled_itemid.astype(int)

data["itemid"] = filled_itemid.replace(0,np.nan)
import missingno as msno

msno.matrix(data[["itemid","productprice","oldproductprice","isinstock"]][:50])
msno.heatmap(data[["itemid","productprice","oldproductprice","isinstock"]])
data["locale"].value_counts()

### Nan value amount %5
data["locale"] = data["locale"].fillna("tr_TR")
data[(data["productprice"]==0)&(data["success"]==1)]
data = data[data["productprice"]!=0]
data[["productprice","oldproductprice"]].info()
data[(data["productprice"].notnull()) & (data["oldproductprice"].isna())]
data["oldproductprice"] = data["oldproductprice"].fillna(data["productprice"])
data[["cartamount","productprice","oldproductprice"]].corr()
data[data["cartamount"]==0]["success"].value_counts()
data[data["cartamount"].isna()]["success"].value_counts()
data["cartamount"] = data["cartamount"].fillna(0)
data["localetr"] = data["locale"].apply(lambda x: 1 if x=="tr_TR" else 0)
clear_data = data[data.notnull().all(axis=1)]
clear_data.info()
clear_data.success.value_counts()
clear_data.pagetype.value_counts()
def outliers_calc(df,col,quantile):
    df_balance = df[col]

    Q1 = df_balance.quantile(quantile[0])
    Q3 = df_balance.quantile(quantile[1])
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[col]>upper_bound)|(df[col]<lower_bound)]
    
outliers_calc(data,"cartamount",(0.005,0.995))
outliers_calc(data,"productprice",(0.05,0.95))
data[["pagetype","cartamount"]].groupby("pagetype").mean().plot(kind="bar")
sns.lmplot(x="cartamount",y="productprice",data=clear_data)
sns.catplot(x="issearched",y="cartamount",data=data,kind="bar")
sns.catplot(x="success",y="sessioncount",data=data,kind="bar")
sns.catplot(x="localetr",y="success",data=data,kind="bar")

data[(data["localetr"]==0) & (data["success"]==1)]
sns.violinplot(x="success",y="datetm",data=data)
data.info()
df = data[data["itemid"].notnull()]
df = df[["itemid","cartamount","issearched","datetm","sessioncount","success"]]
X = df.drop(["success"],axis=1)
y = df["success"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier()

mlp_params = {"alpha":[0.1,0.02,0.001],
             "hidden_layer_sizes":[(5,5,3),(20,10),(20,5)],
              "activation":["logistic"]}
from sklearn.model_selection import GridSearchCV

### Using Small Data !!
mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=2,verbose=2,n_jobs=-1).fit(X_train_scaled,y_train)
mlp_cv_model.best_params_
model = MLPClassifier(activation="logistic",
                      hidden_layer_sizes=(5,5,3),
                      alpha =0.1).fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error

### RMSE
np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=model,X=X,y=y,cv=3)
accuracies
accuracies.mean()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=5,
                       max_features=4,
                       min_samples_split=10,
                       n_estimators=200).fit(X_train,y_train)
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
accuracies=cross_val_score(estimator=model,X=X,y=y,cv=3)
accuracies
accuracies.mean()

feature_imp = pd.Series(model.feature_importances_,
                index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20,10))

sns.barplot(feature_imp,feature_imp.index)
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate":[0.005,0.01,0.04],
            "max_depth":[3,5,8],
            "n_estimators":[20,50,100],
            "num_leaves":[2,4,8]}
lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params,cv=2,n_jobs=-1,verbose=2).fit(X_train,y_train)
lgbm_cv_model.best_params_

lgbm_model = LGBMClassifier(learning_rate=0.005,
                       max_depth=3,
                       n_estimators=20,
                       num_leaves=2).fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

accuracies=cross_val_score(estimator=lgbm_model,X=X,y=y,cv=5)
accuracies
accuracies.mean()
feature_imp = pd.Series(lgbm_model.feature_importances_,
                index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20,10))

sns.barplot(feature_imp,feature_imp.index)
from pdpbox import pdp

def plot_partial_dep(model,data,feature):
    pdp_dist = pdp.pdp_isolate(model=model, dataset=data
                               , model_features=data.columns
                               , feature=feature)
    pdp.pdp_plot(pdp_isolate_out=pdp_dist, feature_name=feature);
plot_partial_dep(model,X_train,"cartamount")

plot_partial_dep(model,X_train,"sessioncount")

plot_partial_dep(model,X_train,"datetm")
