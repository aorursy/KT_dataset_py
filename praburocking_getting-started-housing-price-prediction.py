#importing datalibbs
import pandas as pd
import numpy as np

#importing metrics
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

#importing models
from xgboost import XGBRegressor
from sklearn.preprocessing.imputation import Imputer

train=pd.read_csv("../input/train.csv")
print("train shape "+str(train.shape))
test=pd.read_csv("../input/test.csv")
print("test shape "+str(test.shape))
train_Id=train.pop("Id")
test_id=test.pop("Id")
def skewness(df1):
    df=df1.select_dtypes(exclude=["object"])
    for i in df.columns:
        if df[i].dropna().skew()>=0.75:
            print(str(i)+"   "+str(df[i].skew()))
            df1[i]=np.log1p(df[i])
    return df1
all_data=train.append(test)

all_data=skewness(all_data)
#one hot encoding
all_data=pd.get_dummies(all_data)

#to avoid collision in impution
#imputation may also fill our saleprice with some value (usually mean) ,but we dont want that so to avoid that we fill our saleprice with 0
all_data.SalePrice=all_data["SalePrice"].fillna(0)
def imputation_plus(df1):
    cols_with_missing = (col for col in df1.columns if df1[col].isnull().any())
    for col in cols_with_missing:
        df1[col + '_was_missing'] = df1[col].isnull()
        df1[col + '_was_missing'] = df1[col].isnull()
        return df1
all_data=imputation_plus(all_data)
#filling the missing data using imputation
my_imputer =Imputer()
data_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(all_data))
data_with_imputed_values.columns=all_data.columns
data_with_imputed_values.shape
def getMAC(X,y,model):
    train_X,val_X,train_y,val_y=train_test_split(X,y)
    model.fit(train_X,train_y)
    MAE=mean_absolute_error(model.predict(val_X),val_y)
    return MAE
training_data=data_with_imputed_values.copy()
training_SalePrice=training_data.pop("SalePrice")
XGBoost_data=data_with_imputed_values.iloc[:1460][:].copy()
corrmat = XGBoost_data.corr().abs()
cols = corrmat.nlargest(260, 'SalePrice')['SalePrice'].index
print(cols)
XGBoost_data=XGBoost_data[cols]
XGBoost_SalePrice=XGBoost_data.pop("SalePrice")
my_model = XGBRegressor(n_estimators=2900,learning_rate=0.045)
print(getMAC(XGBoost_data,XGBoost_SalePrice,my_model))
train_X,val_X,train_y,val_y=train_test_split(XGBoost_data,XGBoost_SalePrice)
my_model=clone(my_model)
my_model.fit(train_X, train_y, early_stopping_rounds=5,eval_set=[(val_X, val_y)], verbose=False)
Y_data=data_with_imputed_values.iloc[1460:][:]
Y_data=Y_data[cols]
Y_data.pop("SalePrice")
XGBoost_predict=np.expm1(my_model.predict(Y_data))
XGBoost_result=pd.DataFrame({"Id":test_id,"SalePrice":XGBoost_predict})
XGBoost_result.to_csv("XGBoost.csv",index=False)
XGBoost_result.head()
#0.13