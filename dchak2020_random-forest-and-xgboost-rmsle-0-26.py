import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



from sklearn.model_selection import train_test_split,cross_validate,cross_val_score,cross_val_predict

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler,StandardScaler
#Read training/test set file

df1=pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

df2=pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")



#Define preprocessing function to apply to train & test sets

def preprocess(df):

    #Extract hour,day,month,weekday from datetime

    df["hour"]=pd.to_datetime(df["datetime"]).dt.hour

    df["dayofweek"]=pd.to_datetime(df["datetime"]).dt.dayofweek

    df["dayofmonth"]=pd.to_datetime(df["datetime"]).dt.day

    df["month"]=pd.to_datetime(df["datetime"]).dt.month

    df["year"]=pd.to_datetime(df["datetime"]).dt.year.map({2011:0, 2012:1})

    #One-hot encode weather and season

    df=pd.get_dummies(df,columns=["weather","season"],prefix=["weather","season"],drop_first=True,dtype=int)

    #Drop out datetime and return

    return df.drop(["datetime"],axis=1)



df1=preprocess(df1)

df2=preprocess(df2)
#Visualize correlation matrix.

cor_mat= df1.corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(cor_mat,mask=mask,square=True,annot=True,cbar=True)
sns.barplot(x="hour",y="count",data=df1)
sns.lineplot(x="temp",y="count",data=df1)
#Define features and labels

features=[x for x in df1.columns if x not in ["count","casual","registered"]]

label1=["casual"]

label2=["registered"]

label=["count"]



#Separate features from labels

X_train,y1_train,y2_train,y_train=df1[features],df1[label1],df1[label2],df1[label]

X_test=df2[features]



#Validation set

X_train_0,X_train_1,y_train_0,y_train_1=train_test_split(X_train,y_train,test_size=0.25,random_state=42)
#Train Random Forest Model

rf=RandomForestRegressor(n_estimators=500,n_jobs=-1)



rf.fit(X_train_0,y_train_0)

pred=rf.predict(X_train_1)



#Validation set error

print(mean_squared_log_error(y_train_1,pred)**0.5)
#Transforming labels

rf.fit(X_train_0,y_train_0.apply(lambda x:np.log1p(x)))

pred=np.expm1(rf.predict(X_train_1))



#Validation set error

print(mean_squared_log_error(y_train_1,pred)**0.5)
#Print most important features

print(pd.DataFrame({"Features": features,"Importance" : rf.feature_importances_.round(2)}).sort_values("Importance",ascending=False))
#Train Extreme Gradient Boosting Model



#Create D-matrices for training and validation sets



DM_0=xgb.DMatrix(X_train_0,y_train_0)

DM_1=xgb.DMatrix(X_train_1,y_train_1)



params={"booster":"gbtree"}

xgbr=xgb.train(dtrain=DM_0,num_boost_round=380,params=params)



pred = xgbr.predict(DM_1)



pred[pred<0]=0

#Validation set error

print(mean_squared_log_error(y_train_1,pred)**0.5)
#Transforming labels

DM_0=xgb.DMatrix(X_train_0,y_train_0.apply(lambda x:np.log1p(x)))



params={"booster":"gbtree"}

xgbr=xgb.train(dtrain=DM_0,num_boost_round=380,params=params)



pred = np.expm1(xgbr.predict(DM_1))



pred[pred<0]=0



#Validation set error

print(mean_squared_log_error(y_train_1,pred)**0.5)
#Fine-tuning some hyperparameters

params={"booster":"gbtree","learning_rate":0.09,"n_jobs":-1,"subsample":0.7,"alpha": 0.009}



xgbr=xgb.train(dtrain=DM_0,num_boost_round=380,params=params)



pred = np.expm1(xgbr.predict(DM_1))



#Validation set error

print(mean_squared_log_error(y_train_1,pred)**0.5)