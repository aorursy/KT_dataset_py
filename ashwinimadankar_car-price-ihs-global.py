import pandas as pd   #data pre-processing

import numpy as np    #mathematical operation

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

from math import sqrt #mathematical functions



from sklearn.model_selection import train_test_split #spliting data

from sklearn.preprocessing import StandardScaler,MinMaxScaler  #scaling data

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error #model performace checking



from sklearn.linear_model import LinearRegression #linear regression 

from sklearn.ensemble import RandomForestRegressor#randomforest regression

import xgboost as Xgb                             #Xgboost regession



import warnings

warnings.filterwarnings("ignore") #to ignore warnings
!pwd
data = pd.read_csv("../input/car-price/train_data.csv")
# data["Global_Sales_Sub-Segment_Brand"] = data["Global_Sales_Sub-Segment"]+"_"+data["Brand"]
# data["year_cal"] =  data["year"] - data["Generation_Year"]
# data = data[data["year_cal"]>0]
data.drop(["year","Generation_Year"],axis=1,inplace=True)
data.describe()
data.drop(["Nameplate","vehicle_id","date"],inplace=True,axis=1)
numeric=data.select_dtypes(include=['float64','int64'])

categorical = data.select_dtypes(include=['object'])
numeric.columns
numeric.describe()
# #LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's

# for i, col in enumerate (numeric.columns):

#     plt.figure(i)

#     sns.regplot(x=data[col],y=data['Price_USD'])
# np.percentile(data["Price_USD"],100)
# data = data[data["Price_USD"] < np.percentile(data["Price_USD"],100)]
#LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's



for i, col in enumerate (numeric.columns):

    plt.figure(i)

    sns.regplot(x=data[col],y=data['Price_USD'])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(24,24))



for i, column in enumerate(numeric.columns):

    sns.distplot(data[column],ax=axes[i//3,i%3])
# def remove_outlier(df_in, col_name):

#     q1 = df_in[col_name].quantile(0.25)

#     q3 = df_in[col_name].quantile(0.75)

#     iqr = q3-q1 #Interquartile range

#     fence_low  = q1-(1.5*iqr)

#     fence_high = q3+(1.5*iqr)

#     df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

#     return df_out
# numeric.drop(["Price_USD"],axis=1,inplace=True)
# for i, col in enumerate (numeric.columns):

#     print(col)

#     data = remove_outlier(data,col)
# data["Fuel_Type"].value_counts()
#LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's



for i, col in enumerate (numeric.columns):

    plt.figure(i)

    sns.regplot(x=data[col],y=data['Price_USD'])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(24,24))



for i, column in enumerate(numeric.columns):

    sns.distplot(data[column],ax=axes[i//3,i%3])
numeric=data.select_dtypes(include=['float64','int64'])

categorical = data.select_dtypes(include=['object'])
corr=numeric.corr()

plt.figure(figsize=(15,8))

sns.heatmap(corr,annot=True,cmap="YlGnBu")
#Correlation with output variable

cor_target = abs(corr["Price_USD"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
data=data.drop(columns=categorical)

data.head(2)
data.columns
data = np.log(data)
y = data["Price_USD"]
data.drop(["Fuel_cons_combined","Price_USD","Length"],axis=1,inplace=True)
categorical.columns
# categorical.drop(["Fuel_Type"],axis=1,inplace=True)
categorical["Fuel_Type"].value_counts()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown = 'ignore')

c = ohe.fit_transform(categorical).toarray()
ohe.categories_
categorical.columns
data.values
np.concatenate((data.values,c),axis=1).shape
X=np.concatenate((data.values,c),axis=1)
y
# #Min-Max Scaling of data range 0 to 1

# ms = MinMaxScaler()

# msy =  MinMaxScaler()

# dfX_scaled = ms.fit_transform(X.values)

# # y =  msy.fit_transform(y.values)
# from sklearn.preprocessing import scale



# cols=X.columns

# dfX_scaled=pd.DataFrame(scale(X))

# dfX_scaled.columns=cols

# dfX_scaled.columns
# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=101)
#function to calculate mean absolute percentile error 

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# lr = LinearRegression()

# lr.fit(X_train,y_train)

# pred = lr.predict(X_test)

# print("Mse:",mean_squared_error(y_test,pred))

# rms = sqrt(mean_squared_error(y_test,pred))

# print("Rmse:",rms)

# print("Mape:",mean_absolute_percentage_error(y_test,pred))

# print("R-square:",r2_score(y_test,pred))
# rf = RandomForestRegressor(random_state=101)

# rf.fit(X_train,y_train)

# pred = rf.predict(X_test)

# print("Mse:",mean_squared_error(y_test,pred))

# rms = sqrt(mean_squared_error(y_test,pred))

# print("Rmse:",rms)

# print("Mape:",mean_absolute_percentage_error(y_test,pred))

# print("R-square:",r2_score(y_test,pred))
# import xgboost as Xgb

# xgb = Xgb.XGBRegressor(random_state=101,learning_rate=0.3)

# xgb = xgb.fit(X_train,y_train)

# pred = xgb.predict(X_test)

# print("Mse:",mean_squared_error(y_test,pred))

# rms = sqrt(mean_squared_error(y_test,pred))

# print("Rmse:",rms)

# print("Mape:",mean_absolute_percentage_error(y_test,pred))

# print("R-square:",r2_score(y_test,pred))
import xgboost as Xgb

xgb = Xgb.XGBRegressor(random_state=101,learning_rate=0.3)

xgb = xgb.fit(X, y)

pred = xgb.predict(X)

print("Mse:",mean_squared_error(y,pred))

rms = sqrt(mean_squared_error(y,pred))

print("Rmse:",rms)

# print("Mape:",mean_absolute_percentage_error(y,pred))

print("R-square:",r2_score(y,pred))
# rf = RandomForestRegressor(random_state=101)

# rf = rf.fit(X, y)

# pred = rf.predict(X)

# print("Mse:",mean_squared_error(y,pred))

# rms = sqrt(mean_squared_error(y,pred))

# print("Rmse:",rms)

# # print("Mape:",mean_absolute_percentage_error(y,pred))

# print("R-square:",r2_score(y,pred))
test = pd.read_csv("../input/car-price/oos_data.csv")

test.columns
# test["year_cal"] =  test["year"] ,:""- test["Generation_Year"]

test.drop(["year","Generation_Year","Length"],axis=1,inplace=True)
# test["Global_Sales_Sub-Segment_Brand"] = test["Global_Sales_Sub-Segment"]+"_"+test["Brand"]

test.drop(["Nameplate","vehicle_id","date"],inplace=True,axis=1)

numeric=test.select_dtypes(include=['float64','int64'])

categorical = test.select_dtypes(include=['object'])
# test.drop(["year","Generation_Year"],axis=1,inplace=True)
test.drop(["Fuel_cons_combined"],axis=1,inplace=True)
test.columns
data.columns
categorical.columns
test=test.drop(columns=categorical)
test.columns
test = np.log(test)
len(c[0])
t = ohe.transform(categorical).toarray()
test.shape
test  = np.concatenate((test.values,t),axis=1)
X.shape
test.shape
# cols=test.columns

# test_scaled=pd.DataFrame(scale(test))

# test_scaled.columns=cols

# test_scaled.columns
vehicle_id = pd.read_csv("../input/car-price/oos_data.csv",usecols=["vehicle_id"])
vehicle_id["Price_USD"] = xgb.predict(test)

# vehicle_id["Price_USD"] = msy.inverse_transform(vehicle_id[["Price_USD"]].values)



vehicle_id["Price_USD"] = np.exp(vehicle_id["Price_USD"])

vehicle_id.to_csv("Global_Submission_f_xgb::BrandXlength_gear_ohe.csv",index=False)
vehicle_id