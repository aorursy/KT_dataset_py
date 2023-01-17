import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
train['Date'] = pd.to_datetime(train['Date']).astype('int64')
test['Date'] = pd.to_datetime(test['Date']).astype('int64')
train["ForecastId"]=0
test["Id"]=0
test["ConfirmedCases"]=0
test["Fatalities"]=0
test=test[['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',
       'Fatalities', 'ForecastId']]
#train+testデータを作る
whole = train.append(test, sort=False)
whole.head()
whole["Province_State"].fillna(whole["Country_Region"],inplace= True)
total_case_by_date = train.groupby(["Date"]).aggregate({"ConfirmedCases":np.sum}).reset_index()
#total_case_by_date.head()
total_case_by_date.plot(x="Date", y="ConfirmedCases")
total_case_by_Cuntry = train.groupby(["Country_Region"]).aggregate({"ConfirmedCases":np.sum})
#total_case_by_Cuntry.head()
#total_case_by_Cuntry.hist(bins=100)
total_case_by_Cuntry["ConfirmedCases"].plot(x="Country_Region",y="ConfirmedCases")
train2 = whole[:19404]
test2 = whole[19404:]
whole2 = pd.get_dummies(data=whole, columns=["Province_State","Country_Region"])
#標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cc_df = whole[["ConfirmedCases"]]
cc_std = scaler.fit_transform(cc_df)
cc_std
len(cc_std)
whole["cc_std"]=cc_std
whole.head()
target_col = "ConfirmedCases"
exclude_cols =['Id',"ForecastId","ConfirmedCases","Fatalities","cc_std"]
feature_cols =[]
for col in whole2.columns:
    if col not in exclude_cols:
        feature_cols.append(col)
#feature_cols
whole3 = whole2[:19698]
whole4 = whole2[19698:]
X_train = np.array(whole3[feature_cols])
y_train = np.array(whole3[target_col])
X_val = np.array(whole4[feature_cols])
y_val = np.array(whole4[target_col])
len(X_train)
len(train)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 線形回帰モデルのライブラリ
from sklearn.metrics import mean_squared_error  # MSEのライブラリ
from sklearn.ensemble import RandomForestRegressor
# Enter your code here
rf = RandomForestRegressor(n_estimators=100, random_state=1234)
rf.fit(X_train, y_train)
y_pred_cc = rf.predict(X_val)
rf_mse = mean_squared_error(y_val, y_pred_cc)
print('RandomForestRegressor RMSE: ', round(np.sqrt(rf_mse), 3))
rf_importances = pd.DataFrame(rf.feature_importances_, columns=['importance'], index=feature_cols)
rf_importances.sort_values('importance', ascending=False).iloc[0:15].plot(kind='barh')
print('ランダムフォレストのRMSE: ', round(np.sqrt(rf_mse), 3))
#predict Fatalities

target_col = "Fatalities"
exclude_cols =['Id',"ForecastId","ConfirmedCases","Fatalities","cc_std"]
feature_cols =[]
for col in whole2.columns:
    if col not in exclude_cols:
        feature_cols.append(col)

rf.fit(X_train, y_train)
y_pred_f = rf.predict(X_val)
rf_mse = mean_squared_error(y_val, y_pred_f)
print('RandomForestRegressor RMSE: ', round(np.sqrt(rf_mse), 3))
submission_data = whole4[['ForecastId','ConfirmedCases','Fatalities' ]]
submission_data.loc[:,"ConfirmedCases"] = y_pred_cc
submission_data.loc[:,"Fatalities"] = y_pred_f
submission_data.to_csv("submission.csv", index=False)
submission_data.head()
