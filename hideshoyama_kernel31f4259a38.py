# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
train.columns
#train.assign(area=0)

train["Province_State"]=train["Province_State"].fillna("/")

train['area']=train["Province_State"]+"_"+train["Country_Region"]

train
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
test.assign(area=0)

test["Province_State"]=test["Province_State"].fillna("/")

test['area']=test["Province_State"]+"_"+test["Country_Region"]
test
train_arealist=train["area"].unique()

train_arealist
test_arealist=test["area"].unique()

test_arealist
set(train_arealist)==set(test_arealist)
for cd in range(len(train_arealist)):

    train.loc[train["area"]==train_arealist[cd],'code']=cd

train["code"]=train["code"].astype("int")
for cd in range(len(train_arealist)):

    test.loc[test["area"]==train_arealist[cd],'code']=cd

test["code"]=test["code"].astype("int")
test
train["ConfirmedCases"]=train["ConfirmedCases"].replace(0,np.NaN)

train=train.dropna()
for area in train_arealist:

    days=len(train[train["area"]==area])

#    print(days)

#    print(len(train[train["area"]==area]))

#    print(np.arange(days))

    train.loc[train["area"]==area,'days']=np.arange(days)

#    train[train["area"]==area,days]=np.arange(days)
train=train.assign(bcon=0.0)

for area in train_arealist:

    #    train.loc[train["area"]==train_arealist,"bcon"]=train[train["area"]==train_arealist,"ConfirmedCases"].shift()

#    train[train["area"]==area,"ConfirmedCases"].shift()

#    print(train[train["area"]==area])

    TR=train[train["area"]==area]

#    TR["bcon"]=TR["ConfirmedCases"].shift()

    train.loc[train["area"]==area,'bcon']=TR["ConfirmedCases"].shift()

    train['bcon']=train['bcon'].fillna(-9999)



train.assign(delta_con=0.0)

train["delta_con"]=train["ConfirmedCases"]/train["bcon"]

train["delta_con"]=train["delta_con"].fillna(0)



train=train.assign(bdcon=0)

train=train.assign(bcon3avg=0)

train=train.assign(bd_con3avg=0)

train=train.assign(bbd_con3avg=0)

train=train.assign(bdcon3avgratio=0.0)



for area in train_arealist:

    TR=train[train["area"]==area]

    TR["bdcon"]=TR["delta_con"].shift()

    TR["bdcon"]=TR["bdcon"].fillna(0)

    TR["bcon3avg"]=TR["bcon"].rolling(3).sum()/3

    TR["bd_con3avg"]=TR["bdcon"].rolling(3).sum()/3

    TR["bbd_con3avg"]=TR["bd_con3avg"].shift()

    TR["bdcon3avgratio"]=TR["bd_con3avg"]/TR["bbd_con3avg"]

    train.loc[train["area"]==area]=TR



del train["bcon3avg"]

del train["bd_con3avg"]

del train["bbd_con3avg"]

#train=train.dropna()
train=train.assign(bfat=0.0)

for area in train_arealist:

    #    train.loc[train["area"]==train_arealist,"bcon"]=train[train["area"]==train_arealist,"ConfirmedCases"].shift()

#    train[train["area"]==area,"ConfirmedCases"].shift()

#    print(train[train["area"]==area])

    TR=train[train["area"]==area]

#    TR["bcon"]=TR["ConfirmedCases"].shift()

    train.loc[train["area"]==area,'bfat']=TR["Fatalities"].shift()

    train['bfat']=train['bfat'].fillna(-9999)



train.assign(delta_con=0.0)

train["delta_fat"]=train["Fatalities"]/train["bfat"]

train["delta_fat"]=train["delta_fat"].fillna(0)

train["delta_fat"]=train["delta_fat"].replace(np.inf,1)





train=train.assign(bdfat=0)

train=train.assign(bfat3avg=0)

train=train.assign(bd_fat3avg=0)

train=train.assign(bbd_fat3avg=0)

train=train.assign(bdfat3avgratio=0.0)



for area in train_arealist:

    TR=train[train["area"]==area]

    TR["bdfat"]=TR["delta_fat"].shift()

    TR["bdfat"]=TR["bdfat"].fillna(-1)

    TR["bdfat"]=TR["bdfat"].replace(0,-1)

    TR["bfat3avg"]=TR["bfat"].rolling(3).sum()/3

    TR["bd_fat3avg"]=TR["bdfat"].rolling(3).sum()/3

    TR["bbd_fat3avg"]=TR["bd_fat3avg"].shift()

    TR["bdfat3avgratio"]=TR["bd_fat3avg"]/TR["bbd_fat3avg"]

    train.loc[train["area"]==area]=TR



#del train["bfat3avg"]

#del train["bd_fat3avg"]

#del train["bbd_fat3avg"]

train.info()
train=train.dropna()
train.tail(20)
train[train["area"]=="New York_US"]
TRC=train[["ConfirmedCases","bcon","delta_con","bdcon","bdcon3avgratio"]]

TRF=train[["Fatalities","bfat","delta_fat","bdfat","bdfat3avgratio"]]
xc=TRC.loc[:,["bdcon","bdcon3avgratio"]]

yc=TRC.loc[:,"delta_con"]



xc["bdcon"]=xc["bdcon"]-1

xc["bdcon3avgratio"]=xc["bdcon3avgratio"]-1

yc=yc-1



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(xc.values,yc,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape







import xgboost as xgb

from sklearn.metrics import mean_squared_error

# 学習

model_c = xgb.XGBRegressor(objective="reg:linear")

model_c.fit(x_train, y_train)



# 予測

y_pred = model_c.predict(x_test)

mse=mean_squared_error(y_test, y_pred)

print(mse)







# Feature Importance

fti =model_c.feature_importances_   



feature_names=np.array(xc.columns)

print('Feature Importances:')

for i, feat in enumerate(feature_names):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
xf=TRF.loc[:,["bdfat","bdfat3avgratio"]]

yf=TRF.loc[:,"delta_fat"]



xf["bdfat"]=xf["bdfat"]-1

xf["bdfat3avgratio"]=xf["bdfat3avgratio"]-1

yf=yf-1



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(xf.values,yf,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape







import xgboost as xgb

from sklearn.metrics import mean_squared_error

# 学習

model_f = xgb.XGBRegressor(objective="reg:linear")

model_f.fit(x_train, y_train)



# 予測

y_pred = model_f.predict(x_test)

mse=mean_squared_error(y_test, y_pred)

print(mse)







# Feature Importance

fti =model_f.feature_importances_   



feature_names=np.array(xf.columns)

print('Feature Importances:')

for i, feat in enumerate(feature_names):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
test=pd.merge(test,train,how="left")
OUTPUT=pd.DataFrame(columns=test.columns)

for area in train_arealist:

    print(area)

    TEST=test[test["area"]==area]

    TEST=TEST.reset_index()

    print(TEST)

    TEST["bdcon3avg"]=TEST["bdcon"].rolling(3).sum()/3



    for num in range(10,len(TEST)):

        TESTDATA=TEST[TEST.index==num]

        B1DATA=TEST[TEST.index==num-1]

        B2DATA=TEST[TEST.index==num-2]

        B3DATA=TEST[TEST.index==num-3]

        B4DATA=TEST[TEST.index==num-4]

        

        bcon=B1DATA["ConfirmedCases"].values

        TESTDATA["bcon"][num]=bcon[0]

        bdcon=B1DATA["delta_con"].values

        TESTDATA["bdcon"][num]=bdcon[0]

        bdcon3avgratio=(B1DATA["bdcon"][num-1]+B2DATA["bdcon"][num-2]+B3DATA["bdcon"][num-3])/(B2DATA["bdcon"][num-2]+B3DATA["bdcon"][num-3]+B4DATA["bdcon"][num-4])

        print("bdcon3avgratio",bdcon3avgratio)

        TESTDATA["bdcon3avgratio"]=bdcon3avgratio

        

        bfat=B1DATA["Fatalities"].values

        TESTDATA["bfat"][num]=bfat[0]

        bdfat=B1DATA["delta_fat"].values

        TESTDATA["bdfat"][num]=bdfat[0]

        bdfat3avgratio=(B1DATA["bdfat"][num-1]+B2DATA["bdfat"][num-2]+B3DATA["bdfat"][num-3])/(B2DATA["bdfat"][num-2]+B3DATA["bdfat"][num-3]+B4DATA["bdfat"][num-4])

        TESTDATA["bdfat3avgratio"]=bdfat3avgratio

        

        XC=TESTDATA[["bdcon","bdcon3avgratio"]].values

        XF=TESTDATA[["bdfat","bdfat3avgratio"]].values

        

        YC=model_c.predict(XC)

        YF=model_f.predict(XF)

        

        TESTDATA["delta_con"]=YC+1

        TESTDATA["delta_fat"]=YF+1

        TESTDATA["ConfirmedCases"]=TESTDATA["bcon"]*TESTDATA["delta_con"]

        TESTDATA["Fatalities"]=TESTDATA["bfat"]*TESTDATA["delta_fat"]

        

        TEST[TEST.index==num]=TESTDATA

    TEST=TEST.fillna(0)

    OUTPUT=pd.concat([OUTPUT,TEST])
OUTPUT
test.head(50)
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
submission["ConfirmedCases"]=OUTPUT["ConfirmedCases"].values.astype("int")

submission["Fatalities"]=OUTPUT["Fatalities"].values.astype("int")
submission
submission.to_csv("submission.csv",index=None)