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



world_confirm=pd.read_csv("/kaggle/input/csse-covid-19-data/time_series_19-covid-Confirmed.csv")

world_confirm["Province/State"]=world_confirm["Province/State"].fillna("/")

world_confirm.assign(area=0)

world_confirm["area"]=world_confirm["Province/State"]+"_"+world_confirm["Country/Region"]





clms=['Province/State', 'Country/Region', 'Lat', 'Long','code','area']

country_data=pd.DataFrame(columns=clms)



country_data["Province/State"]=world_confirm["Province/State"]

country_data["Country/Region"]=world_confirm["Country/Region"]

country_data["Lat"]=world_confirm["Lat"]

country_data["Long"]=world_confirm["Long"]

cd=np.arange(len(country_data))

country_data["code"]=cd

country_data["area"]=world_confirm["area"]



world_confirm=world_confirm.set_index("area")

world_confirm=pd.merge(world_confirm,country_data)
df=world_confirm.sort_values(by="3/19/20",ascending=False)

df=df.head(30)

top30list=df.index
del world_confirm["Province/State"]

del world_confirm["Country/Region"]

del world_confirm["Lat"]

del world_confirm["Long"]

del world_confirm["area"]

world_confirm=world_confirm.set_index(["code"])
world_confirm_date=pd.DataFrame(columns=["code","date","confirm"])





for num in range(len(world_confirm)):

#for num in range(1):

    wcT=world_confirm[world_confirm.index==num].T

    wcT=wcT.reset_index()

    wcT=wcT.assign(code=num)

    wcT.columns=(["date","confirm","code"])

    wcT["date"]=pd.to_datetime(wcT["date"])

    

    wcT["confirm"]=wcT["confirm"].replace(0,np.nan)

#    wcT=wcT.dropna()



    wcT["bconfirm"]=wcT["confirm"].shift()

    wcT["bconfirm"]=wcT["bconfirm"].fillna(0)

    wcT["bconfirm"]=wcT["bconfirm"].astype("int")

    wcT["delta_c"]=wcT["confirm"]-wcT["bconfirm"]

    wcT["bdelta_c1"]=wcT["delta_c"].shift()

    wcT["bdelta_c1"]=wcT["bdelta_c1"].fillna(0)

    wcT["bdelta_c1"]=wcT["bdelta_c1"].astype("int")

    wcT["bdelta_c2"]=wcT["delta_c"].shift(2)

    wcT["bdelta_c2"]=wcT["bdelta_c2"].fillna(0)

    wcT["bdelta_c2"]=wcT["bdelta_c2"].astype("int")

    wcT["bdelta_c3"]=wcT["delta_c"].shift(3)

    wcT["bdelta_c3"]=wcT["bdelta_c3"].fillna(0)

    wcT["bdelta_c3"]=wcT["bdelta_c3"].astype("int")

    wcT["con_dbratio"]=(wcT["delta_c"]+wcT["bdelta_c1"]+wcT["bdelta_c2"])/(wcT["bdelta_c1"]+wcT["bdelta_c2"]+wcT["bdelta_c3"])



    wcT1=pd.DataFrame(columns=wcT.columns)

    wcT2=pd.DataFrame(columns=wcT.columns)

    wcT3=pd.DataFrame(columns=wcT.columns)



    wcT=wcT.fillna(0)

    wcT=wcT.replace(np.inf,1)

    wcT=wcT.replace(-np.inf,-1)

    

    wcT["con_dbratio1"]=wcT["con_dbratio"].shift()

    wcT1["con_dbratio"]=wcT["con_dbratio"].shift()

    wcT2["con_dbratio"]=wcT["con_dbratio"].shift(2)

    wcT3["con_dbratio"]=wcT["con_dbratio"].shift(3)

    wcT["3days_con_dbratio"]=(wcT1["con_dbratio"]+wcT2["con_dbratio"]+wcT3["con_dbratio"])/3

    wcT1["3days_con_dbratio"]=wcT["3days_con_dbratio"].shift()

    wcT["delta_3d_con_dbr"]=(wcT["3days_con_dbratio"]/wcT1["3days_con_dbratio"])



    wcT=wcT.fillna(0)

    wcT=wcT.replace(np.inf,1)

    wcT=wcT.replace(-np.inf,-1)

    

    

#    wcT["bdelta_c"]=wcT["delta_c"].shift()

#    wcT["bdelta_c"]=wcT["bdelta_c"].fillna(0)

#    wcT["bdelta_c"]=wcT["bdelta_c"].astype("int")

    

#    wcT["bddelta_c"]=wcT["ddelta_c"].shift()

#    wcT["bddelta_c"]=wcT["bddelta_c"].fillna(0)

#    wcT["bddelta_c"]=wcT["bddelta_c"].astype("int")

    

#    del wcT["bconfirm"]

#    del wcT["bdelta_c"]



#    wcT["delta_c"]=wcT["delta_c"].shift()

#    wcT["delta_c"]=wcT["delta_c"].fillna(0)

#    wcT["delta_c2"]=wcT["delta_c"].shift()

#    wcT["delta_c2"]=wcT["delta_c2"].fillna(0)

#    wcT["delta_c3"]=wcT["delta_c2"].shift()

#    wcT["delta_c3"]=wcT["delta_c3"].fillna(0)

#    wcT["delta_c4"]=wcT["delta_c3"].shift()

#    wcT["delta_c4"]=wcT["delta_c4"].fillna(0)

#    wcT["delta_cavg"]=(wcT["delta_c"]+wcT["delta_c2"]+wcT["delta_c3"]+wcT["delta_c4"])/4

    

#    wcT["ddelta_c"]=wcT["ddelta_c"].shift()

#    wcT["ddelta_c"]=wcT["ddelta_c"].fillna(0)

    

    world_confirm_date=world_confirm_date.append(wcT)



world_confirm_date=world_confirm_date.reset_index()

del world_confirm_date["index"]



world_confirm_date.to_csv("wcd.csv",sep=",",index=False)

world_data=pd.merge(world_confirm_date,country_data)

world_death=pd.read_csv("/kaggle/input/csse-covid-19-data/time_series_19-covid-Deaths.csv")

world_death["Province/State"]=world_death["Province/State"].fillna("/")

world_death.assign(area=0)

world_death["area"]=world_death["Province/State"]+"_"+world_death["Country/Region"]

world_death=pd.merge(world_death,country_data)



del world_death["Province/State"]

del world_death["Country/Region"]

del world_death["Lat"]

del world_death["Long"]

del world_death["area"]

world_death=world_death.set_index(["code"])
world_death_date=pd.DataFrame(columns=["code","date","death"])



for num in range(len(world_death)):

#for num in range(1):

    wdT=world_death[world_death.index==num].T

    wdT=wdT.reset_index()

    wdT=wdT.assign(code=num)

    wdT.columns=(["date","death","code"])

    wdT["date"]=pd.to_datetime(wdT["date"])



    wdT["death"]=wdT["death"].replace(0,np.nan)

#    wdT=wdT.dropna()

    

    wdT["bdeath"]=wdT["death"].shift()

    wdT["bdeath"]=wdT["bdeath"].fillna(0)

    wdT["bdeath"]=wdT["bdeath"].astype("int")

    wdT["delta_d"]=wdT["death"]-wdT["bdeath"]

    wdT["bdelta_d1"]=wdT["delta_d"].shift()

    wdT["bdelta_d1"]=wdT["bdelta_d1"].fillna(0)

    wdT["bdelta_d1"]=wdT["bdelta_d1"].astype("int")

    wdT["bdelta_d2"]=wdT["delta_d"].shift(2)

    wdT["bdelta_d2"]=wdT["bdelta_d2"].fillna(0)

    wdT["bdelta_d2"]=wdT["bdelta_d2"].astype("int")

    wdT["bdelta_d3"]=wdT["delta_d"].shift(3)

    wdT["bdelta_d3"]=wdT["bdelta_d3"].fillna(0)

    wdT["bdelta_d3"]=wdT["bdelta_d3"].astype("int")

    wdT["det_dbratio"]=(wdT["delta_d"]+wdT["bdelta_d1"]+wdT["bdelta_d2"])/(wdT["bdelta_d1"]+wdT["bdelta_d2"]+wdT["bdelta_d3"])





    wdT1=pd.DataFrame(columns=wcT.columns)

    wdT2=pd.DataFrame(columns=wcT.columns)

    wdT3=pd.DataFrame(columns=wcT.columns)



    wdT=wdT.fillna(0)

    wdT=wdT.replace(np.inf,1)

    wdT=wdT.replace(-np.inf,-1)

    

    wdT["det_dbratio1"]=wdT["det_dbratio"].shift()

    wdT1["det_dbratio"]=wdT["det_dbratio"].shift()

    wdT2["det_dbratio"]=wdT["det_dbratio"].shift(2)

    wdT3["det_dbratio"]=wdT["det_dbratio"].shift(3)

    wdT["3days_det_dbratio"]=(wdT1["det_dbratio"]+wdT2["det_dbratio"]+wdT3["det_dbratio"])/3

    wdT1["3days_det_dbratio"]=wdT["3days_det_dbratio"].shift()

    wdT["delta_3d_det_dbr"]=(wdT["3days_det_dbratio"]/wdT1["3days_det_dbratio"])



    wdT=wdT.fillna(0)

    wdT=wdT.replace(np.inf,1)

    wdT=wdT.replace(-np.inf,-1)



    

    

#    wdT["b_det_dbr"]=wdT["det_dbratio"].shift()

    

#    wdT=wdT.fillna(0)

#    wdT=wdT.replace(np.inf,10)



    

#    wdT["bdelta_d"]=wdT["delta_d"].shift()

#    wdT["bdelta_d"]=wdT["bdelta_d"].fillna(0)

#    wdT["bdelta_d"]=wdT["bdelta_d"].astype("int")



#    wdT["ddelta_d"]=wdT["delta_d"]-wdT["bdelta_d"]

#    wdT["bddelta_d"]=wdT["ddelta_d"].shift()

#    wdT["bddelta_d"]=wdT["bddelta_d"].fillna(0)

#    wdT["bddelta_d"]=wdT["bddelta_d"].astype("int")

    

#    del wdT["bdeath"]

#    del wdT["bdelta_d"]



#    wdT["delta_d"]=wdT["delta_d"].shift()

#    wdT["delta_d"]=wdT["delta_d"].fillna(0)

#    wdT["delta_d2"]=wdT["delta_d"].shift()

#    wdT["delta_d2"]=wdT["delta_d2"].fillna(0)

#    wdT["delta_d3"]=wdT["delta_d2"].shift()

#    wdT["delta_d3"]=wdT["delta_d3"].fillna(0)

#    wdT["delta_d4"]=wdT["delta_d3"].shift()

#    wdT["delta_d4"]=wdT["delta_d4"].fillna(0)

#    wdT["delta_davg"]=(wdT["delta_d"]+wdT["delta_d2"]+wdT["delta_d3"]+wdT["delta_d4"])/4





    

    

    world_death_date=world_death_date.append(wdT)



world_death_date=world_death_date.reset_index()

del world_death_date["index"]

#world_death_date.columns=(["code","date","death"])



world_data=pd.merge(world_data,world_death_date)
world_recover=pd.read_csv("/kaggle/input/csse-covid-19-data/time_series_19-covid-Recovered.csv")

world_recover["Province/State"]=world_recover["Province/State"].fillna("/")

world_recover.assign(area=0)

world_recover["area"]=world_recover["Province/State"]+"_"+world_recover["Country/Region"]

world_recover=pd.merge(world_recover,country_data)



del world_recover["Province/State"]

del world_recover["Country/Region"]

del world_recover["Lat"]

del world_recover["Long"]

del world_recover["area"]

world_recover=world_recover.set_index(["code"])
world_recover_date=pd.DataFrame(columns=["code","date","recover"])



for num in range(len(world_recover)):

#for num in range(1):

    wrT=world_recover[world_recover.index==num].T

    wrT=wrT.reset_index()

    wrT=wrT.assign(code=num)

    wrT.columns=(["date","recover","code"])



    wrT["date"]=pd.to_datetime(wrT["date"])



    wrT["brecover"]=wrT["recover"].shift()

    wrT["brecover"]=wrT["brecover"].fillna(0)

    wrT["brecover"]=wrT["brecover"].astype("int")

    wrT["delta_r"]=wrT["recover"]-wrT["brecover"]

    wrT["bdelta_r1"]=wrT["delta_r"].shift()

    wrT["bdelta_r1"]=wrT["bdelta_r1"].fillna(0)

    wrT["bdelta_r1"]=wrT["bdelta_r1"].astype("int")

    wrT["bdelta_r2"]=wrT["delta_r"].shift(2)

    wrT["bdelta_r2"]=wrT["bdelta_r2"].fillna(0)

    wrT["bdelta_r2"]=wrT["bdelta_r2"].astype("int")

    wrT["bdelta_r3"]=wrT["delta_r"].shift(3)

    wrT["bdelta_r3"]=wrT["bdelta_r3"].fillna(0)

    wrT["bdelta_r3"]=wrT["bdelta_r3"].astype("int")

    wrT["rec_dbratio"]=(wrT["delta_r"]+wrT["bdelta_r1"]+wrT["bdelta_r2"])/(wrT["bdelta_r1"]+wrT["bdelta_r2"]+wrT["bdelta_r3"])



    wrT1=pd.DataFrame(columns=wrT.columns)

    wrT2=pd.DataFrame(columns=wrT.columns)

    wrT3=pd.DataFrame(columns=wrT.columns)



    wrT=wrT.fillna(0)

    wrT=wrT.replace(np.inf,1)

    wrT=wrT.replace(-np.inf,-1)

    

    wrT["rec_dbratio1"]=wrT["rec_dbratio"].shift()

    wrT1["rec_dbratio"]=wrT["rec_dbratio"].shift()

    wrT2["rec_dbratio"]=wrT["rec_dbratio"].shift(2)

    wrT3["rec_dbratio"]=wrT["rec_dbratio"].shift(3)

    wrT["3days_rec_dbratio"]=(wrT1["rec_dbratio"]+wrT2["rec_dbratio"]+wrT3["rec_dbratio"])/3

    wrT1["3days_rec_dbratio"]=wrT["3days_rec_dbratio"].shift()

    wrT["delta_3d_rec_dbr"]=(wrT["3days_rec_dbratio"]/wrT1["3days_rec_dbratio"])



    wrT=wrT.fillna(0)

    wrT=wrT.replace(np.inf,1)

    wrT=wrT.replace(-np.inf,-1)

    

#    wrT["b_rec_dbr"]=wrT["rec_dbratio"].shift()

    

#    wrT=wrT.fillna(0)

#    wrT=wrT.replace(np.inf,10)



    

#    wrT["bdelta_r"]=wrT["delta_r"].shift()

#    wrT["bdelta_r"]=wrT["bdelta_r"].fillna(0)

#    wrT["bdelta_r"]=wrT["bdelta_r"].astype("int")



#    wrT["ddelta_r"]=wrT["delta_r"]-wrT["bdelta_r"]

#    wrT["bddelta_r"]=wrT["ddelta_r"].shift()

#    wrT["bddelta_r"]=wrT["bddelta_r"].fillna(0)

#    wrT["bddelta_r"]=wrT["bddelta_r"].astype("int")



    

#    del wrT["brecover"]

#    del wrT["bdelta_r"]



#    wrT["delta_r"]=wrT["delta_r"].shift()

#    wrT["delta_r"]=wrT["delta_r"].fillna(0)

#    wrT["delta_r2"]=wrT["delta_r"].shift()

#    wrT["delta_r2"]=wrT["delta_r2"].fillna(0)

#    wrT["delta_r3"]=wrT["delta_r2"].shift()

#    wrT["delta_r3"]=wrT["delta_r3"].fillna(0)

#    wrT["delta_r4"]=wrT["delta_r3"].shift()

#    wrT["delta_r4"]=wrT["delta_r4"].fillna(0)

#    wrT["delta_ravg"]=(wrT["delta_r"]+wrT["delta_r2"]+wrT["delta_r3"]+wrT["delta_r4"])/4



    

    

    

    world_recover_date=world_recover_date.append(wrT)



world_recover_date=world_recover_date.reset_index()

del world_recover_date["index"]

#world_recover_date.columns=(["code","date","recover"])



world_data=pd.merge(world_data,world_recover_date)
world_data.to_csv("world_data.csv",sep=",",index=None)

wdata=world_data.loc[:,['code', 'date','Lat', 'Long','area',

                        'confirm', 'bconfirm', 'delta_c', 'bdelta_c1', 'con_dbratio','con_dbratio1', '3days_con_dbratio', 'delta_3d_con_dbr',

                        'death','bdeath', 'delta_d', 'bdelta_d1', 'det_dbratio', 'det_dbratio1', '3days_det_dbratio','delta_3d_det_dbr',

                        'recover', 'brecover', 'delta_r', 'bdelta_r1', 'rec_dbratio','rec_dbratio1', '3days_rec_dbratio', 'delta_3d_rec_dbr']]

wdata.assign(days=0)

start=wdata["date"][0]

wdata["days"]=(wdata["date"]-start).dt.days

wdata["remaining_patient"]=wdata["bconfirm"]-wdata["bdeath"]-wdata["brecover"]

#wdata["delta_rpavg"]=wdata["delta_cavg"]-wdata["delta_davg"]-wdata["delta_ravg"]

#wdata["delta_rp"]=wdata["delta_c"]-wdata["delta_d"]-wdata["delta_r"]
wdata.to_csv("world_data.csv",sep=",",index=None)
wdata=pd.DataFrame(wdata)

wdata["code"]=wdata["code"].astype("int")

wdata["confirm"]=wdata["confirm"].astype("int")

wdata["death"]=wdata["death"].astype("int")

wdata["recover"]=wdata["recover"].astype("int")

wdata["remaining_patient"]=wdata["remaining_patient"].astype("int")
for num in range(468):

    area_name=country_data[country_data["code"]==num]["area"].astype(str)

    filename=("cd_%d.csv" % num)

    wdata[wdata["code"]==num].to_csv(filename)
Train=pd.DataFrame(columns=wdata.columns)

for num in range(468):

    if num in top30list:

        print(num)

        print("\t",wdata[wdata["code"]==num])

        wd=wdata[wdata["code"]==num]

        Train=Train.append(wd)

Train=Train.reset_index()

del Train["index"]
Train=Train.fillna(0)
yc=Train["con_dbratio"]

yd=Train["det_dbratio"]

yr=Train["rec_dbratio"]
x=Train.copy()

del x["date"]

del x["Lat"]

del x["Long"]

del x["code"]

del x["days"]
xc=x.loc[:,["con_dbratio1","3days_con_dbratio","delta_3d_con_dbr"]]

xc=xc.dropna()



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(xc.values,yc,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape



from sklearn.ensemble import RandomForestRegressor as RFR

model_c = RFR(n_estimators=1000, max_depth=5, random_state=1)

model_c.fit(x_train, y_train)



from sklearn.metrics import accuracy_score



#print(model.coef_)

#print(model.intercept_)

#print(model.get_params())



#y_pred=mode_cl.predict(x_test)

#print(model.score(x_test,y_test))







# Feature Importance

fti =model_c.feature_importances_   



feature_names=np.array(xc.columns)

print('Feature Importances:')

for i, feat in enumerate(feature_names):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
xd=x.loc[:,["det_dbratio1","3days_det_dbratio","delta_3d_det_dbr"]]

xd=xd.dropna()



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(xd.values,yd.values,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape



from sklearn.ensemble import RandomForestRegressor as RFR

model_d = RFR(n_estimators=100, max_depth=5, random_state=1)

model_d.fit(x_train, y_train)

#print(model.predict(x_test))

#print(model.score(x_test,y_test))



from sklearn.metrics import accuracy_score



#print(model.coef_)

#print(model.intercept_)

#print(model.get_params())



#y_pred=model.predict(x_test)

#print(model.score(x_test,y_test))







# Feature Importance

fti =model_d.feature_importances_   



feature_names=np.array(xd.columns)

print('Feature Importances:')

for i, feat in enumerate(feature_names):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
xr=x.loc[:,["rec_dbratio1","3days_rec_dbratio","delta_3d_rec_dbr"]]

xr=xr.replace(np.inf,np.nan)

xr=xr.dropna()



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(xr.values,yr.values,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape



from sklearn.ensemble import RandomForestRegressor as RFR

model_r = RFR(n_estimators=100, max_depth=5, random_state=1)

model_r.fit(x_train, y_train)

#print(model.predict(x_test))

#print(model.score(x_test,y_test))



from sklearn.metrics import accuracy_score



#print(model.coef_)

#print(model.intercept_)

#print(model.get_params())



#y_pred=model.predict(x_test)

#print(model.score(x_test,y_test))







# Feature Importance

fti =model_r.feature_importances_   



feature_names=np.array(xr.columns)

print('Feature Importances:')

for i, feat in enumerate(feature_names):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
import pandas as pd

CATEST=pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")



CATEST["Province/State"]=CATEST["Province/State"].fillna("/")

CATEST.assign(area=0)

CATEST["area"]=CATEST["Province/State"]+"_"+CATEST["Country/Region"]

CATEST=pd.merge(CATEST,country_data)

CATEST["Date"]=pd.to_datetime(CATEST["Date"])

CATEST["days"]=(CATEST["Date"]-start).dt.days



CATEST=pd.merge(CATEST,country_data)



CATEST.columns=['ForecastId', 'Province/State', 'Country/Region', 'Lat', 'Long', 'date',

       'area', 'code', 'days']

CATEST=pd.merge(CATEST,wdata,on=('code','date','Lat','Long'),how='left')

del CATEST["days_y"]



CATEST.columns=['ForecastId', 'Province/State', 'Country/Region', 'Lat', 'Long', 'date',

       'area', 'code', 'days', 'area_y', 'confirm', 'bconfirm', 'delta_c',

       'bdelta_c1', 'con_dbratio', 'con_dbratio1', '3days_con_dbratio',

       'delta_3d_con_dbr', 'death', 'bdeath', 'delta_d', 'bdelta_d1',

       'det_dbratio', 'det_dbratio1', '3days_det_dbratio', 'delta_3d_det_dbr',

       'recover', 'brecover', 'delta_r', 'bdelta_r1', 'rec_dbratio',

       'rec_dbratio1', '3days_rec_dbratio', 'delta_3d_rec_dbr',

       'remaining_patient']

CATEST["remaining_patient"]=CATEST["bconfirm"]-CATEST["bdeath"]-CATEST["brecover"]

#CATEST["delta_rpavg"]=CATEST["delta_cavg"]-CATEST["delta_davg"]-CATEST["delta_ravg"]

#CATEST["ddelta_rp"]=CATEST["ddelta_c"]-CATEST["ddelta_d"]-CATEST["ddelta_r"]



import datetime

CATrain=Train[Train["code"]==100]

CATEST0=pd.DataFrame(columns=CATrain.columns)

for d in range(1,5):

    CAT0=CATrain[CATrain["date"]==CATEST["date"][0]-datetime.timedelta(days=d)]

    CATEST0=pd.concat([CATEST0,CAT0])

CATEST0=CATEST0.sort_values("date")
CATEST1=pd.concat([CATEST0,CATEST])

CATEST1=CATEST1.reset_index(drop=True)




TESTDATA=pd.DataFrame(columns=CATEST1.columns)

for num in range(5,len(CATEST1)):

    print("Num:",num)

    TESTDATA=CATEST1[CATEST1.index==num]

    B1DATA=CATEST1[CATEST1.index==num-1]

    B2DATA=CATEST1[CATEST1.index==num-2]

    B3DATA=CATEST1[CATEST1.index==num-3]

    B4DATA=CATEST1[CATEST1.index==num-4]

    B5DATA=CATEST1[CATEST1.index==num-5]





    

    b1confirm=B1DATA['confirm'].values

    b1con_dbr=B1DATA['con_dbratio'].values

    b1delta_c=B1DATA['delta_c'].values

    b2delta_c=B2DATA['delta_c'].values

    b3delta_c=B3DATA['delta_c'].values

    b4delta_c=B4DATA['delta_c'].values

    b5delta_c=B5DATA['delta_c'].values



    b1_3d_con_dbr=(b1delta_c+b2delta_c+b3delta_c)/(b2delta_c+b3delta_c+b4delta_c)

    b2_3d_con_dbr=(b2delta_c+b3delta_c+b4delta_c)/(b3delta_c+b4delta_c+b5delta_c)

    

#    b1ddelta_c=B1DATA['ddelta_c'].values

#    b2confirm=B2DATA['confirm'].values

#    b3confirm=B3DATA['confirm'].values

#    b4confirm=B4DATA['confirm'].values

    print("bcon",b1confirm)

    TESTDATA['bconfirm'][num]=b1confirm[0]

    TESTDATA['bdelta_c1'][num]=b1delta_c[0]

    TESTDATA['con_dbratio1'][num]=b1con_dbr[0]

    TESTDATA["3days_con_dbratio"][num]=b1_3d_con_dbr[0]

    TESTDATA["delta_3d_con_dbr"][num]=b1_3d_con_dbr[0]/b2_3d_con_dbr[0]

    #    TESTDATA['bdelta_c'][num]=b1delta_c[0]

#    TESTDATA['bddelta_c'][num]=b1ddelta_c[0]





    

    

    

#    TESTDATA['delta_cavg'][num]=(b1confirm[0]-b4confirm[0])/4

#    TESTDATA['ddelta_c'][num]=(b1confirm[0]-b3confirm[0])

#    TESTDATA['confirm'][num]=TESTDATA['confirm'][num]+TESTDATA['delta_cavg'][num]

    

    

    b1death=B1DATA['death'].values

    b1det_dbr=B1DATA['det_dbratio'].values

    b1delta_d=B1DATA['delta_d'].values

    b2delta_d=B2DATA['delta_d'].values

    b3delta_d=B3DATA['delta_d'].values

    b4delta_d=B4DATA['delta_d'].values

    b5delta_d=B5DATA['delta_d'].values



    b1_3d_det_dbr=(b1delta_d+b2delta_d+b3delta_d)/(b2delta_d+b3delta_d+b4delta_d)

    b2_3d_det_dbr=(b2delta_d+b3delta_d+b4delta_d)/(b3delta_d+b4delta_d+b5delta_d)



#    b1delta_d=B1DATA['delta_d'].values

#    b1ddelta_d=B1DATA['ddelta_d'].values

#    b2death=B2DATA['death'].values

#    b3death=B3DATA['death'].values

#    b4death=B4DATA['death'].values

    print("bdet",b1death)

    TESTDATA['bdeath'][num]=b1death[0]

    TESTDATA['bdelta_d1'][num]=b1delta_d[0]

    TESTDATA['det_dbratio1'][num]=b1det_dbr[0]

    TESTDATA["3days_det_dbratio"][num]=b1_3d_det_dbr[0]

    TESTDATA["delta_3d_det_dbr"][num]=b1_3d_det_dbr[0]/b2_3d_det_dbr[0]



#    TESTDATA['bdelta_d'][num]=b1delta_d[0]

#    TESTDATA['bddelta_d'][num]=b1ddelta_d[0]



#   TESTDATA['delta_davg'][num]=(b1death[0]-b4death[0])/4

#    TESTDATA['ddelta_d'][num]=(b1death[0]-b3death[0])

#    TESTDATA['death'][num]=TESTDATA['death'][num]+TESTDATA['delta_davg'][num]

    



    b1recover=B1DATA['recover'].values

    b1rec_dbr=B1DATA['rec_dbratio'].values

    b1delta_r=B1DATA['delta_r'].values

    b2delta_r=B2DATA['delta_r'].values

    b3delta_r=B3DATA['delta_r'].values

    b4delta_r=B4DATA['delta_r'].values

    b5delta_r=B5DATA['delta_r'].values



    b1_3d_rec_dbr=(b1delta_r+b2delta_r+b3delta_r)/(b2delta_r+b3delta_r+b4delta_r)

    b2_3d_rec_dbr=(b2delta_r+b3delta_r+b4delta_r)/(b3delta_r+b4delta_r+b5delta_r)





    

#    b1delta_r=B1DATA['delta_r'].values

#    b1ddelta_r=B1DATA['ddelta_r'].values

#    b2recover=B2DATA['recover'].values

#    b3recover=B3DATA['recover'].values

#    b4recover=B4DATA['recover'].values

    print("brec",b1recover)

    TESTDATA['brecover'][num]=b1recover[0]

    TESTDATA['bdelta_r1'][num]=b1delta_r[0]

    TESTDATA['rec_dbratio1'][num]=b1rec_dbr[0]

    TESTDATA["3days_rec_dbratio"][num]=b1_3d_rec_dbr[0]

    TESTDATA["delta_3d_rec_dbr"][num]=b1_3d_rec_dbr[0]/b2_3d_rec_dbr

#    TESTDATA['bdelta_r'][num]=b1delta_r[0]

#    TESTDATA['bddelta_r'][num]=b1ddelta_r[0]



    #    TESTDATA['delta_ravg'][num]=(b1recover[0]-b4recover[0])/4

#    TESTDATA['ddelta_r'][num]=(b1recover[0]-b3recover[0])

#    TESTDATA['recover'][num]=TESTDATA['recover'][num]+TESTDATA['delta_ravg'][num]





#    TESTDATA['remaining_patient'][num]=TESTDATA['bconfirm'][num]-TESTDATA['bdeath'][num]-TESTDATA['brecover'][num]

#    TESTDATA['delta_rpavg'][num]=TESTDATA['delta_cavg'][num]-TESTDATA['delta_davg'][num]-TESTDATA['delta_ravg'][num]

#    TESTDATA['ddelta_rp'][num]=TESTDATA['ddelta_c'][num]-TESTDATA['ddelta_d'][num]-TESTDATA['ddelta_r'][num]



    XC=TESTDATA[["con_dbratio1","3days_con_dbratio","delta_3d_con_dbr"]]

    XC=XC.fillna(0)

    XD=TESTDATA[["det_dbratio1","3days_det_dbratio","delta_3d_det_dbr"]]

    XD=XD.fillna(0)

    XR=TESTDATA[["rec_dbratio1","3days_rec_dbratio","delta_3d_rec_dbr"]]

    XR=XR.fillna(0)



    YC=model_c.predict(XC)

    YD=model_d.predict(XD)

    YR=model_r.predict(XR)

    

    TESTDATA["con_dbratio"][num]=YC[0]

    TESTDATA["det_dbratio"][num]=YD[0]

    TESTDATA["rec_dbratio"][num]=YR[0]

    TESTDATA["delta_c"][num]=TESTDATA["con_dbratio"][num]*TESTDATA["bdelta_c1"][num]

    TESTDATA["delta_d"][num]=TESTDATA["det_dbratio"][num]*TESTDATA["bdelta_d1"][num]

    TESTDATA["delta_r"][num]=TESTDATA["rec_dbratio"][num]*TESTDATA["bdelta_r1"][num]

    TESTDATA["confirm"][num]=TESTDATA["delta_c"][num]+TESTDATA["bconfirm"][num]

    TESTDATA["death"][num]=TESTDATA["delta_d"][num]+TESTDATA["bdeath"][num]

    TESTDATA["recover"][num]=TESTDATA["delta_r"][num]+TESTDATA["brecover"][num]



    CATEST1[CATEST1.index==num]=TESTDATA
CATEST1
YC,YD,YR
submission=pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")
submission["ConfirmedCases"]=CATEST1["confirm"][4:].values.astype("int")

submission["Fatalities"]=CATEST1["death"][4:].values.astype("int")
submission
submission.to_csv("submission.csv",sep=",",index=False)