!pip install chart_studio

!pip install windrose

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns

from windrose import WindroseAxes

import os

print(os.listdir("../input"))

%matplotlib inline

data_T_start=pd.read_csv("../input/wind-turbine-scada-dataset/T1.csv")

data_T_start.head()
turbine_no="T1" #for powercurve graph
data_T_start.info()
data1_T=data_T_start.copy()
data1_T.rename(columns={'LV ActivePower (kW)':'ActivePower(kW)',"Wind Speed (m/s)":"WindSpeed(m/s)","Wind Direction (°)":"Wind_Direction"},

                inplace=True)

data1_T.head()
data1_T.rename(columns={'Date/Time':'Time'},inplace=True)

data1_T.head()
#function for finding months

def find_month(x):

    if " 01 " in x:

        return "Jan"

    elif " 02 " in x:

        return "Feb"

    elif " 03 " in x:

        return "March"    

    elif " 04 " in x:

        return "April"    

    elif " 05 " in x:

        return "May"    

    elif " 06 " in x:

        return "June"    

    elif " 07 " in x:

        return "July"    

    elif " 08 " in x:

        return "August"    

    elif " 09 " in x:

        return "Sep"    

    elif " 10 " in x:

        return "Oct"    

    elif " 11 " in x:

        return "Nov"    

    else:

        return "Dec"    
#add months

data1_T['Month']=data1_T.Time.apply(find_month)
data1_T.Month.unique()
data1_T.head()
#function for rewriting wind speed for 0.5 intervals. 

#For example: wind speeds between 3.25 and 3.75 turns 3.5,wind speeds between 3.75 and 4.25 turns 4.0

def mean_speed(x):

    list=[]

    i=0.25

    while i<=25.5:

        list.append(i)

        i+=0.5

        

    for i in list:

        if x < i:

            x=i-0.25

            return x
#adding a new column as "mean_WindSpeed" with function mean_speed().

data1_T["mean_WindSpeed"]=data1_T["WindSpeed(m/s)"].apply(mean_speed)

data1_T.head()
#function for rewriting wind direction for 30 intervals. 

#For example: wind directions between 15 and 45 turns 30,wind speeds between 45 and 75 turns 60

def mean_direction(x):

    list=[]

    i=15

    while i<=375:

        list.append(i)

        i+=30

        

    for i in list:

        if x < i:

            x=i-15

            if x==360:

                return 0

            else:

                return x
#adding a new column as "mean_Direction" with function mean_direction().

data1_T["mean_Direction"]=data1_T["Wind_Direction"].apply(mean_direction)

data1_T.head()
#function for rewriting wind direction with letters. 

#For example: 0=N, 30=NNE 60=NEE etc.

def find_direction(x):

    if x==0:

        return "N"

    if x==30:

        return "NNE"

    if x==60:

        return "NEE" 

    if x==90:

        return "E" 

    if x==120:

        return "SEE" 

    if x==150:

        return "SSE" 

    if x==180:

        return "S" 

    if x==210:

        return "SSW" 

    if x==240:

        return "SWW" 

    if x==270:

        return "W" 

    if x==300:

        return "NWW" 

    if x==330:

        return "NNW"

  
#adding a new column as "Direction" with function find_direction().

data1_T["Direction"]=data1_T["mean_Direction"].apply(find_direction)

data1_T.head()
#Number of wind speed values between 3.5 and 25. 

len(data1_T["WindSpeed(m/s)"][(data1_T["WindSpeed(m/s)"]>3.5) & (data1_T["WindSpeed(m/s)"]<=25)])
#Values bigger than 25. 

data1_T["WindSpeed(m/s)"][data1_T["WindSpeed(m/s)"]>25].value_counts()
#Remove the data that wind speed is smaller than 3.5 and bigger than 25.5

#We do that because according to turbine power curve turbine works between these values.

data2_T=data1_T[(data1_T["WindSpeed(m/s)"]>3.5) & (data1_T["WindSpeed(m/s)"]<=25.5)]
#Number of values where wind speed is bigger than 3.5 and active power is zero. 

#If wind speed is bigger than 3.5 and active power is zero, this means turbine is out of order. we must eliminate these.

len(data2_T["ActivePower(kW)"][(data2_T["ActivePower(kW)"]==0)&(data2_T["WindSpeed(m/s)"]>3.5)])
#Eliminate datas where wind speed is bigger than 3.5 and active power is zero.

data3_T=data2_T[((data2_T["ActivePower(kW)"]!=0)&(data2_T["WindSpeed(m/s)"]>3.5)) | (data2_T["WindSpeed(m/s)"]<=3.5)]
#Number of values

len(data3_T["WindSpeed(m/s)"])
data3_T.head(10)
#the mean value of Nordex_Powercurve(kW) when mean_WindSpeed is 5.5

data3_T["Theoretical_Power_Curve (KWh)"][data3_T["mean_WindSpeed"]==5.5].mean()
#we create clean data and add a columns where calculating losses. 

#Loss is difference between the Nordex_Powercurve and ActivePower. 

data_T_clean=data3_T.sort_values("Time")

data_T_clean["Loss_Value(kW)"]=data_T_clean["Theoretical_Power_Curve (KWh)"]-data_T_clean["ActivePower(kW)"]

data_T_clean["Loss(%)"]=data_T_clean["Loss_Value(kW)"]/data_T_clean["Theoretical_Power_Curve (KWh)"]*100

#round the values to 2 digit.

data_T_clean=data_T_clean.round({'ActivePower(kW)': 2, 'WindSpeed(m/s)': 2, 'Theoretical_Power_Curve (KWh)': 2,

                                   'Wind_Direction': 2, 'Loss_Value(kW)': 2, 'Loss(%)': 2})



data_T_clean.head()
#creating summary speed dataframe from clean data.

DepGroupT_speed = data_T_clean.groupby("mean_WindSpeed")

data_T_speed=DepGroupT_speed.mean()

#removing the unnecessary columns.

data_T_speed.drop(columns={"WindSpeed(m/s)","Wind_Direction","mean_Direction"},inplace=True)

#creating a windspeed column from index values.

listTspeed_WS=data_T_speed.index.copy()

data_T_speed["WindSpeed(m/s)"]=listTspeed_WS

#changing the place of columns.

data_T_speed=data_T_speed[["WindSpeed(m/s)","ActivePower(kW)","Theoretical_Power_Curve (KWh)","Loss_Value(kW)","Loss(%)"]]

#changing the index numbers.

data_T_speed["Index"]=list(range(1,len(data_T_speed.index)+1))

data_T_speed.set_index("Index",inplace=True)

#rounding the values to 2 digit

data_T_speed=data_T_speed.round({"WindSpeed(m/s)": 1, 'ActivePower(kW)': 2, 'Theoretical_Power_Curve (KWh)': 2, 'Loss_Value(kW)': 2, 'Loss(%)': 2})

#creating a count column that shows the number of wind speed from clean data.

data_T_speed["count"]=[len(data_T_clean["mean_WindSpeed"][data_T_clean["mean_WindSpeed"]==i]) 

                        for i in data_T_speed["WindSpeed(m/s)"]]

data_T_speed
#create summary direction dataframe from clean data.

DepGroupT_direction = data_T_clean.groupby("Direction")

data_T_direction=DepGroupT_direction.mean()

#remove the unnecessary columns.

data_T_direction.drop(columns={"WindSpeed(m/s)","Wind_Direction"},inplace=True)

#create a column from index.

listTdirection_Dir=data_T_direction.index.copy()

data_T_direction["Direction"]=listTdirection_Dir

#change the name of mean_WindSpeed column as  WindSpeed.

data_T_direction["WindSpeed(m/s)"]=data_T_direction["mean_WindSpeed"]

data_T_direction.drop(columns={"mean_WindSpeed"},inplace=True)

#change the place of columns.

data_T_direction=data_T_direction[["Direction","mean_Direction","ActivePower(kW)","Theoretical_Power_Curve (KWh)","WindSpeed(m/s)",

                                     "Loss_Value(kW)","Loss(%)"]]

#change the index numbers.

data_T_direction["Index"]=list(range(1,len(data_T_direction.index)+1))

data_T_direction.set_index("Index",inplace=True)

#create a count column that shows the number of directions from clean data.

data_T_direction["count"]=[len(data_T_clean["Direction"][data_T_clean["Direction"]==i]) 

                        for i in data_T_direction["Direction"]]

#round the values to 2 digit

data_T_direction=data_T_direction.round({'WindSpeed(m/s)': 1,'ActivePower(kW)': 2, 'Theoretical_Power_Curve (KWh)': 2,

                                           'Loss_Value(kW)': 2, 'Loss(%)': 2})

#sort by mean_Direction

data_T_direction=data_T_direction.sort_values("mean_Direction")

data_T_direction.drop(columns={"mean_Direction"},inplace=True)
data_T_direction
#Drawing graph of mean powers according to wind direction.

def bar_graph():

    fig = plt.figure(figsize=(20,10))

    plt.bar(data_T_direction["Direction"],data_T_direction["Theoretical_Power_Curve (KWh)"],label="Theoretical Power Curve",align="edge",width=0.3)

    plt.bar(data_T_direction["Direction"],data_T_direction["ActivePower(kW)"],label="Actual Power Curve",align="edge",width=-0.3)

    plt.xlabel("Wind Direction")

    plt.ylabel("Power (kW)")

    plt.title("Wind Farm {} Mean Power Values vs Direction".format(turbine_no))

    plt.legend()

    plt.show()

bar_graph()
#creating summary direction total dataframe from direction data.

data_T_direction_total=data_T_direction.copy()

#removing the unnecessary columns.

data_T_direction_total.drop(columns={"count","ActivePower(kW)","Theoretical_Power_Curve (KWh)","Loss_Value(kW)","Loss(%)"},inplace=True)

#calculating the total values from direction data.

data_T_direction_total["Total_Generation(MWh)"]=data_T_direction["ActivePower(kW)"]*data_T_direction["count"]/6000

data_T_direction_total["Theoretical_PC_Total_Generation(MWh)"]=data_T_direction["Theoretical_Power_Curve (KWh)"]*data_T_direction["count"]/6000

data_T_direction_total["Total_Loss(MWh)"]=data_T_direction_total["Theoretical_PC_Total_Generation(MWh)"]-data_T_direction_total["Total_Generation(MWh)"]

data_T_direction_total["Loss(%)"]=data_T_direction_total["Total_Loss(MWh)"]/data_T_direction_total["Theoretical_PC_Total_Generation(MWh)"]*100

#rounding the values to 2 digit

data_T_direction_total=data_T_direction_total.round({'WindSpeed(m/s)': 1,'Total_Generation(MWh)': 2, 'Theoretical_PC_Total_Generation(MWh)': 2,

                                           'Total_Loss(MWh)': 2, 'Loss(%)': 2})

#changing the place of columns.

data_T_direction_total=data_T_direction_total[["Direction","Total_Generation(MWh)","Theoretical_PC_Total_Generation(MWh)","WindSpeed(m/s)",

                                     "Total_Loss(MWh)","Loss(%)"]]
data_T_direction_total
#Drawing graph of total generations according to wind direction.

def bar_graph():

    fig = plt.figure(figsize=(20,10))

    plt.bar(data_T_direction_total["Direction"],data_T_direction_total["Theoretical_PC_Total_Generation(MWh)"],label="Theoretical Power Curve",align="edge",width=0.3)

    plt.bar(data_T_direction_total["Direction"],data_T_direction_total["Total_Generation(MWh)"],label="Actual Power Curve",align="edge",width=-0.3)

    plt.xlabel("Wind Direction")

    plt.ylabel("Energy Generation (MWh)")

    plt.title("Wind Farm {} Total Energy Generation Values vs Direction".format(turbine_no))

    plt.legend()

    plt.show()

bar_graph()
#Drawing graph of total loss according to wind direction.

def bar_graph():

    fig = plt.figure(figsize=(20,10))

    plt.bar(data_T_direction_total["Direction"],data_T_direction_total["Total_Loss(MWh)"],

            label="Total_Loss(MWh)",align="center",width=0.5, color="red",picker=5)

    plt.xlabel("Wind Direction")

    plt.ylabel("Total Loss (MWh)")

    plt.title("Wind Farm {} Total Loss Values vs Direction".format(turbine_no))

    plt.legend()

    plt.show()

bar_graph()
#creating summary dataframes for all directions from clean data in a for loop as we did when creating direction dataframe.

list_data=[]

list_yon=["N","NNE","NEE","E","SEE","SSE","S","SSW","SWW","W","NWW","NNW"]

for i in range(0,12):

    data1T_A=data_T_clean[data_T_clean["Direction"]==list_yon[i]]

    #

    DepGroup_A = data1T_A.groupby("mean_WindSpeed")

    data_T_A=DepGroup_A.mean()

    #

    data_T_A.drop(columns={"WindSpeed(m/s)","Wind_Direction","mean_Direction"},inplace=True)

    #

    listTA_WS=data_T_A.index.copy()

    data_T_A["WindSpeed(m/s)"]=listTA_WS

    #

    data_T_A=data_T_A[["WindSpeed(m/s)","ActivePower(kW)","Theoretical_Power_Curve (KWh)","Loss_Value(kW)","Loss(%)"]]

    #

    data_T_A["Index"]=list(range(1,len(data_T_A.index)+1))

    data_T_A.set_index("Index",inplace=True)

    #

    data_T_A=data_T_A.round({'ActivePower(kW)': 2, 'Theoretical_Power_Curve (KWh)': 2, 'Loss_Value(kW)': 2, 'Loss(%)': 2})

    #

    data_T_A["count"]=[len(data1T_A["mean_WindSpeed"][data1T_A["mean_WindSpeed"]==x]) 

                            for x in data_T_A["WindSpeed(m/s)"]]

    list_data.append(data_T_A)

    

data_T_N=list_data[0]

data_T_NNE=list_data[1]

data_T_NEE=list_data[2]

data_T_E=list_data[3]

data_T_SEE=list_data[4]

data_T_SSE=list_data[5]

data_T_S=list_data[6]

data_T_SSW=list_data[7]

data_T_SWW=list_data[8]

data_T_W=list_data[9]

data_T_NWW=list_data[10]

data_T_NNW=list_data[11]
#Drawing power curve of the turbine.

def graph_WT():

    fig = plt.figure(figsize=(20,10))

    plt.plot(data_T_speed["WindSpeed(m/s)"],data_T_speed["Theoretical_Power_Curve (KWh)"],label="Theoretical Power Curve",

             marker="o",markersize=10,linewidth = 5)

    plt.plot(data_T_speed["WindSpeed(m/s)"],data_T_speed["ActivePower(kW)"],label="Actual Power Curve",

             marker="o",markersize=10,linewidth = 5)

    plt.xlabel("Wind Speed (m/s)")

    plt.ylabel("Power (kW)")

    plt.title("Wind Farm {} Power Curve".format(turbine_no))

    plt.legend()

    plt.show()

    fig.savefig("{}_Powercurve.png".format(turbine_no))

    plt.close(fig)

        

graph_WT()
#Drawing power curve of the turbine for all directions.

list_table=[data_T_N,data_T_NNE,data_T_NEE,data_T_E,data_T_SEE,data_T_SSE,data_T_S,

            data_T_SSW,data_T_SWW,data_T_W,data_T_NWW,data_T_NNW]



list_tableName=["N","NNE","NEE","E","SEE","SSE","S","SSW","SWW","W","NWW","NNW"]



def graph_T(i):

    fig = plt.figure(figsize=(20,10))  

    plt.plot(list_table[i]["WindSpeed(m/s)"],list_table[i]["Theoretical_Power_Curve (KWh)"],label="Theoretical Power Curve",

             marker="o",markersize=10,linewidth = 5)

    plt.plot(list_table[i]["WindSpeed(m/s)"],list_table[i]["ActivePower(kW)"],label="Actual Power Curve",

             marker="o",markersize=10,linewidth = 5)

    plt.xlabel("Wind Speed (m/s)")

    plt.ylabel("Power (kW)")

    plt.title("Wind Farm {} Power Curve According to {} Wind".format(turbine_no,list_tableName[i]))

    plt.legend()

    plt.show()

    fig.savefig("{}_{}_Powercurve.jpeg".format(turbine_no,list_tableName[i]))

    plt.close(fig)



# "N"=0, "NNE"=1,"NEE"=2,"E"=3,"SEE"=4,"SSE"=5,"S"=6,"SSW"=7,"SWW"=8,"W"=9,"NWW"=10,"NNW"=11



for i in range(0,12):

    graph_T(i)
#plotting the data distribution

plt.figure(figsize=(10, 8))

for i in range(4):

    plt.subplot(2, 2, i+1)

    sns.kdeplot(data_T_start.iloc[:,i+1], shade=True)

    plt.title(data_T_start.columns[i+1])

plt.tight_layout()

plt.show()
ax = WindroseAxes.from_ax()

ax.bar(data_T_start['Wind Direction (°)'], data_T_start['Wind Speed (m/s)'], normed=True, opening=0.8, edgecolor='white')

ax.set_legend()

plt.show()
import datetime

import time
data=pd.read_csv('../input/wind-turbine-scada-dataset/T1.csv')

data.head()
data.rename(columns={'Theoretical_Power_Curve (KWh)':'Power','LV ActivePower (kW)':'ActivePower',"Wind Speed (m/s)":"WindSpeed","Wind Direction (°)":"WindDirection"},

                inplace=True)

data.head()
data['Time']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[4])

data.head()
temp=data['Time'][1:].values - data['Time'][0:-1].values

temp=np.array([0]+list(temp))
data['TimeDiff']=temp
data.tail()
data['Gust']=np.array([0]+list(data['WindSpeed'][1:].values-data['WindSpeed'][:-1].values))

data.head()
data.dtypes
data_=data[data['TimeDiff'].isin([10,-50])]

data_.head()
len(data_)
data_=data_[data_['ActivePower']>=0]

data_.describe()
x=data_.sort_values(by='WindSpeed')['WindSpeed']

y=data_.sort_values(by='WindSpeed')['Power']

z=data_.sort_values(by='WindSpeed')['Gust']



plt.plot(x,y)

plt.plot(z,y)
x_train=data_[['WindSpeed','WindDirection','Gust']].values

y_train=data_['ActivePower'].values
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.svm import SVR

from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, StackingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error,r2_score

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostRegressor

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingRegressor
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=15).get_n_splits(data_.values)

    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.5))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score
HistGB=HistGradientBoostingRegressor(learning_rate=0.18,max_iter=100,max_leaf_nodes=4)



score = rmsle_cv(HistGB)

print("Hist Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score


KNR=KNeighborsRegressor(n_neighbors=8, weights='uniform', algorithm='auto', leaf_size=100, p=1, metric='minkowski')



score = rmsle_cv(KNR)

print("KNR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score
model_xgb1 = xgb.XGBRegressor(learning_rate=2e-2, max_depth=4, 

                             min_child_weight=1.1, n_estimators=100,

                             reg_alpha=0.3, reg_lambda=0.7,

                             #subsample=0.5213, silent=1,

                             nthread = -1)
model_xgb1.fit(x_train,y_train)
preds=model_xgb1.predict(x_train)

score=mean_squared_error(y_train,preds)

score**0.5
r2_score(y_train,preds)
model_xgb1.predict(np.array([6.02,110.0,0.23]).reshape(1,-1))
model_lgb1 = lgb.LGBMRegressor(objective='regression',num_leaves=2,

                              learning_rate=0.07, n_estimators=80,

                              #max_bin = 255, bagging_fraction = 0.8,

                              #bagging_freq = 5, feature_fraction = 0.8,

                              #feature_fraction_seed=9, bagging_seed=9,

                              reg_alpha=0.3,reg_lambda=0.7,

                              #min_data_in_leaf =3, min_sum_hessian_in_leaf = 2

                             )
model_lgb1.fit(x_train,y_train)
preds=model_lgb1.predict(x_train)

score=mean_squared_error(y_train,preds)

score**0.5
r2_score(y_train,preds)
model_lgb1.predict(np.array([6.02,110.0,0.23]).reshape(1,-1))
test=[]

for i in range(1,6):

    test.append(pd.read_csv(f'../input/testwindpower/test{i}.csv'))

test=pd.concat(test,axis=0)

test.head()
test.sort_values(by='Date',inplace=True)

test.head()
test.columns=['Date','Speed','Direction','Energy']

test.head()
len(test)
for col in test.columns[1::2]:

    test[col]=test[col].apply(lambda x: float(x.replace(".","").replace(",",".")))
test.head()
test['Gust']=np.array([0]+list(test['Speed'][1:].values-test['Speed'][:-1]))

test.head()
x_val=test[['Speed','Direction','Gust']].values

y_val=test['Energy'].values

x_val.shape
preds1=model_xgb1.predict(x_val)

preds1.shape
preds2=model_lgb1.predict(x_val)

preds2.shape
score=mean_squared_error(y_val,preds1)

score**0.5
r2_score(y_val,preds1)
score=mean_squared_error(y_val,preds2)

score**0.5
r2_score(y_val,preds2)
sam=1000

f=2

plt.plot(range(sam,f*sam),y_val[sam:f*sam])

plt.plot(range(sam,f*sam),preds1[sam:f*sam])

plt.title('XGB')
sam=700

f=4

plt.plot(range(sam,f*sam),y_val[sam:f*sam])

plt.plot(range(sam,f*sam),preds2[sam:f*sam])

plt.title('LGB')