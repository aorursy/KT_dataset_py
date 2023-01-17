import re
import numpy as np
from datetime import datetime, timedelta

import pandas as pd
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 1000)

from fastai import *
from fastai.tabular import * 

from pathlib import Path
path=Path("../input/rossmann-store-sales/")
traindf=pd.read_csv(path/"train.csv",low_memory=False,parse_dates=["Date"])
traindf.shape
traindf.head()
traindf.dtypes
traindf.info()
testdf=pd.read_csv(path/"test.csv",low_memory=False,parse_dates=["Date"])
testdf.shape
testdf.info()
#Temporarliy combine test and train data. We do this do make feature engieering easier
data=traindf.append(testdf,sort=False)
data.sort_values(["Store","Date"],inplace=True,ascending=False)
data.reset_index(inplace=True,drop=True)
data.shape
#The test set is 47 days. Use the last 47 days of the training data for validation
# Set Sales and Customers to np.nan in the validation period set and add back before saving. This is to avoid overfitting from some of the feature engeering. For example calculating average sales also in the validation period
create_validation_set=False
if create_validation_set:
    validation_days=47
    valid_idx=data[(data.Date>=(traindf.Date.max()- timedelta(days=validation_days)))  & (data.Date<=traindf.Date.max())].index.tolist()
    valid_data=data.loc[valid_idx][["Sales","Customers"]]
    data.loc[valid_idx,["Sales","Customers"]]=np.nan
data.sort_values(["Store","Date"],inplace=True,ascending=False)
data.reset_index(inplace=True,drop=True)
#Set datatypes
data.Promo=data.Promo.astype(bool)
data.SchoolHoliday=data.SchoolHoliday.astype(bool)
#StateHoliday has values:['0', 'a','b', 'c']
data["StateHolidayBool"]=data.StateHoliday!='0'
data.StateHoliday=data.StateHoliday.astype("category")
data.dtypes
#Add extra date columns
add_datepart(data,"Date",drop=False)
data["Quarter"]=data.Date.dt.quarter.astype(np.int64)
data["DaysSince2010"]=(data["Date"] - pd.Timestamp('2010-1-1')).dt.days

#mondays of every week. Much safer to use week start in groupby than ["Year","Week"]. With ["Year","week"] you will get wrong results around newyear
data['Weekstart']=data['Date'] - pd.to_timedelta(arg=data['Date'].dt.weekday, unit='D') 
#There are some days where the store is open with zero sales. Probably a mistake
#This might be important for learning because there are higher sales before and after closed dates
#List mistakes
#data[(data.Open==1) & (data.Sales==0)] 
#fixing:
data.loc[data.Sales==0,"Open"]=0
data.Open=data.Open.astype(bool)

#Store 622 has some NaNs in test data, assume store is closed
data.loc[data.Open.isna(),"Open"]=False
#Also usefull to have the Closed column
data["Closed"]=~data["Open"]
store_dtypes= {
    "StoreType":"category",
    "Assortment":"category",
    "Promo2":"bool"
}

storedf=pd.read_csv(path/"store.csv",low_memory=False,dtype=store_dtypes)
len(storedf)
storedf.head()
storedf.info()
data=data.merge(storedf,how="left",on="Store")
len(data[data.Assortment.isnull()])
#Convert OpenSinceYear and OpenSinceMonth to one column with the date the competition open
data["CompetitionOpenSince"] = pd.to_datetime(dict(year=data.CompetitionOpenSinceYear,
                                                 month=data.CompetitionOpenSinceMonth, day=15))
#Number of days to or since competition open.
#Negative numbers if competition will open in the future
data["CompetitionDaysOpen"] = data.Date.subtract(data.CompetitionOpenSince).dt.days
#Create column that indicates that CompetitionOpenSince is missing
data["CompetitionOpenNA"]=False
data.loc[data.CompetitionOpenSinceYear.isna(),"CompetitionOpenNA"]=True

data["CompetitionDistanceNA"]=False
data.loc[data.CompetitionDistance.isna(),"CompetitionDistanceNA"]=True
#Fill missing values
#Assume that missing CompetitionOpenSince data is because no competition has opened yet and that they will open in 100 days
import datetime
from dateutil.relativedelta import relativedelta
CompetitionOpen = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(days=100)

data['CompetitionOpenSinceYear'] = data.CompetitionOpenSinceYear.fillna(CompetitionOpen.year).astype(np.int32)
data['CompetitionOpenSinceMonth'] = data.CompetitionOpenSinceMonth.fillna(CompetitionOpen.month).astype(np.int32)
data['CompetitionOpenSince'] = data.CompetitionOpenSince.fillna(CompetitionOpen)
data['CompetitionDaysOpen'] = data.CompetitionDaysOpen.fillna(100).astype(np.int32)

#Assume missing CompetitionDistance data is beacuse the competition is to far away to be registered
data.loc[data.CompetitionDistance.isna(),"CompetitionDistance"]= data.CompetitionDistance.max()*2  
# Create a categorical datapoint.
# Assume that events more than 12 months in the past or future has small effect.
# Reduce number of categories 
data["CompetitionMonthsOpen"] = data["CompetitionDaysOpen"]//30
data.loc[data.CompetitionMonthsOpen>12, "CompetitionMonthsOpen"] = 12
data.loc[data.CompetitionMonthsOpen<-12, "CompetitionMonthsOpen"] = -12
data["CompetitionYearsOpen"] = data["CompetitionDaysOpen"]//365
data.loc[data.CompetitionYearsOpen<-2, "CompetitionYearsOpen"] = -3

#data.CompetitionMonthsOpen.unique()
data["Promo2Na"]=False
data.loc[data.PromoInterval.isna(),"Promo2Na"]=True
                
#Fill missing values
#Assume that missing Promo2 data is because no competition has opened yet and that they will open in 100 days
from dateutil.relativedelta import relativedelta
Promo2Open = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(days=100)

#assume that missing data is because no promo2 has happend at this store yet
data['Promo2SinceYear'] = data.Promo2SinceYear.fillna(Promo2Open.year).astype(np.int32)
data['Promo2SinceWeek'] = data.Promo2SinceWeek.fillna(Promo2Open.isocalendar()[1]).astype(np.int32)
#Convert Promo2SinceYear and Promo2SinceWeek to one column with the date the competition open
# Adding -1 for day, this is required by to_datetime
dateYWw=data.Promo2SinceYear.astype(str) +"-" + data.Promo2SinceWeek.astype(str)+"-1"
data["Promo2Since"]=pd.to_datetime(dateYWw,format="%Y-%W-%w")
#Number of dayes to or since Promo2 started.
#Negative numbers if Promo2 will start in the future
data["Promo2Days"] = data.Date.subtract(data["Promo2Since"]).dt.days
# Create a categorical datapoint.
# Assume that events more than 24 weeks in the past or future has small effect.
data["Promo2Weeks"] = data["Promo2Days"]//7
data.loc[data.Promo2Weeks<-25, "Promo2Weeks"] = -25
data.loc[data.Promo2Weeks>25, "Promo2Weeks"] = 25
data["Promo2Years"] = data["Promo2Days"]//365
#Promo2 is only active in some months create column that reflects this
#  1,2,3: Promo2 active and in promointerval
#  0:     Promo2 active but not in promointerval
#  -1:    Promo2 not started for this store yet
#  -2:    Promo2 never active for this store

data["Promo2ActiveMonth"]=0
#Promo2 active
data.loc[data.Month.isin([1,4,7,10]) & (data.PromoInterval=="Jan,Apr,Jul,Oct"),"Promo2ActiveMonth"]=1
data.loc[data.Month.isin([2,5,8,11]) & (data.PromoInterval=="Feb,May,Aug,Nov"),"Promo2ActiveMonth"]=2
data.loc[data.Month.isin([3,6,9,12]) & (data.PromoInterval=="Mar,Jun,Sept,Dec"),"Promo2ActiveMonth"]=3
#Promo2 not started yet
data.loc[(data["Promo2Since"]>data["Date"]) & data["Promo2"],"Promo2ActiveMonth"]=-1
#Promo2 never active for this store
data.loc[data["Promo2"]==0,"Promo2ActiveMonth"]=-2
data.Promo2ActiveMonth=data.Promo2ActiveMonth.astype("category")
#Convert to category
data.PromoInterval=data.PromoInterval.astype("category")
data.dtypes
path=Path("../input/rossmann-store-extra/")
store_states=pd.read_csv(path/"store_states.csv",low_memory=False,dtype={"State":"category"})
store_states.shape
store_states.head()
data=data.merge(store_states,how="left",on="Store")
len(data[data.State.isnull()])
state_names=pd.read_csv(path/"state_names.csv",low_memory=False)
state_names.shape
state_names.head()
data=data.merge(state_names,how="left",on="State")
len(data[data.StateName.isnull()])
#Set column datatype
data.State=data.State.astype("category")
data.StateName=data.StateName.astype("category")
googletrend=pd.read_csv(path/"googletrend.csv",low_memory=False)
googletrend.shape
googletrend.head()
googletrend.info()
strdate=googletrend.week.str.split(' - ', expand=True)[0]
googletrend['Date']=pd.to_datetime(strdate, format='%Y-%m-%d')
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'
googletrend["Year"]=googletrend.Date.dt.year
googletrend["Week"]=googletrend.Date.dt.week

data=data.merge(googletrend[["State","Year","Week","trend"]],how="left",on=["State","Year", "Week"])
len(data[data.trend.isnull()])
#Extract data for Germany as a seperat column
trend_de = googletrend[googletrend.file == 'Rossmann_DE'][["Year","Week","trend"]]
trend_de.rename(columns={"trend":"trend_DE"},inplace=True)
trend_de.head()
data = data.merge(trend_de, 'left', ["Year", "Week"])
len(data[data.trend_DE.isnull()])
weather=pd.read_csv(path/"weather.csv",low_memory=False,parse_dates=["Date"])
weather.rename(columns={"file":"StateName"},inplace=True)
weather.head()
weather["Max_TemperatureC_chnage"]=weather.groupby("StateName")["Max_TemperatureC"].diff().fillna(0)
weather.info()
weather.Events.unique()
#Convert to one column per type
events_dummies=weather.Events.str.get_dummies(sep='-')
weather=weather.join(events_dummies)
#weather.drop("Events",inplace=True,axis=1) 
#Check missing
#weather[weather.Max_VisibilityKm.isna()].sort_values(["StateName","Date"])
#Only some dates in between that is missing. Filling with next day
weather.Max_VisibilityKm=weather.Max_VisibilityKm.fillna(method="ffill")
weather.Mean_VisibilityKm=weather.Mean_VisibilityKm.fillna(method="ffill")
weather.Min_VisibilitykM=weather.Min_VisibilitykM.fillna(method="ffill")
weather.CloudCover=weather.CloudCover.fillna(method="ffill")

#Lots of missing, most likley not usefull. Dropping
weather.drop("Max_Gust_SpeedKm_h",inplace=True,axis=1)
data = data.merge(weather,how="left", on=["StateName","Date"])
len(data[data.Mean_TemperatureC.isnull()])
def add_days_since_last_event_in_group(df,group,date,event,ascending=True,fillna=True,zero_on_event=False):
    """
    It is common when working with time series data to extract data that explains relationships 
    across rows as opposed to columns, e.g When there is an event it is usefull to track Time until 
    next event Time since last event
    
    https://stackoverflow.com/questions/45022226/find-days-since-last-event-pandas-dataframe
    
    This version is supposed to be equal to the the one developed by Jeremy Howard in the fast.ai course. 
    The only difference is that this version is vectorized so it runs a bit faster
        
    https://github.com/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb
    """
    
    #Basic error checking
    #if df[date].dtype != np.dtype('datetime64[ns]'):
    #    raise ("Date must be datetime64 object")
        
    #Set colname
    if ascending==True:
        colname= "Before" + event
    elif ascending==False:
        colname= "After"  + event
    else:
        raise ("Ascending must be a boolean")
        
    
    temp=df[[group,date,event]].sort_values([group,date],ascending=ascending)
    temp['days_since_last_record'] = temp.groupby(group)[date].diff().abs() #dt.days
    cumalative=temp[event].shift().cumsum()
    
    df.loc[:,colname]= temp.groupby(['Store', cumalative])['days_since_last_record'].cumsum()
    if fillna:
        df[colname].fillna(0,inplace=True)
    if zero_on_event:
        df.loc[df[event]==True,colname]=0
data["index"]=data.index
add_days_since_last_event_in_group(data,"Store","index","SchoolHoliday",ascending=True)
add_days_since_last_event_in_group(data,"Store","index","SchoolHoliday",ascending=False)

add_days_since_last_event_in_group(data,"Store","index","Closed",ascending=True)
add_days_since_last_event_in_group(data,"Store","index","Closed",ascending=False)

add_days_since_last_event_in_group(data,"Store","index","Promo",ascending=True)
add_days_since_last_event_in_group(data,"Store","index","Promo",ascending=False)

add_days_since_last_event_in_group(data,"Store","index","StateHolidayBool",ascending=True)
add_days_since_last_event_in_group(data,"Store","index","StateHolidayBool",ascending=False)
data.loc[:,"Promo2ActiveMonthBool"]=(data.Promo2ActiveMonth!=0)
add_days_since_last_event_in_group(data,"Store","index","Promo2ActiveMonthBool",ascending=True)
add_days_since_last_event_in_group(data,"Store","index","Promo2ActiveMonthBool",ascending=False)
#data.set_index("Date",inplace=True)
columns = ['SchoolHoliday', 'StateHolidayBool', 'Promo',"Closed","Promo2ActiveMonthBool"]
bwd = data[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum().drop("Store",axis=1).reset_index(0,drop=True)
data=data.join(bwd, rsuffix='_bw')
fwd = data[['Store']+columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum().drop("Store",axis=1).reset_index(0,drop=True)
data=data.join(fwd, rsuffix='_fw')
wsales=data[data.Sales>0]
month= wsales[["Month","Store","Sales"]].groupby(["Month","Store"],as_index=False).agg(["mean","median","std","max","min"])
month.columns=pd.Index(["Month_"+ a +"_" + b for a,b in month.columns.tolist()])
data=data.merge(month,how="left",on=["Month","Store"])
len(data[data.Month_Sales_mean.isnull()])
year= wsales[["Year","Store","Sales"]].groupby(["Year","Store"],as_index=False).agg(["mean","median","std","max","min"])
year.columns=pd.Index(["Year_"+ a +"_" + b for a,b in year.columns.tolist()])
data=data.merge(year,how="left",on=["Year","Store"])
len(data[data.Year_Sales_mean.isnull()])
Dayofweek= wsales[["Dayofweek","Year","Store","Sales"]].groupby(["Year","Dayofweek","Store"]).agg(["mean","median","std","max","min"])
dayofweek_cols=["Dayofweek_"+ a +"_" + b for a,b in Dayofweek.columns.tolist()]
Dayofweek.columns=pd.Index(dayofweek_cols)


Dayofweek.reset_index(inplace=True)
data=data.merge(Dayofweek,how="left",on=["Year","Dayofweek","Store"])
len(data[data.Dayofweek_Sales_mean.isnull()])
#Most days are never open on sundays, fill missing data with 0
data[dayofweek_cols]=data[dayofweek_cols].fillna(0)
len(data[data.Dayofweek_Sales_mean.isnull()])
Dayofweekpromo= wsales[["Year","Dayofweek","Store","Promo","Sales"]].groupby(["Year","Dayofweek","Promo","Store"]).agg(["mean","median","std","max","min"])
Dayofweekpromo_cols=["Dayofweek_promo_"+ a +"_" + b for a,b in Dayofweekpromo.columns.tolist()]
Dayofweekpromo.columns=pd.Index(Dayofweekpromo_cols)

Dayofweekpromo.reset_index(inplace=True)
data=data.merge(Dayofweekpromo,how="left",on=["Year","Dayofweek","Store","Promo"])
len(data[data.Dayofweek_promo_Sales_mean.isnull()])
#Most days are never open on sundays, fill missing data with 0
data[Dayofweekpromo_cols]=data[Dayofweekpromo_cols].fillna(0)
len(data[data.Dayofweek_promo_Sales_mean.isnull()])
agg=data[["Store","Sales","Customers"]].groupby("Store").sum()
sales_customer_ratio=(agg.Sales/agg.Customers).to_frame(name="ratio-sales-customer").reset_index()
data=data.merge(sales_customer_ratio,how="left",on="Store")
week=data[data.Dayofweek.isin([0,1,2,3,4])][["Sales","Store"]].groupby(["Store"]).mean()

saturday=data[data.Dayofweek==5][["Sales","Store"]].groupby(["Store"]).mean()
saturday_week=(saturday/week).reset_index().rename(columns={"Sales":"ratio-saturday-week"})
data=data.merge(saturday_week,how="left",on="Store")

sunday=data[data.Dayofweek==6][["Sales","Store"]].groupby(["Store"]).mean()
sunday_week=(sunday/week).reset_index().rename(columns={"Sales":"ratio-sunday-week"})
data=data.merge(sunday_week,how="left",on="Store")
promo=data[["Store","Sales","Promo"]].groupby(["Store","Promo"]).mean().reset_index().pivot_table(values="Sales",index="Store",columns="Promo")
promo_ratio=(promo[True]/promo[False]).to_frame(name="ratio-promo-nopromo")
data=data.merge(promo_ratio,how="left",on="Store")
#Todo
#Todo
thisweek=data[["Store","Weekstart","Promo","Open","StateHolidayBool","SchoolHoliday"]].groupby(["Weekstart","Store"]).agg({'Promo':'sum','Open':'sum','StateHolidayBool':'sum','SchoolHoliday':'sum'})
prevweek=thisweek.groupby("Store").shift(1).reset_index().fillna(method="bfill")
nextweek=thisweek.groupby("Store").shift(-1).reset_index().fillna(method="ffill")
thisweek.reset_index(inplace=True)
data=data.merge(thisweek,on=["Store","Weekstart"],how="left",suffixes=("","_thisweek"),)
data=data.merge(prevweek,on=["Store","Weekstart"],how="left",suffixes=("","_prevweek"))
data=data.merge(nextweek,on=["Store","Weekstart"],how="left",suffixes=("","_nextweek"))
if create_validation_set:
    data.loc[valid_idx,["Sales","Customers"]]=valid_data
    trainname="train-and-validation.feather"
else:
    trainname="train.feather"

    
testdata=data[data.Sales.isna()].reset_index(drop=True)
traindata=data[data.Sales.notna()].reset_index(drop=True)
traindata.to_feather(trainname)
testdata.to_feather("test.feather")