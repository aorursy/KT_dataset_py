from argparse import Namespace

#There are 1115 stores. Select a  small sample to do experimentation on. Select all for full training.
num_sample_stores=1115

#The test set is 47 days. Normally use the last 47 days of the training data for validation. Se to 0 and use all data for traing when submitting to kaggle
valid_days=0

#Hyperparameters
s= Namespace( **{
    "l1":4497,
    "l2":2328,
    "ps1":0.2771132028380148,
    "ps2":0.15631474446268287,
    "emb_drop":0.14301109844119272,
    "batchsize":64,
    "lrate":0.0660858230905056,
    "lrate_ratio":9,
    "wd":0.17305139150930285,
    "l1epoch":4,
    "l2epoch":3,
    "l3epoch":8,
})

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fastai import *
from fastai.tabular import * 


#display results
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf

plotly.offline.init_notebook_mode(connected=False)
cf.go_offline()

%matplotlib inline
%reload_ext autoreload
%autoreload 2


pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 500)
!ls ../input/
path=Path("../input/rossmann-data-engineering/")
traindf=pd.read_feather(path/"train.feather")
testdf=pd.read_feather(path/"test.feather")
#Remove zero sales
traindata=traindf[traindf["Sales"]!=0]

#Select sample base num_sample_stores 
sample_stores=list(range(1,num_sample_stores+1)) 
traindata=traindata[traindata.Store.isin(sample_stores)].sample(frac=1,random_state=0).reset_index(drop=True) 
#Select size validation set based on valid_days variable 
from datetime import datetime, timedelta
valid_idx=traindata[traindata.Date>=(traindata.Date.max()- timedelta(days=valid_days))].index.tolist()
#Convert datetime columns to int64 for traning
datecols=traindata.select_dtypes(include="datetime").columns.tolist()
traindata[datecols]=traindata[datecols].astype("int64")
testdf[datecols]=testdf[datecols].astype("int64")
procs = [FillMissing, Categorify, Normalize]
dep_var = 'Sales'
#cont_names,cat_names= cont_cat_split(sample_train,dep_var="Sales")
cont_names=[
 'CompetitionDistance',
 'Week',
 'Day',
 'Dayofyear',
 'Elapsed',
 'ratio-sales-customer',
 'ratio-saturday-week',
 'ratio-sunday-week',
 'ratio-promo-nopromo',
 'Promo_thisweek',
 'Open_thisweek',
 'StateHolidayBool_thisweek',
 'SchoolHoliday_thisweek',
 'Promo_prevweek',
 'Open_prevweek',
 'StateHolidayBool_prevweek',
 'SchoolHoliday_prevweek',
 'Promo_nextweek',
 'Open_nextweek',
 'StateHolidayBool_nextweek',
 'SchoolHoliday_nextweek',
 'Promo2Days',
 'CompetitionDaysOpen',
 'trend',
 'trend_DE',
 'Max_Humidity',
 'Max_Wind_SpeedKm_h',
 'Mean_Humidity',
 'Mean_TemperatureC',
 'Max_TemperatureC_chnage',
 'Month_Sales_mean',
 'Year_Sales_mean',
 'Dayofweek_Sales_mean',
 'Dayofweek_promo_Sales_mean',
 'BeforeSchoolHoliday',
 'AfterSchoolHoliday',
 'BeforeClosed',
 'AfterClosed',
 'BeforePromo',
 'AfterPromo',
 'BeforeStateHolidayBool',
 'AfterStateHolidayBool',
 'Promo2ActiveMonthBool',
 'BeforePromo2ActiveMonthBool',
 'AfterPromo2ActiveMonthBool',
 'SchoolHoliday_fw',
 'StateHolidayBool_fw',
 'Promo_fw',
 'Closed_fw',
 'Promo2ActiveMonthBool_fw',
    'CompetitionOpenSince', 'Promo2Since'
]
cat_names=[
  'Store',  
  'DayOfWeek',
 'Open',
 'Promo',
 'StateHoliday',
 'SchoolHoliday',
 'StoreType',
 'Assortment',
 'Promo2',
 'PromoInterval',
 'Year',
 'Month',
 'Dayofweek',
 'Is_month_end',
 'Is_month_start',
 'Is_quarter_end',
 'Is_quarter_start',
 'Is_year_end',
 'Is_year_start',
 'Promo2SinceYear',
 'Promo2Na',
 'Events',
'Fog',
 'Hail',
 'Rain',
 'Snow',
 'Thunderstorm',
 'Quarter',
 'CompetitionOpenNA',
 'CompetitionDistanceNA',
 'CompetitionOpenSinceYear',
  'State'
]

max_log_y = np.log(np.max(traindata['Sales']))#*1.2
y_range = torch.tensor([0, max_log_y], device=defaults.device)
databunch = (TabularList.from_df(traindata, path="", cat_names=cat_names, cont_names=cont_names, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(testdf, path=path, cat_names=cat_names, cont_names=cont_names))
                .databunch())
databunch.batch_size=s.batchsize
learn = tabular_learner(databunch, layers=[s.l1,s.l2], ps=[s.ps1,s.ps2], emb_drop=s.emb_drop, y_range=y_range, metrics=exp_rmspe)
learn.model
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(s.l1epoch, s.lrate, wd=s.wd)
learn.fit_one_cycle(s.l2epoch, s.lrate/s.lrate_ratio, wd=s.wd)
learn.fit_one_cycle(s.l3epoch, s.lrate/(s.lrate_ratio*s.lrate_ratio), wd=s.wd)
valid_preds=learn.get_preds(DatasetType.Valid)
traindata["SalesPreds"]=pd.Series(index=traindata.iloc[valid_idx].index,data=np.exp(valid_preds[0].numpy().T[0]))
#Define error function
def rmspe_metric(act,pred):
       return np.sqrt(np.mean(((act-pred)/act)**2))
rmspe_metric(traindata.Sales,traindata.SalesPreds)
#Sort stores by how much error
store_rmspe=traindata.groupby(["Store"]).apply(lambda x:rmspe_metric(x.Sales,x.SalesPreds)).sort_values(ascending=False)
store_rmspe.iplot(kind="histogram")
store_rmspe[:10]
t=traindata.set_index("Date")
#Stores with most error
for store in store_rmspe.index[:4].tolist():
    t[t.Store==store][["Sales","SalesPreds"]].iplot(kind="bar",barmode="overlay",title="Store {}".format(store))
#Stores with least error
for store in store_rmspe.index[-4:].tolist():
    t[t.Store==store][["Sales","SalesPreds"]].iplot(kind="bar",barmode="overlay",title="Store {}".format(store))
test_preds=learn.get_preds(DatasetType.Test)
testdf["Sales"]=np.exp(test_preds[0].data).numpy().T[0]
testdf[["Id","Sales"]]=testdf[["Id","Sales"]].astype("int")
testdf[["Id","Sales"]].to_csv("rossmann_submission.csv",index=False)
#!kaggle competitions submit -c rossmann-store-sales -f rossmann_submission.csv -m "rossman with extra features"