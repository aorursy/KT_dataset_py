from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 500)

from fastai import *
from fastai.tabular import * 


#Using plotly for display results
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=False)

import cufflinks as cf
cf.go_offline()

%matplotlib inline
%reload_ext autoreload
%autoreload 2

#Read CSV
path=Path("../input/")
traindf=pd.read_csv(path/"train.csv",low_memory=False,parse_dates=["Date"])
testdf=pd.read_csv(path/"test.csv",low_memory=False,parse_dates=["Date"])

store_dtypes= {
    "StoreType":"category",
    "Assortment":"category",
    "Promo2":"bool"
}
storedf=pd.read_csv(path/"store.csv",low_memory=False,dtype=store_dtypes)

#Merge store table
testdf=testdf.merge(storedf,how="left",on="Store")
traindf=traindf.merge(storedf,how="left",on="Store")

#Add date columns
add_datepart(traindf,"Date",drop=False)
add_datepart(testdf,"Date",drop=False)
#Fast.ai is complaning beacause of missign data in test set. 
testdf.CompetitionDistance=testdf.CompetitionDistance.fillna(0)
#Zero sales are ignored in competition, and is causing problems with evaluation function
traindf=traindf[traindf["Sales"]!=0]
num_sample_stores=100

sample_stores=list(range(1,num_sample_stores+1))
sample_train=traindf[traindf.Store.isin(sample_stores)].sample(frac=1,random_state=0).reset_index(drop=True)
valid_days=47
valid_idx=sample_train[sample_train.Date>=(sample_train.Date.max()- timedelta(days=valid_days))].index.tolist()
procs = [FillMissing, Categorify, Normalize]
dep_var = 'Sales'
#cont_names,cat_names= cont_cat_split(sample_train,dep_var="Sales")
cont_names=['Store',
 'CompetitionDistance',
 'CompetitionOpenSinceMonth',
 'CompetitionOpenSinceYear',
 'Promo2SinceWeek',
 'Promo2SinceYear',
 'Week',
 'Day',
 'Dayofyear',
 'Elapsed']
cat_names=['DayOfWeek',
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
 'Is_year_start']
max_log_y = np.log(np.max(sample_train['Sales']))#*1.2
y_range = torch.tensor([0, max_log_y], device=defaults.device)

databunch = (TabularList.from_df(sample_train, path="", cat_names=cat_names, cont_names=cont_names, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .databunch())
learn = tabular_learner(databunch, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.08, y_range=y_range, metrics=exp_rmspe)
learn.model
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2, wd=0.2)
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.fit_one_cycle(5, 1e-4, wd=0.2)
valid_preds=learn.get_preds(DatasetType.Valid)
sample_train["SalesPreds"]=pd.Series(index=sample_train.iloc[valid_idx].index,data=np.exp(valid_preds[0].numpy().T[0]))
#Define error function
def rmspe_metric(act,pred):
       return np.sqrt(np.mean(((act-pred)/act)**2))
rmspe_metric(sample_train.Sales,sample_train.SalesPreds)
#Sort stores by how much error
store_rmspe=sample_train.groupby(["Store"]).apply(lambda x:rmspe_metric(x.Sales,x.SalesPreds)).sort_values(ascending=False)
store_rmspe.iplot(kind="histogram")
store_rmspe[:3]

t=sample_train.set_index("Date")
#Stores with most error
for store in store_rmspe.index[:4].tolist():
    t[t.Store==store][["Sales","SalesPreds"]].iplot(kind="bar",barmode="overlay",title="Store {}".format(store))
#Stores with least error
for store in store_rmspe.index[-4:].tolist():
    t[t.Store==store][["Sales","SalesPreds"]].iplot(kind="bar",barmode="overlay",title="Store {}".format(store))
full_train=traindf.sample(frac=1,random_state=0).reset_index(drop=True)
valid_idx=[] #No validation set. 
databunch = (TabularList.from_df(full_train, path="", cat_names=cat_names, cont_names=cont_names, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(testdf, path=path, cat_names=cat_names, cont_names=cont_names))
                .databunch())
learn = tabular_learner(databunch, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.08, y_range=y_range, metrics=exp_rmspe)
learn.fit_one_cycle(5, 1e-2, wd=0.2)
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.fit_one_cycle(5, 1e-4, wd=0.2)
test_preds=learn.get_preds(DatasetType.Test)
testdf["Sales"]=np.exp(test_preds[0].data).numpy().T[0]
testdf[["Id","Sales"]]=testdf[["Id","Sales"]].astype("int")
testdf[["Id","Sales"]].to_csv("rossmann_submission.csv",index=False)