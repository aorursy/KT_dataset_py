# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing libraries

import pandas as pd

import numpy as np



import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



import matplotlib.pyplot as plt

import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec

import seaborn as sns

%matplotlib inline



from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
# importing datasets

calender= pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

price= pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

train_val= pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

submission= pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
calender.head()
import pandas_profiling as npp

profile = npp.ProfileReport(calender)

profile
train_val.head()
train_val.shape
train_val_state= train_val.drop(labels= ['id', 'item_id', 'dept_id', 'cat_id', 'store_id'],axis=1)

train_val_state= train_val_state.groupby("state_id", as_index=False).sum()

train_val_state= train_val_state.T

train_val_state= train_val_state.rename(columns=train_val_state.iloc[0]).drop(train_val_state.index[0])

train_val_state= train_val_state.reset_index()

train_val_state= train_val_state.rename(columns={"index":"d"})

train_val_state= pd.merge(train_val_state, calender, how="inner", on="d")



# plotting

fig = go.Figure()

fig.add_trace(go.Scatter(x=train_val_state["date"], y=train_val_state["CA"], name="CA", line_color='gray', opacity=1.0))

fig.add_trace(go.Scatter(x=train_val_state["date"], y=train_val_state["TX"], name="TX", line_color='purple', opacity=1.0))

fig.add_trace(go.Scatter(x=train_val_state["date"], y=train_val_state["WI"], name="WI", line_color='salmon', opacity=0.8))

fig.update_layout(title_text='Sales per State over the year', font_size=15)

fig.show()
train_val_1= train_val.copy()

train_val_1["Total Sale"]= train_val_1.sum(axis=1)



# plotting

plt.figure(figsize=(9,5))

plt.style.use('seaborn-darkgrid')



sns.barplot(train_val_1["store_id"], train_val_1["Total Sale"], palette="Reds")

plt.title("Sale per Store", fontsize=25)

plt.xlabel("Store Id", fontsize=15)

plt.ylabel("Total Sale", fontsize=15)
train_val_2= train_val.drop(labels= ["id", "dept_id", "item_id"], axis=1)

train_val_2= train_val_2.groupby(["state_id", "store_id", "cat_id"], as_index=False).sum()

train_val_2["Total Sale"]=train_val_2.sum(axis=1)

train_val_2= train_val_2[["state_id", "store_id", "cat_id", "Total Sale"]]



#plotting

fig = px.bar(train_val_2, x= "store_id", y= "Total Sale",color= "cat_id",  barmode= "group",  facet_row= "state_id", 

             category_orders= {"state_id": ["CA", "TX", "WI"]}, height=600, color_discrete_map={'FOODS':'indigo', 'HOBBIES':'pink', 'HOUSEHOLD':'purple' }

            )

fig.update_traces(marker_line_color='peachpuff', marker_line_width=3, opacity=0.9)



fig.update_layout(title_text= "Category-wise Total Sale per Store per State :",font_size=12,         

                  annotations=[dict(text='CA', font_size=20, font_color="indianred"),

                               dict(text='TX', font_size=20,  font_color="indianred"),

                               dict(text='WI', font_size=20,  font_color="indianred")])

fig.show()
train_val_cat= train_val.drop(labels= ['id', 'item_id', 'dept_id', 'state_id', 'store_id'],axis=1)

train_val_cat= train_val_cat.groupby("cat_id", as_index=False).sum()

train_val_cat= train_val_cat.T

train_val_cat= train_val_cat.rename(columns=train_val_cat.iloc[0]).drop(train_val_cat.index[0])

train_val_cat= train_val_cat.reset_index()

train_val_cat= train_val_cat.rename(columns={"index":"d"})

train_val_cat= pd.merge(train_val_cat, calender, how="inner", on="d")

train_val_cat= train_val_cat[["FOODS","HOBBIES","HOUSEHOLD","year"]]

# train_val_cat= train_val_cat.groupby("year").sum()

train_val_cat = train_val_cat.groupby('year')['FOODS','HOBBIES','HOUSEHOLD'].sum().T



#plotting

plt.style.use('seaborn-darkgrid')

train_val_cat.plot(kind='bar',figsize=(12,7), width=0.6, color=["palegreen", "limegreen", "forestgreen","dimgray","black","darkgray"])

plt.title("Category Sales by year", fontsize=22)

plt.show()
plt.style.use('seaborn-darkgrid')

colors= ["rosybrown","rosybrown","darkcyan","rosybrown","darkcyan","rosybrown","rosybrown"]

train_val.groupby('dept_id').count()["id"].plot(kind='bar',figsize=(10,6),width= 0.6, edgecolor="darkcyan", linewidth=2, color=colors, title= 'Sales by Department')



plt.show()
CA= train_val_state[train_val_state["snap_CA"]==0]

CA_snap= train_val_state[train_val_state["snap_CA"]==1]



TX= train_val_state[train_val_state["snap_TX"]==0]

TX_snap= train_val_state[train_val_state["snap_TX"]==1]



WI= train_val_state[train_val_state["snap_WI"]==0]

WI_snap= train_val_state[train_val_state["snap_WI"]==1]



# plotting

fig = make_subplots(rows=1, cols=3, column_widths=[0.4,0.4,0.4], specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(labels=["Sale without SNAP", "Sale with SNAP"], values=[CA["CA"].sum(), CA_snap["CA"].sum()]), 1,1)

fig.add_trace(go.Pie(labels=["Sale without SNAP", "Sale with SNAP"], values=[TX["TX"].sum(), TX_snap["TX"].sum()]), 1,2)

fig.add_trace(go.Pie(labels=["Sale without SNAP", "Sale with SNAP"], values=[WI["WI"].sum(), WI_snap["WI"].sum()]), 1,3)



fig.update_traces(hole=.6, hoverinfo="label+percent", textinfo='percent' ,marker=dict(colors=['aquamarine', 'darkturquoise'], line=dict(color='gray', width=2.5)))



fig.update_layout(title_text= "SNAP Purchase effect on overall sales per State :",font_size=15,         

                  annotations=[dict(text='CA', x=0.15, y=0.4, font_size=30, showarrow=True, font_color="indianred"),

                               dict(text='TX', x=0.50, y=0.4, font_size=30, showarrow=True, font_color="indianred"),

                               dict(text='WI', x=0.87, y=0.4, font_size=30, showarrow=True, font_color="indianred")])

fig.show()
event1= calender[calender["event_name_1"].notnull()]

event1.loc[85, "event_name_1"]="OrthodoxEaster + Easter"

event1.loc[827, "event_name_1"]="OrthodoxEaster + Cinco De Mayo"

event1.loc[1177, "event_name_1"]="Easter + OrthodoxEaster"

event1.loc[1233, "event_name_1"]="NBAFinalsEnd + Father's day"

event1.loc[1968, "event_name_1"]="NBAFinalsEnd + Father's day"



event1= pd.merge(train_val_state[["d","CA","TX","WI"]], event1[["d","event_name_1"]], on="d", how="inner").drop(labels=["d"], axis=1)

event1["Total Sale"]= event1["CA"] + event1["TX"] + event1["WI"]

event1= event1.groupby("event_name_1", as_index=False).sum() 

event1= event1.sort_values("Total Sale",ascending=True)



# plotting

plt.figure(figsize=(10,14))



plt.barh(event1["event_name_1"],event1["Total Sale"], color="olive")

plt.xlabel("Total Sale", fontsize=18)

plt.ylabel("Events", fontsize=18)

plt.title("Sale on Events", fontsize=22)

# plt.xticks(rotation="vertical")
# creating time-series of train_validation dataset

train_val_series= train_val[train_val.columns[6:]]



# plotting first four time-series

gs=gridspec.GridSpec(2,2)

plt.figure(figsize=(20,5))



ax=pl.subplot(gs[0,0])

plt.plot(train_val_series.iloc[0])

plt.title("First time-series")



ax=pl.subplot(gs[0,1])

plt.plot(train_val_series.iloc[1])

plt.title("Second time-series")



ax=pl.subplot(gs[1,0])

plt.plot(train_val_series.iloc[2])

plt.title("Third time-series")



ax=pl.subplot(gs[1,1])

plt.plot(train_val_series.iloc[3])

plt.title("Fourth time-series")



plt.tight_layout()
# checking the stationarity of the time-series(first 15 time-series):



# Augmented Dickey Fuller(ADF) test:

from statsmodels.tsa.stattools import adfuller



for i in range(15):

    result= adfuller(train_val_series.iloc[i])   

    print("\n")

    print(f"Time-Series {i+1}")

    print("test statistics:", result[0])

    print("p-value:", result[1])

    print("critical values:")

    for key,value in result[4].items():

        print("\t", key,value)



    if (result[0]< result[4]["5%"]) & (result[1]< 0.05):

        print("reject Null Hypothesis: time-series is Stationary")

    else:

        print("failed to reject Null Hypothesis: time-series is Non-Stationary")
# Making first time-series as Stationary 

train_val_series.iloc[0]= train_val_series.iloc[0] - train_val_series.iloc[0].shift(1)

train_val_series.fillna(0,inplace=True)



# Checking the stationarity of the first time-series

result= adfuller(train_val_series.iloc[0])

print(f"Time-Series 1")

print("test statistics:", result[0])

print("p-value:", result[1])

print("critical values:")

for key,value in result[4].items():

    print("\t", key,value)



if (result[0]< result[4]["5%"]) & (result[1]< 0.05):

    print("reject Null Hypothesis: time-series is Stationary")

else:

    print("failed to reject Null Hypothesis: time-series is Non-Stationary")



# Plotting    

plt.figure(figsize=(20,5))    

plt.plot(train_val_series.iloc[0])

plt.title("First time-series")
# Determining number of lags:

import statsmodels.graphics.tsaplots as sgt



plt.figure(figsize=(30,5))

sgt.plot_pacf(train_val_series.iloc[0], lags=40, zero= False)

plt.title("Partial Auto Correlation", fontsize=20)



plt.figure(figsize=(30,5))

sgt.plot_acf(train_val_series.iloc[0], lags=40, zero= False)

plt.title("Auto Correlation", fontsize=20)
# MODEL

import statsmodels.api as sm



model1= sm.tsa.statespace.SARIMAX(train_val_series.iloc[0], order=(1,1,1), seasonal_order=(1,1,1,30))

model1_fit= model1.fit()



model1_fit.summary()
#defining higher lag model

model2= sm.tsa.statespace.SARIMAX(train_val_series.iloc[0], order=(2,1,1), seasonal_order=(2,1,1,30))

model2_fit= model2.fit()



model2_fit.summary()
# Comparing two models



# Log Likelihood Ratio test:

from scipy.stats.distributions import chi2

 

def LLR_test(model_1, model_2, DF=1): # takes model to be compared, DF= diff. in model number which is 1

    L1= model1.fit().llf            # llr test for model 1  

    L2= model2.fit().llf            # llr test for model 2

    LR= (2*(L2-L1))                 # diffenence b/w their llr test

    p= chi2.sf(LR, DF).round(3)     # should be <0.05

    return p

LLR_test(model1, model2, DF=1)
plt.figure(figsize=(20,5))



plt.plot(train_val_series.iloc[0])

plt.plot(model2_fit.fittedvalues)

plt.title("Actual vs Predicted", fontsize=15)

plt.legend(["Actual", "Predicted"])



print("RMSE:", np.sqrt((sum((train_val_series.iloc[0] - model2_fit.fittedvalues)**2))/len(train_val_series.iloc[0])))