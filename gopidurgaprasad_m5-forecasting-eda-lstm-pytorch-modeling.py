import os

import gc

import time

import math

import datetime

from math import log, floor

from sklearn.neighbors import KDTree



import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.utils import shuffle

from tqdm.notebook import tqdm as tqdm



import seaborn as sns

from matplotlib import colors

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import pywt

from statsmodels.robust import mad



import scipy

import statsmodels

from scipy import signal

import statsmodels.api as sm

from fbprophet import Prophet

from scipy.signal import butter, deconvolve

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



import joblib

from tqdm import tqdm_notebook as tqdm



import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn import preprocessing

import sklearn



import warnings

warnings.filterwarnings("ignore")
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    
def read_data(PATH):

    print('Reading files...')

    calendar = pd.read_csv(f'{PATH}/calendar.csv')

    calendar = reduce_mem_usage(calendar)

    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

    sell_prices = pd.read_csv(f'{PATH}/sell_prices.csv')

    sell_prices = reduce_mem_usage(sell_prices)

    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

    sales_train_validation = pd.read_csv(f'{PATH}/sales_train_validation.csv')

    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    submission = pd.read_csv(f'{PATH}/sample_submission.csv')

    return calendar, sell_prices, sales_train_validation, submission



calendar, sell_prices, sales_train_validation, submission = read_data("../input/m5-forecasting-accuracy")
sales_train_validation_melt = pd.melt(sales_train_validation, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='day', value_name='demand')
sales_CA_1 = sales_train_validation_melt[sales_train_validation_melt.store_id == "CA_1"]

new_CA_1 = pd.merge(sales_CA_1, calendar, left_on="day", right_on="d", how="left")

new_CA_1 = pd.merge(new_CA_1, sell_prices, left_on=["store_id", "item_id", "wm_yr_wk"],right_on=["store_id", "item_id", "wm_yr_wk"], how="left")

new_CA_1["day_int"] = new_CA_1.day.apply(lambda x: int(x.split("_")[-1]))
new_CA_1.head()
day_sum = new_CA_1.groupby("day_int")[["sell_price", "demand"]].agg("sum").reset_index()


fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Scatter(x=day_sum.day_int, 

                         y=day_sum.demand,

                         #showlegend=False,

                         mode="lines",

                         name="demand",

                         #marker=dict(color="mediumseagreen"),

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=day_sum.day_int, 

                         y=day_sum.sell_price,

                         #showlegend=False,

                         mode="lines",

                         name="sell_price",

                         #marker=dict(color="mediumseagreen")

                         ),

             

              row=2,col=1           

              )



fig.update_layout(height=1000, title_text="SUM -> Demand  and Sell_price")

fig.show()
# For each day we count_nonzeros over products sell_price and demand



day_sum = new_CA_1.groupby("day_int")[["demand","event_name_1" ]].agg({"demand": np.count_nonzero, "event_name_1": "first"}).reset_index()

def count_nulls(series):

    return len(series) - series.count()



cout_null = new_CA_1.groupby("day_int")["sell_price"].agg(count_nulls).reset_index()


fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Scatter(x=cout_null.day_int, 

                         y=cout_null.sell_price,

                         #showlegend=False,

                         mode="lines",

                         name="sell_price",

                         #marker=dict(color="mediumseagreen")

                        ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=day_sum.day_int, 

                         y=day_sum.demand,

                         #showlegend=False,

                         mode="lines",

                         name="demand",

                         #marker=dict(color="mediumseagreen")

                        ),

             

              row=2,col=1           

              )



fig.update_layout(height=1000, title_text="Count_Nonzero -> Sell_price  and Demand")

fig.show()
item_id = new_CA_1.groupby("item_id")[["sell_price", "demand"]].agg({

    "sell_price": ["max", "mean", "min"],

    "demand" : ["max", "mean", "min"]

}).reset_index()
fig = make_subplots(rows=1, cols=1)



item_id = item_id.sort_values(("sell_price", "max"))

fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["sell_price", "max"],

                         #showlegend=Ture,

                         mode="lines",

                         name="max",

                         #marker=dict(color="mediumseagreen")

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["sell_price", "mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="mean",

                         #marker=dict(color="yellow")

                         ),

             

              row=1,col=1           

              )



fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["sell_price", "min"],

                         #showlegend=Ture,

                         mode="lines",

                         name="min",

                         #marker=dict(color="blue")

                         ),

             

              row=1,col=1           

              )



fig.update_layout(height=500, title_text="Sell_price")

fig.show()


fig = make_subplots(rows=1, cols=1)



item_id = item_id.sort_values(("demand", "max"))

fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["demand", "max"],

                         #showlegend=Ture,

                         mode="lines",

                         name="max",

                         #marker=dict(color="mediumseagreen")

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["demand", "mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="mean",

                         #marker=dict(color="yellow")

                         ),

             

              row=1,col=1           

              )



fig.add_trace(go.Scatter(x=item_id["item_id"], 

                         y=item_id["demand", "min"],

                         #showlegend=Ture,

                         mode="lines",

                         name="min",

                         #marker=dict(color="blue")

                         ),

             

              row=1,col=1           

              )



fig.update_layout(height=500, title_text="Demand")

fig.show()
# For each item week_days vs week_ends over days sell_price and demand



week_end = new_CA_1[new_CA_1.weekday == "Sunday"]

week_day = new_CA_1[new_CA_1.weekday != "Sunday"]



week_end = week_end.groupby("item_id")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()

week_end.columns = ['_'.join(col).strip() for col in week_end.columns.values]



week_day = week_day.groupby("item_id")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()

week_day.columns = ['_'.join(col).strip() for col in week_day.columns.values]
fig = go.Figure()



fig.add_trace(go.Scatter(x=week_end["item_id_"],

                         y=week_end["demand_mean"],

                         mode="lines",

                         name="week_day"



))



fig.add_trace(go.Scatter(x=week_end["item_id_"],

                         y=week_day["demand_mean"],

                         mode="lines",

                         name="normal_day"



))



fig.update_layout(height=500, title_text="Demand")

fig.show()

fig = go.Figure()



fig.add_trace(go.Scatter(x=week_end["item_id_"],

                         y=week_end["sell_price_mean"],

                         mode="lines",

                         name="week_day"



))



fig.add_trace(go.Scatter(x=week_end["item_id_"],

                         y=week_day["sell_price_mean"],

                         mode="lines",

                         name="normal_day"



))



fig.update_layout(height=500,title_text="Sell_price")

fig.show()

events = new_CA_1[~new_CA_1.event_name_1.isna()]

events = events.groupby("event_name_1")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()

events.columns = ['_'.join(col).strip() for col in events.columns.values]
fig = go.Figure()



fig.add_trace(go.Scatter(x=events["event_name_1_"],

                         y=events["demand_mean"],

                         mode="lines",

                         name="week_day"

))



fig.update_layout(height=500, title_text="Demand")

fig.show()

fig = go.Figure()



fig.add_trace(go.Scatter(x=events["event_name_1_"],

                         y=events["sell_price_mean"],

                         mode="lines",

                         name="week_day"



))



fig.update_layout(height=500, title_text="Sell_price")

fig.show()

## Number of items contain each Category



def n_unique(series):

    return series.nunique()



Category_count = new_CA_1.groupby("cat_id")["item_id"].agg(n_unique).reset_index()
fig = px.bar(Category_count, y="item_id", x="cat_id", color="cat_id", title="Category Item Count")



fig.update_layout(height=500, width=600)

fig.show()
## For each category mean of deman and sell_price



Category = new_CA_1.groupby(["day_int","cat_id"])[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()

Category.columns = ['_'.join(col).strip() for col in Category.columns.values]



FOODS = Category[Category.cat_id_ == "FOODS"]

HOBBIES = Category[Category.cat_id_ == "HOBBIES"]

HOUSEHOLD = Category[Category.cat_id_ == "HOUSEHOLD"]


fig = make_subplots(rows=1, cols=1)





fig.add_trace(go.Scatter(x=FOODS["day_int_"], 

                         y=FOODS["demand_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="FOODS",

                         #marker=dict(color="mediumseagreen")

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 

                         y=HOBBIES["demand_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOBBIES",

                         #marker=dict(color="yellow")

                         ),

             

              row=1,col=1           

              )



fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 

                         y=HOUSEHOLD["demand_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOUSEHOLD",

                         #marker=dict(color="blue")

                         ),

             

              row=1,col=1           

              )



fig.update_layout(height=500, title_text="Demand Mean Over Category by day-by-day")

fig.show()


fig = make_subplots(rows=1, cols=1)





fig.add_trace(go.Scatter(x=FOODS["day_int_"], 

                         y=FOODS["demand_max"],

                         #showlegend=Ture,

                         mode="lines",

                         name="FOODS",

                         #marker=dict(color="mediumseagreen")

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 

                         y=HOBBIES["demand_max"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOBBIES",

                         #marker=dict(color="yellow")

                         ),

             

              row=1,col=1           

              )



fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 

                         y=HOUSEHOLD["demand_max"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOUSEHOLD",

                         #marker=dict(color="blue")

                         ),

             

              row=1,col=1           

              )



fig.update_layout(height=500, title_text="Demand Max Over Category by day-by-day")

fig.show()


fig = make_subplots(rows=1, cols=1)





fig.add_trace(go.Scatter(x=FOODS["day_int_"], 

                         y=FOODS["sell_price_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="FOODS",

                         #marker=dict(color="mediumseagreen")

                         ),



              row=1,col=1         

              )



fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 

                         y=HOBBIES["sell_price_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOBBIES",

                         #marker=dict(color="yellow")

                         ),

             

              row=1,col=1           

              )



fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 

                         y=HOUSEHOLD["sell_price_mean"],

                         #showlegend=Ture,

                         mode="lines",

                         name="HOUSEHOLD",

                         #marker=dict(color="blue")

                         ),

             

              row=1,col=1           

              )



fig.update_layout(height=500, title_text="Sell_price Mean Over Category by day-by-day")

fig.show()
fig = go.Figure()



fig.add_trace(go.Box(x=FOODS.cat_id_, y=FOODS.demand_mean, name="FOODS"))



fig.add_trace(go.Box(x=HOUSEHOLD.cat_id_, y=HOUSEHOLD.demand_mean, name="HOUSEHOLD"))



fig.add_trace(go.Box(x=HOBBIES.cat_id_, y=HOBBIES.demand_mean, name="HOBBIES"))





fig.update_layout(yaxis_title="Demand", xaxis_title="Time", title="Demand Mean vs. Category")
fig = go.Figure()



fig.add_trace(go.Box(x=FOODS.cat_id_, y=FOODS.sell_price_mean, name="FOODS"))



fig.add_trace(go.Box(x=HOUSEHOLD.cat_id_, y=HOUSEHOLD.sell_price_mean, name="HOUSEHOLD"))



fig.add_trace(go.Box(x=HOBBIES.cat_id_, y=HOBBIES.sell_price_mean, name="HOBBIES"))





fig.update_layout(yaxis_title="Sell Price", xaxis_title="Time", title="Sell Price Mean vs. Category")
## Number of items contain each Deportments



def n_unique(series):

    return series.nunique()



dep_count = new_CA_1.groupby("dept_id")["item_id"].agg(n_unique).reset_index()
px.bar(dep_count, y="item_id", x="dept_id", color="dept_id", title="Deportment Item Count")
## For each Deportment mean of deman and sell_price



dep = new_CA_1.groupby(["day_int","dept_id"])[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()

dep.columns = ['_'.join(col).strip() for col in dep.columns.values]


fig = make_subplots(rows=1, cols=1)



for each_dep in dep.dept_id_.unique():

    dep_df = dep[dep.dept_id_ == each_dep]

    fig.add_trace(go.Scatter(x=dep_df["day_int_"], 

                             y=dep_df["demand_mean"],

                             #showlegend=Ture,

                             mode="lines",

                             name=each_dep,

                             #marker=dict(color="mediumseagreen")

                             ),



                  row=1,col=1         

                  )

    

fig.update_layout(title_text="Demand Mean Over Deportments by day-by-day")

fig.show()


fig = make_subplots(rows=1, cols=1)



for each_dep in dep.dept_id_.unique():

    dep_df = dep[dep.dept_id_ == each_dep]

    fig.add_trace(go.Scatter(x=dep_df["day_int_"], 

                             y=dep_df["sell_price_mean"],

                             #showlegend=Ture,

                             mode="lines",

                             name=each_dep,

                             #marker=dict(color="mediumseagreen")

                             ),



                  row=1,col=1         

                  )

    

fig.update_layout(title_text="Sell Prices Mean Over Deportments by day-by-day")

fig.show()
fig = go.Figure()



for each_dep in dep.dept_id_.unique():

    dep_df = dep[dep.dept_id_ == each_dep]



    fig.add_trace(go.Box(x=dep_df.dept_id_, y=dep_df.demand_mean, name=each_dep))

    





fig.update_layout(yaxis_title="Demand", xaxis_title="Time", title="Demand Mean vs. Deportment")
fig = go.Figure()



for each_dep in dep.dept_id_.unique():

    dep_df = dep[dep.dept_id_ == each_dep]



    fig.add_trace(go.Box(x=dep_df.dept_id_, y=dep_df.sell_price_mean, name=each_dep))

    

    

fig.update_layout(yaxis_title="Sell Price", xaxis_title="Time", title="Sell Price Mean vs. Deportment")
# take only one stor for demo



CA1 = new_CA_1

CA1.head()
CA1 = CA1[["item_id","day_int", "demand", "sell_price", "date"]]

CA1.fillna(0, inplace=True)

print(CA1.shape)

CA1.head()
def date_features(df):

    

    df["date"] = pd.to_datetime(df["date"])

    df["day"] = df.date.dt.day

    df["month"] = df.date.dt.month

    df["week_day"] = df.date.dt.weekday



    df.drop(columns="date", inplace=True)



    return df



def sales_features(df):



    df.sell_price.fillna(0, inplace=True)



    return df



def demand_features(df):



    df["lag_t28"] = df["demand"].transform(lambda x: x.shift(28))

    df["rolling_mean_t7"] = df["demand"].transform(lambda x:x.shift(28).rolling(7).mean())

    df['rolling_mean_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    df['rolling_mean_t60'] = df['demand'].transform(lambda x: x.shift(28).rolling(60).mean())

    df['rolling_mean_t90'] = df['demand'].transform(lambda x: x.shift(28).rolling(90).mean())

    df['rolling_mean_t180'] = df['demand'].transform(lambda x: x.shift(28).rolling(180).mean())

    df['rolling_std_t7'] = df['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    df['rolling_std_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).std())



    df.fillna(0, inplace=True)



    return df
!mkdir "something_spl"

# Saving each item with there item name.npy



for item in tqdm(CA1.item_id.unique()):

    one_item = CA1[CA1.item_id == item][["demand", "sell_price", "date"]]

    item_df = date_features(one_item)

    item_df = sales_features(item_df)

    item_df = demand_features(item_df)

    joblib.dump(item_df.values, f"something_spl/{item}.npy")
# create dataframe for loading npy files and  train valid split



data_info = CA1[["item_id", "day_int"]]



# total number of days -> 1913

# for training we are taking data between 1800 < train <- 1913-28-28 = 1857



train_df = data_info[(1800 < data_info.day_int) &( data_info.day_int < 1857)]



# valid data is given last day -> 1885 we need to predict next 28days



valid_df = data_info[data_info.day_int == 1885]
label = preprocessing.LabelEncoder()

label.fit(train_df.item_id)

label.transform(["FOODS_3_827"])
class DataLoading:

    def __init__(self, df, train_window = 28, predicting_window=28):

        self.df = df.values

        self.train_window = train_window

        self.predicting_window = predicting_window



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, item):

        df_item = self.df[item]

        item_id = df_item[0]

        day_int = df_item[1]

        

        item_npy = joblib.load(f"something_spl/{item_id}.npy")

        item_npy_demand = item_npy[:,0]

        features = item_npy[day_int-self.train_window:day_int]

    



        predicted_demand = item_npy_demand[day_int:day_int+self.predicting_window]



        item_label = label.transform([item_id])

        item_onehot = [0] * 3049

        item_onehot[item_label[0]] = 1



        list_features = []

        for f in features:

            one_f = []

            one_f.extend(item_onehot)

            one_f.extend(f)

            list_features.append(one_f)



        return {

            "features" : torch.Tensor(list_features),

            "label" : torch.Tensor(predicted_demand)

        }
## for exaple one item



datac = DataLoading(train_df)

n = datac.__getitem__(100)

n["features"].shape, n["label"].shape
class LSTM(nn.Module):

    def __init__(self, input_size=3062, hidden_layer_size=100, output_size=28):

        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)



        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),

                            torch.zeros(1,1,self.hidden_layer_size))

        

    def forward(self, input_seq):



        lstm_out, self.hidden_cell = self.lstm(input_seq)



        lstm_out = lstm_out[:, -1]



        predictions = self.linear(lstm_out)



        return predictions

# loss function

def criterion1(pred1, targets):

    l1 = nn.MSELoss()(pred1, targets)

    return l1
def train_model(model,train_loader, epoch, optimizer, scheduler=None, history=None):

    model.train()

    total_loss = 0

    

    t = tqdm(train_loader)

    

    for i, d in enumerate(t):

        

        item = d["features"].cuda().float()

        y_batch = d["label"].cuda().float()



        optimizer.zero_grad()



        out = model(item)

        loss = criterion1(out, y_batch)



        total_loss += loss

        

        t.set_description(f'Epoch {epoch+1} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))



        if history is not None:

            history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()

            history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']



        loss.backward()

        optimizer.step()

        



def evaluate_model(model, val_loader, epoch, scheduler=None, history=None):

    model.eval()

    loss = 0

    pred_list = []

    real_list = []

    RMSE_list = []

    with torch.no_grad():

        for i,d in enumerate(tqdm(val_loader)):

            item = d["features"].cuda().float()

            y_batch = d["label"].cuda().float()



            o1 = model(item)

            l1 = criterion1(o1, y_batch)

            loss += l1

            

            o1 = o1.cpu().numpy()

            y_batch = y_batch.cpu().numpy()

            

            for pred, real in zip(o1, y_batch):

                rmse = np.sqrt(sklearn.metrics.mean_squared_error(real, pred))

                RMSE_list.append(rmse)

                pred_list.append(pred)

                real_list.append(real)



    loss /= len(val_loader)

    

    if scheduler is not None:

        scheduler.step(loss)



    print(f'\n Dev loss: %.4f RMSE : %.4f'%(loss, np.mean(RMSE_list)))

    
DEVICE = "cuda"

TRAIN_BATCH_SIZE = 512

TEST_BATCH_SIZE = 128

EPOCHS = 1

start_e = 1





model = LSTM()

model.to(DEVICE)



train_dataset = DataLoading(train_df)



train_loader = torch.utils.data.DataLoader(

    dataset=train_dataset,

    batch_size= TRAIN_BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    drop_last=True

)





valid_dataset = DataLoading(valid_df)



valid_loader = torch.utils.data.DataLoader(

    dataset=valid_dataset,

    batch_size= TEST_BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    drop_last=True

)





optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)



for epoch in range(start_e, EPOCHS+1):

    train_model(model, train_loader, epoch, optimizer, scheduler=scheduler, history=None)

    evaluate_model(model, valid_loader, epoch, scheduler=scheduler, history=None)
