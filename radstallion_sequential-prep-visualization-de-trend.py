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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

sample_submission = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

sample_submission.head()
dtype = {}

for i in range(2000):

    dtype[f"d_{i}"] = np.uint16

sales_train_validation = pd.read_csv(

    "/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv",

    dtype = dtype

).set_index(["cat_id","id","item_id","dept_id","store_id","state_id"])

sales_train_validation.columns = [int(col.split("_")[-1]) for col in sales_train_validation.columns]

sales_train_validation.columns = pd.MultiIndex.from_tuples(

    [ ("n_sales",i)for i in sales_train_validation.columns]

)

sales_train_validation.head()
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv",dtype={"sell_price":float})



sell_calendar = calendar[["wm_yr_wk","d"]].join(sell_prices.set_index("wm_yr_wk"),on="wm_yr_wk")

calendar["d"] = calendar["d"].apply(lambda x : int(x.split("_")[-1]))

sell_calendar.head()
sell_price_calendar = pd.pivot_table(sell_calendar, values='sell_price', index=['store_id', 'item_id'],

                    columns=['d'], aggfunc="last", fill_value=0)

sell_price_calendar.columns = [int(col.split("_")[-1]) for col in sell_price_calendar.columns]

sell_price_calendar = sell_price_calendar.sort_index(axis=1)

sell_price_calendar.columns = pd.MultiIndex.from_tuples(

    [ ("sell_prices",i)for i in sell_price_calendar.columns]

)



sell_price_calendar.head()
sales_dataset = sales_train_validation.join(sell_price_calendar,on=["store_id","item_id"])

revenue = sales_dataset["sell_prices"]*sales_dataset["n_sales"]

revenue.columns = pd.MultiIndex.from_tuples(

    [ ("revenue",i)for i in revenue.columns]

)

sales_dataset = sales_dataset.join(revenue)

sales_dataset.head()
weekly = np.cos(np.array(range(2000))*2*np.pi/7)

monthly = np.cos(np.array(range(2000))*2*np.pi/(365.2425/12))

yearly = np.cos(np.array(range(2000))*2*np.pi/365.2425)

d = np.array(range(2000))

cyclic_time = pd.DataFrame({

    "weekly":weekly,

    "monthly":monthly,

    "yearly":yearly,

    "w_step":np.array(range(2000))%7/7,

    "m_step":np.array(range(2000))%(365.2425/12)/(365.2425/12),

    "y_step":np.array(range(2000))%365.2425/365.2425,

    "d":d

}).set_index("d")

# cyclic_time = (cyclic_time+1)/2

cyclic_time["mean"] =cyclic_time.mean(axis=1)

cyclic_time = cyclic_time.join(pd.get_dummies(calendar[["wday","month","d"]].set_index("d"),columns=["wday","month"])).fillna(0)

cyclic_time.to_pickle("cyclic_time.zip.pkl",compression="zip")

cyclic_time[["mean","yearly","monthly"]].plot(figsize=(32/2, 9/2))

cyclic_time.head()
nsales_percata = sales_dataset["n_sales"].groupby(["cat_id","state_id"]).agg(np.mean).transpose()



sales_dataset["n_sales"].groupby(["cat_id"]).agg(np.mean).transpose().plot(figsize=(32/2, 9/2))

plt.title("ALL mean n sales by cat_id")

for cata in ["FOODS","HOBBIES","HOUSEHOLD"]:

    nsales_percata[cata].plot(figsize=(32/2, 9/2))

    plt.title(f"{cata} mean n sales")

from sklearn.linear_model import LinearRegression

for cata in ["FOODS","HOBBIES","HOUSEHOLD"]:

    

    ser = nsales_percata[cata].transpose().mean()

    reg = LinearRegression().fit(ser.index.values.reshape(-1, 1), ser.values)

    trend = reg.predict(ser.index.values.reshape(-1, 1))

    nsales_percata[(cata,"trends")] = trend

    nsales_percata[cata].plot(figsize=(32/2, 9/2))

    plt.title(f"{cata} mean with trends")



    

nsales_percata[[("FOODS","trends"),("HOBBIES","trends"),("HOUSEHOLD","trends")]].plot(figsize=(32/2, 9/2))

plt.title(f"trends for each catagoy")

from tqdm.auto import tqdm

tqdm.pandas()

def apply_trend(x):

    reg = LinearRegression().fit(x.index.values.reshape(-1, 1), x.values)

    trend = reg.predict(x.index.values.reshape(-1, 1))

    return trend





n_sales_trends = sales_dataset["n_sales"].copy()

n_sales_trends[n_sales_trends.columns] = np.stack(n_sales_trends.progress_apply(apply_trend,axis=1))



n_sales_detrends = sales_dataset["n_sales"].copy()

n_sales_detrends = n_sales_detrends - n_sales_trends



n_sales_detrends.groupby(["cat_id","state_id"]).agg(np.mean).transpose().plot(figsize=(32/2, 9/2))

plt.title(f"detrended data")
n_sales_detrends.columns = pd.MultiIndex.from_tuples(

    [ ("n_sales_detrends",i)for i in n_sales_detrends.columns]

)

n_sales_trends.columns = pd.MultiIndex.from_tuples(

    [ ("n_sales_trends",i)for i in n_sales_trends.columns]

)
plt.figure(figsize=(32/2, 9/2))

plt.plot(n_sales_detrends["n_sales_detrends"].agg(np.mean).transpose()[-365:-1])

plt.plot(cyclic_time[["weekly","monthly"]][1914-365:1914],alpha=0.5)

plt.legend(["average detrended nsales","weekly sin t=7","monthly sin t=30"])


sales_dataset = sales_dataset.join(n_sales_detrends).join(n_sales_trends)

sales_dataset.to_pickle("sales_dataset.zip.pkl",compression="zip")

sales_dataset.columns.unique(0)
calendar.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error





index = 1

Y = sales_dataset["n_sales"].iloc[index][-1000:-28]

X = cyclic_time.loc[Y.index]

reg = RandomForestRegressor(n_estimators=5).fit(X, Y)

# reg = LinearRegression().fit(X, Y)

print("training")

print(f"R2 {reg.score(X, Y)}")

print(f"RMSE {np.sqrt(mean_squared_error(reg.predict(X),Y))}")

plt.scatter(reg.predict(X),Y.values)



print("validation")



Y_val = sales_dataset["n_sales"].iloc[index][-28:]

X_val = cyclic_time.loc[Y_val.index]

plt.scatter(reg.predict(X_val),Y_val.values)

print(f"R2 {reg.score(X_val, Y_val)}")

print(f"RMSE {np.sqrt(mean_squared_error(reg.predict(X_val),Y_val))}")
index_list = sales_dataset["n_sales"].index



def get_score_from_idx(idx):

    Y = sales_dataset["n_sales"].loc[idx][-600:-28]

    X = cyclic_time.loc[Y.index]

    reg = RandomForestRegressor(n_estimators=5).fit(X, Y)

    reg = LinearRegression().fit(X, Y)

    

    Y_val = sales_dataset["n_sales"].iloc[index][-28:]

    X_val = cyclic_time.loc[Y_val.index]

    return {

        "index":idx,

        "rmse":np.sqrt(mean_squared_error(reg.predict(X_val),Y_val))

    }



ret = []

for idx in tqdm(index_list[:10]):

    ret.append(get_score_from_idx(idx))
SCORE = pd.DataFrame(ret).set_index("index")

SCORE.index = pd.MultiIndex.from_tuples(

    SCORE.index 

)

SCORE.hist()

SCORE.describe()