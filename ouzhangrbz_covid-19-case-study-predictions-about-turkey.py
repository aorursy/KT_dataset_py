import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

from plotly.subplots import make_subplots

import seaborn as sns

import datetime
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head() 
data.tail()
data.columns
data.info()
ser = pd.Series([5, 6, np.NaN, np.NaN])

ser.isna()

ser.isna().mean()
for each in data:

    print(each)
NAN = [(each, data[each].isna().mean()*100) for each in data]

NAN = pd.DataFrame(NAN, columns=["name","rate"])

NAN
data["Province/State"].fillna("Unknown",inplace=True)
NAN = [(each, data[each].isna().mean()*100) for each in data]

NAN = pd.DataFrame(NAN, columns=["name","rate"])

NAN
data.head()
data[["Confirmed","Deaths","Recovered"]] = data[["Confirmed","Deaths","Recovered"]].astype(int)
print(data.loc[:,["Confirmed","Deaths","Recovered"]])
data.describe()
data['Country/Region'].replace("Mainland China","China", inplace=True)
data['Country/Region']
data["current_case"] = data["Confirmed"] - data["Deaths"] - data["Recovered"]

data.head()
data_Frame_Last_Update = data[data["ObservationDate"] == max(data["ObservationDate"])].reset_index()

data_Frame_Last_Update.head()
data.corr()
corr_data = data[["Confirmed","Deaths","Recovered"]]

corr_data.head()
corr_data.corr()
f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(corr_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
plt.bar(data["ObservationDate"],data["Confirmed"])

plt.show()
