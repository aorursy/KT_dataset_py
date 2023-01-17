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
# Các thư viện cần thiết

import seaborn as sns

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
# Hiển thị 10 dòng đầu trong train_data

train_data.head(10)
# Hiển thị 10 dòng đầu trong test_data

test_data.head(10)
train_data.isnull().sum()
test_data.isnull().sum()
train_data=train_data.drop(columns=["County","Province_State","Id"])

test_data=test_data.drop(columns=["County","Province_State","ForecastId"])
train_data.head(10)
test_data.head(10)
train_data.Country_Region.nunique()
train_data.Date.nunique()
# lấy cột Country_Region và Population

df_countries_popu = train_data[["Country_Region", "Population"]]

df_countries_popu.drop_duplicates(subset=["Country_Region", "Population"], inplace=True)

df_countries_popu = df_countries_popu.groupby(["Country_Region"], as_index=False).sum()



# top 10 nước có Population cao nhất

df_top10_countries_popu = df_countries_popu.nlargest(10, "Population")

df_top10_countries_popu
# Gom nhóm theo Country_Region

df_country_grouped=train_data.groupby(["Country_Region"], as_index=False).sum()

df_country_grouped
df_top10_countries = df_country_grouped.nlargest(10, "TargetValue")

df_top10_countries[["Country_Region", "TargetValue"]]
# lấy ra list tên 10 nước có tổng TargetValue lớn nhất

top10_countries = list(df_top10_countries["Country_Region"])



# Gom nhóm train_data theo top 10 nước và theo thời gian

df_top10_countries_grouped = train_data[train_data.Country_Region.isin(top10_countries)]

df_top10_countries_grouped = df_top10_countries_grouped.groupby(["Country_Region", "Date"], as_index=False).sum()

#df_top10_countries_grouped
df_target_value_daily = pd.pivot_table(

    df_top10_countries_grouped,

    values="TargetValue",

    index=["Country_Region"],

    columns="Date"

)

df_target_value_daily
# Gom nhóm train_data (chỉ xét những dòng mà Target có giá trị Fatalities) theo top 10 nước và theo thời gian

df_fatalities = train_data[train_data.Country_Region.isin(top10_countries)]

df_fatalities = df_fatalities[df_fatalities["Target"]=="Fatalities"]

df_fatalities = df_fatalities.groupby(["Country_Region", "Date"], as_index=False).sum()

#df_fatalities
df_fatalities_daily = pd.pivot_table(

    df_fatalities,

    values="TargetValue",

    index=["Country_Region"],

    columns="Date"

)

df_fatalities_daily
# Xem mối tương quan giữa các thuộc tính (column) Population, Weight, TargetValue

sns.pairplot(train_data)
fig = px.pie(

    df_country_grouped,

    values="TargetValue", names="Country_Region",

    title="TargetValue (gồm ConfirmedCases và Fatalities) của các nước",

    color_discrete_sequence=px.colors.sequential.Aggrnyl

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
fig = px.bar(

    df_top10_countries, 

    y='TargetValue', x='Country_Region',

    text='TargetValue', color="TargetValue",

    title="Top 10 nước có TargetValue lớn nhất"

)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
train_data_fatalities = train_data[train_data["Target"]=="Fatalities"]



fig = px.pie(

    train_data_fatalities,

    values="TargetValue", names="Country_Region",

    title="TargetValue (chỉ xét Fatalities) của các nước",

    color_discrete_sequence=px.colors.sequential.Aggrnyl

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
fig = px.pie(

    train_data,

    values="TargetValue", names="Target",

    title="TargetValue của ConfirmedCases và Fatalities",

    color_discrete_sequence=px.colors.sequential.Emrld

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
fig = px.pie(

    df_countries_popu, 

    values="Population", names="Country_Region",

    title="Population của các nước",

    color_discrete_sequence=px.colors.sequential.Aggrnyl

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
fig = px.bar(

    df_top10_countries_popu,

    y='Population', x='Country_Region',

    text='Population', color="Population",

    title="Top 10 nước có Population lớn nhất")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
px.line(

    df_top10_countries_grouped,

    x="Date", y="TargetValue", color="Country_Region", 

    title="TargetValue theo ngày của top 10 nước có TargetValue lớn nhất"

)
fig = go.Figure(data=go.Heatmap(

        z=df_target_value_daily.values,# TargetValue

        x=df_target_value_daily.columns,# Date

        y=df_target_value_daily.index,#Country_Region

        colorscale="tempo"))



fig.update_layout(title="TargetValue theo ngày của top 10 nước có TargetValue lớn nhất")



fig.show()
px.line(

    df_fatalities,

    x="Date", y="TargetValue", color="Country_Region",

    title="TargetValue (chỉ xét Fatalities) theo ngày của top 10 nước có TargetValue lớn nhất"

)
fig = go.Figure(data=go.Heatmap(

        z=df_fatalities_daily.values,# TargetValue

        x=df_fatalities_daily.columns,# Date

        y=df_fatalities_daily.index,#Country_Region

        colorscale="tempo"))



fig.update_layout(title="TargetValue (chỉ xét Fatalities) theo ngày của top 10 nước có TargetValue lớn nhất")



fig.show()
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
train_data=train_data.drop(columns=["County","Province_State","Id"])

test_data=test_data.drop(columns=["County","Province_State","ForecastId"])
train_data
test_data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Chuẩn hóa Country_Region thành số nguyên

train_data.iloc[:,0] = le.fit_transform(train_data.iloc[:,0].values)



# Chuẩn hóa Target thành số nguyên

train_data.iloc[:,4] = le.fit_transform(train_data.iloc[:,4].values)
# Chuẩn hóa Country_Region thành số nguyên

test_data.iloc[:,0] = le.fit_transform(test_data.iloc[:,0].values)



# Chuẩn hóa Target thành số nguyên

test_data.iloc[:,4] = le.fit_transform(test_data.iloc[:,4].values)
dates = pd.to_datetime(train_data["Date"], format="%Y-%m-%d")

train_data["Date"] = dates.dt.strftime("%Y%m%d").astype(int)
dates = pd.to_datetime(test_data["Date"], format="%Y-%m-%d")

test_data["Date"] = dates.dt.strftime("%Y%m%d").astype(int)
train_data
test_data
# tập features: bỏ đi cột "TargetValue" từ train_data

x_train = train_data.drop("TargetValue", axis = 1)  # axis: 1 cho column, 0 cho row



# tập labels: chỉ lấy cột "TargetValue" từ train_data

y_train = train_data["TargetValue"]
x_train
y_train
from sklearn.model_selection import train_test_split 

# tập test chiếm 20%

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
# thư viện cần thiết

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
pipe_dtr = Pipeline([

    ("scl", StandardScaler()),

    ("DecisionTreeRegressor", DecisionTreeRegressor())

])
pipe_dtr.fit(x_train, y_train)
pipe_dtr.score(x_test, y_test)
y_pred_dtr = pipe_dtr.predict(x_test)
fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test,

                         mode='lines', name='y_test'),

             row=1, col=1)



fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_pred_dtr,

                         mode='lines',name='y_pred'),

             row=2, col=1)

fig.update_layout(title="So sánh y_test và y_pred")

fig.show()
pipe_rfr = Pipeline([

    ("scl", StandardScaler()),

    ("RandomForestRegressor", RandomForestRegressor())

])
pipe_rfr.fit(x_train, y_train)
pipe_rfr.score(x_test, y_test)  # r2_score
y_pred_rfr = pipe_rfr.predict(x_test)
fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test,

                         mode='lines', name='y_test'),

             row=1, col=1)



fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_pred_rfr,

                         mode='lines',name='y_pred'),

             row=2, col=1)

fig.update_layout(title="So sánh y_test và y_pred")

fig.show()
pred = pipe_rfr.predict(test_data)
result = pd.DataFrame({

    "Id": np.arange(pred.shape[0]),

    "TargetValue": pred

})

result
df_sub = result.groupby(["Id"])["TargetValue"].quantile([0.05, 0.5, 0.95]).reset_index()

df_sub.head(15)
df_sub["Id"] = df_sub["Id"] + 1

df_sub["Id"] = df_sub["Id"].astype("str")

df_sub["level_1"] = df_sub["level_1"].astype("str")

df_sub["Id"] = df_sub["Id"] + "_" + df_sub["level_1"]



df_sub = df_sub.drop("level_1", axis=1)



df_sub.columns = ["ForecastId_Quantile", "TargetValue"]



#df_sub.head(15)

df_sub
df_sub.to_csv("submission.csv", index=False)