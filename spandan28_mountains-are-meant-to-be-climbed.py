! pip install pygam
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
climbing_data = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")

weather_data = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")

climbing_data.head()
weather_data.head()
weather_data.index[weather_data["Date"]=="12/27/2015"]
joined_table = pd.merge(climbing_data, weather_data, how="left", on=["Date"])
joined_table.head(10)
joined_table.loc[joined_table["Succeeded"]==1]

succeded_table = joined_table[["Route","Succeeded"]].groupby("Route").sum().reset_index()

succeded_table.columns = ["Routes", "Suceceeded"]


#fig = px.bar(succeded_table, x="Routes", y ="Suceceeded")

#fig.show()
attempts_made_route = joined_table[["Route","Attempted"]].groupby("Route").sum().reset_index()

attempts_made_route.columns=["Routes", "Attempted"]

#fig = px.bar(attempts_made_route, x="Routes", y="Attempted")

#fig.show()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



data = [go.Bar(x=attempts_made_route.Routes,

               y=attempts_made_route.Attempted, name = "Attempted"),

        go.Bar(x=succeded_table.Routes,

               y=succeded_table.Suceceeded, name = 'Succeess'),]



layout = go.Layout(barmode='stack', title = 'Sucesss v/s Attempt')



fig = go.Figure(data=data, layout=layout)

iplot(fig)
sucess_rate = pd.merge(attempts_made_route,succeded_table, how="left", on=["Routes"] )

#sucess_rate.columns = ["Route", "Success Percentage"]

sucess_rate["Sucess Percentage"] = sucess_rate.Suceceeded / sucess_rate.Attempted *100

sucess_rate.head(10)
import plotly.express as px

fig = px.bar(sucess_rate,x= "Routes", y = "Sucess Percentage")



fig.show()

data_4_analysis = joined_table.drop(columns=["Date", "Attempted","Succeeded", "Battery Voltage AVG"])

data_4_analysis = data_4_analysis.dropna()

data_4_analysis.head(5)
data_4_analysis["Route"] = data_4_analysis["Route"].astype("category")

data_4_analysis["Route_cat"] = data_4_analysis["Route"].cat.codes

X = data_4_analysis.drop(columns=["Route", "Success Percentage" ])

Y = data_4_analysis["Success Percentage"]

import seaborn as sns

corr =data_4_analysis.drop(columns=["Route"]).corr()

plt.figure(figsize=(15, 10))

sns.heatmap(corr,xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from sklearn import preprocessing

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score, r2_score



minmax_scaler = preprocessing.MinMaxScaler()

x_nor = minmax_scaler.fit_transform(X)

X_normalized = pd.DataFrame(x_nor, columns=X.columns)

X_normalized.head()



X_train,  X_valid,Y_train, Y_valid = train_test_split(X_normalized, Y, test_size=0.01)
model = sm.OLS(Y_train.values, X_train.values)

model_res = model.fit()

print(model_res.summary())

predict = model_res.predict(X_valid.values)

print(f"Explained vairance score is {explained_variance_score(Y_valid.values,predict)}")

print(f"Coefficient of determination is {r2_score(Y_valid.values, predict)}")

from pygam import LinearGAM

gam_model = LinearGAM()

gam_model.fit(X_train.values, Y_train.values)

gam_predict = gam_model.predict(X_valid.values)

gam_model.summary()
print(f"Explained vairance for GAM model score is {explained_variance_score(Y_valid.values,gam_predict)}")

print(f"Coefficient of determination for GAM model is {r2_score(Y_valid.values, gam_predict)}")
