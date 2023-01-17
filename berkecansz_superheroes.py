# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/marvel-superheroes/charcters_stats.csv")
df
Name=df.sort_values(by="Total",ascending=False).head(10)["Name"]
Total=df.sort_values(by="Total",ascending=False).head(10)["Total"]
Total_point=pd.DataFrame({"Name":Name,"Total":Total})
Total_point


Name=df.sort_values(by="Power",ascending=False).head(10)["Name"]
Power=df.sort_values(by="Power",ascending=False).head(10)["Power"]
Power_point=pd.DataFrame({"Name":Name,"Power":Power})
Power_point


Name=df.sort_values(by="Strength",ascending=False).head(10)["Name"]
Strength=df.sort_values(by="Strength",ascending=False).head(10)["Strength"]
Strength_point=pd.DataFrame({"Name":Name,"Strength":Strength})
Strength_point

Name=df.sort_values(by="Speed",ascending=False).head(10)["Name"]
Speed=df.sort_values(by="Speed",ascending=False).head(10)["Speed"]
Speed_point=pd.DataFrame({"Name":Name,"Speed":Speed})
Speed_point
Name=df.sort_values(by="Intelligence",ascending=False).head(10)["Name"]
Intelligence=df.sort_values(by="Intelligence",ascending=False).head(10)["Intelligence"]
Intelligence_point=pd.DataFrame({"Name":Name,"Intelligence":Intelligence})
Intelligence_point

Name=df.sort_values(by="Durability",ascending=False).head(10)["Name"]
Durability=df.sort_values(by="Durability",ascending=False).head(10)["Durability"]
Durability_point=pd.DataFrame({"Name":Name,"Durability":Durability})
Durability_point

Name=df.sort_values(by="Combat",ascending=False).head(10)["Name"]
Combat=df.sort_values(by="Combat",ascending=False).head(10)["Combat"]
Combat_point=pd.DataFrame({"Name":Name,"Combat":Combat})
Combat_point

def Name(Name):
    return df[df["Name"]==Name]
Name('Deadpool')
def alignment(Alignment):
    return df[df["Alignment"]==Alignment]
good=alignment('good')
bad=alignment('bad')
neutral=alignment('neutral')

good
bad
neutral
Name=good.sort_values(by="Total",ascending=False).head(10)["Name"]
Total=good.sort_values(by="Total",ascending=False).head(10)["Total"]
Total_point=pd.DataFrame({"Name":Name,"Total":Total})
Total_point

Name=bad.sort_values(by="Total",ascending=False).head(10)["Name"]
Total=bad.sort_values(by="Total",ascending=False).head(10)["Total"]
Total_point=pd.DataFrame({"Name":Name,"Total":Total})
Total_point

Name=neutral.sort_values(by="Total",ascending=False).head()["Name"]
Total=neutral.sort_values(by="Total",ascending=False).head()["Total"]
Total_point=pd.DataFrame({"Name":Name,"Total":Total})
Total_point

mean=df[["Alignment","Total"]].groupby(["Alignment"],as_index=True).mean()
mean=mean.sort_values(by='Total',ascending=False)
mean


Name=df.sort_values(by="Total",ascending=False).head(10)["Name"]
Strength=df.sort_values(by="Total",ascending=False).head(10)["Strength"]
Speed=df.sort_values(by="Total",ascending=False).head(10)["Speed"]
Intelligence=df.sort_values(by="Total",ascending=False).head(10)["Intelligence"]

Ts=pd.DataFrame({"Name":Name,"Strength":Strength,"Speed":Speed,"Intelligence":Intelligence})
Ts
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot

trace1=go.Scatter(
x=Ts["Name"],
y=Ts["Strength"],
mode="lines",
name="Strength",
)
trace2=go.Scatter(
x=Ts["Name"],
y=Ts["Speed"],
mode="lines",
name="Speed",
)
trace3=go.Scatter(
x=Ts["Name"],
y=Ts["Intelligence"],
mode="lines",
name="Intelligence",
)
layout=go.Layout(title="Strength-Speed-Intelligence",xaxis=dict(title="Best_TP"))
fig=go.Figure(data=[trace1,trace2,trace3],layout=layout)
iplot(fig)