# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
data.head()
data.shape
data.info()
data.describe()
data.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
data.hist(figsize = (16, 8))
plt.show()
sns.heatmap(data.corr(), annot=True)
plt.show()
import plotly.express as px
import plotly.graph_objects as go
fig = px.scatter(data, x="cases", y="state", 
                 color="cases", 
                 hover_data=['deaths','county', 'date', 'fips'], 
                 title = "Cases")
fig.show()
fig1 = px.scatter(data, x="deaths", y="state", 
                 color="cases", 
                 hover_data=['cases','county', 'date', 'fips'], 
                 title = "Deaths")
fig1.show()
states = data.state.unique()
states
data_frame = pd.DataFrame()
data_frame["state"] = data["state"]
data_frame = np.array(data_frame)
data_frame_case = pd.DataFrame()
data_frame_case["cases"] = data["cases"]
data_frame_case = np.array(data_frame_case)
data_frame_death = pd.DataFrame()
data_frame_death["deaths"] = data["deaths"]
data_frame_death = np.array(data_frame_death)
def state_wise_death(country):
    death = []
    for i in range(len(data_frame)):
        if data_frame[i] == country:
            x1 = data_frame[i]
            e = i
            death.append(data_frame_death[e])
            
    death = np.array(death)
    results = death[:, 0]
    data_frame_part_1 = pd.DataFrame()
    data_frame_part_1["death"] = results
    plt.figure(figsize = (10, 4))
    sns.distplot(data_frame_part_1["death"])
    plt.title("Deaths Graph in " + country)
def state_wise(country):
    case = []
    for i in range(len(data_frame)):
        if data_frame[i] == country:
            x1 = data_frame[i]
            e = i
            case.append(data_frame_case[e])
            
    case = np.array(case)
    result = case[:, 0]
    data_frame_part = pd.DataFrame()
    data_frame_part["cases"] = result
    plt.figure(figsize = (10, 4))
    sns.distplot(data_frame_part["cases"])
    plt.title("Cases Graph in " + country)
country = input("Enter the country name : ")
state_wise(country)
state_wise_death(country)