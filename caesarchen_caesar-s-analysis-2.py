import pandas as pd

data = pd.read_csv("../input/restaurant-scores-lives-standard.csv")
data.head()
data.info()
data["business_id"].value_counts().head(10)
data["inspection_type"].value_counts()
data["violation_description"].value_counts().head(10)
data["risk_category"].value_counts()
data["risk_category"] = data["risk_category"].astype("category")
data["inspection_score"].value_counts().sort_index()
data["inspection_date"].value_counts().sort_index()
data["inspection_date"] = pd.to_datetime(data["inspection_date"])
data["inspection_year"] = pd.DatetimeIndex(data["inspection_date"]).year
data["inspection_month"] = pd.DatetimeIndex(data["inspection_date"]).month
data["inspection_month_year"] = pd.to_datetime(data["inspection_date"]).dt.to_period('M')
data["inspection_month_year"] = data["inspection_month_year"].astype(str)
#number of business
num_bus = len(data["business_id"].unique())
print("Number of Business")
print(num_bus)
#violation rate
num_isp = len(data["inspection_id"])
num_vio = len(data["violation_id"])-len(data["violation_id"][data["violation_id"].isnull()])
vio_rate = num_vio / num_isp
print("Violation Rate")
print(vio_rate)
#violation rate of 2018
data_2018 = data[data["inspection_year"] == 2018]
num_isp_2018 = len(data_2018["inspection_id"])
num_vio_2018 = len(data_2018["violation_id"])-len(data_2018["violation_id"][data_2018["violation_id"].isnull()])
vio_rate_2018 = num_vio_2018 / num_isp_2018
print("Violation Rate of 2018")
print(vio_rate_2018)
#violation by business
vio_by_bus = num_vio / num_bus
print(vio_by_bus)
#average inspection score
avg_score = data["inspection_score"].mean()
print("Average Score")
print(avg_score)
avg_score_2018 = data_2018["inspection_score"].mean()
print("Average Score of 2018")
print(avg_score_2018)
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="risk_category", data=data)
sns.countplot(x="risk_category", data=data_2018)
data_2018_high = data_2018[data_2018["risk_category"] == "High Risk"]
fig, ax = plt.subplots(1, figsize=(12,8))
data_2018_high["violation_description"].value_counts().plot(kind='bar', ax=ax)
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

data1 = [go.Bar(
            x=data_2018_high["violation_description"].value_counts().index,
            y=data_2018_high["violation_description"].value_counts().reset_index(drop=True)
    )]

layout1 = dict(title = "Top Violation Description With High Risk in 2018",)

fig = dict(data=data1, layout=layout1)
iplot(fig)
# distribution of inspection
ax = data.groupby("inspection_month_year").size().plot.line(figsize = (12,6))
ax.set_title('Inspection Distribution')
data2 = [go.Scatter(x=data["inspection_month_year"].value_counts().sort_index().index, y=data["inspection_month_year"].value_counts().sort_index().reset_index(drop=True))]

layout2 = dict(title = "Number of Inspections by Month",)

fig = dict(data=data2, layout=layout2)
iplot(fig)
labels = data_2018["risk_category"].value_counts().index
values = data_2018["risk_category"].value_counts().reset_index(drop=True)

data3 = [go.Pie(labels=labels, values=values)]
layout3 = dict(title="Risk Category Distribution of 2018",)

fig = dict(data=data3, layout=layout3)
iplot(fig)