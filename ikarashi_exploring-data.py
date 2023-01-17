# Import basic packages

import pandas as pd

import numpy as np

pd.options.display.max_columns = None



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode()

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import display



# Output plots in notebook

% matplotlib inline

% config InlineBackend.figure_format = 'retina'



import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_data = pd.read_csv("../input/train.csv")

train_data.head()
train_data.shape
print(train_data.columns)
train_data.info(verbose=False)
train_data.isnull().sum().rename("NaN").to_frame().transpose()
macro_data = pd.read_csv("../input/macro.csv")

macro_data.head()
macro_data.shape
macro_data.info(verbose=False)
macro_data.isnull().sum().rename("NaN").to_frame().transpose()
sns.set(style="whitegrid", font_scale=1.3)

plt.figure(figsize=(15,8))

ax = sns.distplot(train_data["price_doc"])

ax.set(xlim=(0,None))

plt.title("Price_doc distribution")
train_data["price_doc"].describe().to_frame().transpose()
corr_data = train_data.corr()

corr_target = corr_data[["price_doc"]]

corr_target["sort"] = corr_target["price_doc"].abs()

corr_target["column_name"] = corr_target.index

corr_target.sort_values("sort", ascending=False, inplace=True)
data = [go.Bar(

            x=corr_target["column_name"][1:15].values,

            y=corr_target["price_doc"][1:15].values

        )]

layout = go.Layout(

            title="Top 15 high correlation variables")

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="basic_bar")
train_data[corr_target.index[1:15]].describe()
train_data[corr_target.index[1:15]].isnull().sum().rename("NaN").to_frame().transpose()
df = pd.merge(train_data, macro_data, on="timestamp")

df.head()
df.info(verbose=False)
df_object_cols = df.columns.drop(df._get_numeric_data().columns)

df_object = df[df_object_cols]

df_object.head()
dummies = pd.get_dummies(df_object.drop("timestamp", axis=1))



# timestamp

dummies["Year"] = df_object["timestamp"].apply(lambda x: int(x.split("-")[0]))

dummies["Month"] = df_object["timestamp"].apply(lambda x: int(x.split("-")[1]))

dummies["Day"] = df_object["timestamp"].apply(lambda x: int(x.split("-")[2]))
df = pd.concat([df._get_numeric_data(), dummies], axis=1)

df.head()
df.info(verbose=False)
df.isnull().sum().rename("NaN").to_frame().transpose()