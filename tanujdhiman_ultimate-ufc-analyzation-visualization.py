# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv("../input/ultimate-ufc-dataset/most-recent-event.csv")
data.head()
data.describe()
data.shape
data.info()
col = data.columns
col
num_col = data._get_numeric_data().columns
num_col
len(num_col)
a = list(set(col) - set(num_col))
a
data["finish_details"].unique()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data["R_Stance"] = label.fit_transform(data["R_Stance"])
data["gender"] = label.fit_transform(data["gender"])
data["Winner"] = label.fit_transform(data["Winner"])
data["better_rank"] = label.fit_transform(data["better_rank"])
data["weight_class"] = label.fit_transform(data["R_fighter"])
data["B_Stance"] = label.fit_transform(data["B_Stance"])
data["finish"] = label.fit_transform(data["finish"])
data.info()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,9))

sns.heatmap(data.corr(), annot=False, linewidths = 1, cmap="coolwarm", linecolor="k")
number_of_columns = data.columns
numeric_columns = data._get_numeric_data().columns
len(numeric_columns)
data_analyze = data[numeric_columns]
data_analyze
data_analyze.shape
data["title_bout"].index
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data_analyze["title_bout"] = label.fit_transform(data_analyze["title_bout"])
data_analyze['title_bout']
data_analyze
data_analyze.isnull().sum()
data_analyze.hist(figsize= (40, 30))
plt.show()
sns.distplot(data_analyze)
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(data_analyze[:10], cmap=cmap, n_levels=60, shade=True)
import plotly.express as px
fig = px.scatter(data, x="R_odds", y="Winner", 
                 color="R_ev",
                 size='R_ev', 
                 title = "Red Fighter")
fig.show()
fig = px.scatter(data, x="B_ev", y="Winner", 
                 color="B_odds",  
                 title = "Blue Fighter")
fig.show()
X = data_analyze.iloc[:, 0:103].values
y = data_analyze.iloc[:, 103:104].values
X.shape
y