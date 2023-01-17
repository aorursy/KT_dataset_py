import numpy as np

import pandas as pd
df = pd.read_csv("../input/Minimum Wage Data.csv", encoding="Latin")

df.head()
grp = df.groupby("State")

Alb = grp.get_group("Alaska").set_index("Year")

Alb.head()
min_wage_2018 = pd.DataFrame()

for name, group in grp:

  if min_wage_2018.empty:

    min_wage_2018 = group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name})

  else:

    min_wage_2018=min_wage_2018.join(group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name}))

min_wage_2018.head(10)
min_wage_2018.std()
df.set_index("Year")[["Low.2018"]]
df[df["State"]=="Alabama"].set_index("Year").head()
coor_matrix = min_wage_2018.corr()

coor_matrix
list_of_list = []

for column in coor_matrix:

    #list_of_list.append(list(min_wage_2018[column].values))

    list_of_list.insert(0, list(coor_matrix[column].values))

list_of_list
!pip install plotly
list(reversed(list(coor_matrix.columns)))
import plotly

import plotly.plotly as py

import plotly.graph_objs as go



#plotting the heated correlation matrix



trace = go.Heatmap(z=list_of_list,

                   x=list(coor_matrix.columns),

                   y=list(reversed(list(coor_matrix.columns))))

data=[trace]

py.iplot(data, filename='labelled-heatmap')
# Which states Low.2018 column is 0 at least in one year.

df[df["Low.2018"]==0]["State"].unique()
#dropna(axis=0) drops all the row if a cell is NaN in that row. Which is default.

#dropna(axis=1) drops all the column if a cell is NaN in that column.

min_wage_2018.replace(0, np.NaN).dropna(axis=1).corr()