import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
notices = pd.read_csv("../input/procurement-notices.csv", parse_dates=["Publication Date", "Deadline Date"])
notices.sample(5)
np.unique(notices["Notice Type"])
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

grouped = notices.groupby(["Publication Date", "Notice Type"])["Project ID"].count().unstack()

data = list()

# for each Notice Type, plot the count of Project IDs
for column in grouped.columns:
    data.append(go.Scatter(
        x = grouped.index,
        y = grouped[column],
        name = column
        )
    )
iplot(data)
# grouped["Contract Award"]
project_by_country = notices.groupby("Country Name", as_index = False)["Project ID"].count()
project_by_country.sample(5)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

iplot([go.Choropleth(
    locationmode='country names',
    locations=project_by_country["Country Name"].values,
    text=project_by_country["Country Name"],
    z=project_by_country["Project ID"]
)])
