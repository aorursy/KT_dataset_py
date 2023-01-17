import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

# load museum data
museums_data = pd.read_csv("../input/museums.csv")



museums_data.keys()
sCity = pd.DataFrame(museums_data['City (Administrative Location)'].value_counts())
sCity.columns = ['Count']
sCity['City (Administrative Location)'] = sCity.index.tolist()
sCity.sort_values(by="Count",ascending=False)
sCity = sCity.reset_index(drop=True)
sCity
plt.figure(figsize=(160,160))
temp = sCity[sCity['Count']>=50]
init_notebook_mode(connected=True)
labels=temp['City (Administrative Location)']
values=temp['Count']
trace=go.Pie(labels=labels,values=values)

iplot([trace])