import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, warnings, gc

color = sns.color_palette()
warnings.filterwarnings("ignore")
%matplotlib inline

wage = pd.read_csv("../input/Minimum Wage Data.csv", encoding = "Windows-1252")
wage.head()
wage.describe()
wage.info()
diff_count = 0
idex = []

for i, (w1, w2) in enumerate(zip(wage["High.Value"], wage["Low.Value"])):
    if abs(w1 - w2) > 0:
        diff_count += 1
        idex.append(i)
        
print("There are {} times that some states changed wage".format(diff_count))
wage.iloc[idex]
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

print("Plotly environment setup, done!")
data = []
states = wage["State"].unique()

for s in states:
    temp = wage[wage["State"] == s][["Year", "High.Value"]]
    trace = go.Scatter(x = temp["Year"], y = temp["High.Value"], name = s)
    data.append(trace)

layout = dict(title = "US minimum wage from 1967 - 2017",
              xaxis = dict(title = "Year", ticklen = 5, zeroline = False),
              yaxis = dict(title = "Wage", ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)

regions = {"Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
                         "New Jersey", "New York", "Pennsylvania"],
           "Mid-West": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", 
                        "Missouri", "Nebraska", "North Dakota", "South Dakota"],
           "South": ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", 
                     "District of Columbia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee", 
                     "Arkansas", "Louisiana", "Oklahoma", "Texas"],
           "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", 
                    "Alaska", "California", "Hawaii", "Oregon", "Washington"],
           "Other": ["Federal (FLSA)", "Guam", "Puerto Rico", "U.S. Virgin Islands"]}

def finding_regions(state):
    if state in regions["Northeast"]:
        return "Northeast"
    elif state in regions["Mid-West"]:
        return "Mid-West"
    elif state in regions["South"]:
        return "South"
    elif state in regions["West"]:
        return "West"
    elif state in regions["Other"]:
        return "Other"
    
wage["Region"] = wage["State"].apply(finding_regions)
for r in regions.keys():
    data = []
    states = regions[r]
    for s in states:
        temp = wage[wage["State"] == s][["Year", "High.Value"]]
        trace = go.Scatter(x = temp["Year"], y = temp["High.Value"], name = s)
        data.append(trace)

    layout = dict(title = "{} region minimum wage from 1967 - 2017".format(r),
                  xaxis = dict(title = "Year", ticklen = 5, zeroline = False),
                  yaxis = dict(title = "Wage", ticklen = 5, zeroline = False))
    fig = dict(data = data, layout = layout)
    iplot(fig)
