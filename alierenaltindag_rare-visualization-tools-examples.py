import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

from pandas.tools.plotting import parallel_coordinates

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.info()
dic = {"c1":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"c2":[1,2,3,np.nan,5,6,7,8,np.nan,10,11,12,13,np.nan,15],"c3":[1,2,3,4,5,np.nan,7,8,9,np.nan,11,12,13,14,np.nan]}

data = pd.DataFrame(dic)



import missingno as msno

msno.matrix(df)

plt.show()
msno.bar(df)

plt.show()
df2 = pd.read_csv("../input/world-happiness/2016.csv")
df2.head()
remove_list = ["Country","Happiness Rank","Upper Confidence Interval","Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity"]

df2.drop(remove_list,axis=1, inplace = True)



query = df2[(df2.Region == "Sub-Saharan Africa") | (df2.Region == "Central and Eastern Europe") | (df2.Region == "Latin America and Caribbean")]

regions = query.Region

happiness = query["Happiness Score"]

lower_confidence = query["Lower Confidence Interval"]

economy = query["Economy (GDP per Capita)"]

dystopia = query["Dystopia Residual"]



data = pd.DataFrame({"Region":regions,"Happiness Score":happiness,"Lower Confidence Interval":lower_confidence,"Economy (GDP per Capita)":economy,"Dystopia Residual":dystopia})



# Visualization

plt.figure(figsize=(15,10))

parallel_coordinates(data, 'Region', colormap=plt.get_cmap("Set1"))

plt.title("Visualize according to features")

plt.show()
df2.head()
from matplotlib_venn import venn2

happiness = df2.iloc[:,1]

lower_confidence = df2.iloc[:,2]

venn2(subsets = (len(happiness)-15, len(lower_confidence)-15, 15),set_labels = ("Happiness Score","Lower Confidence Interval"))

plt.show()
df2.columns
feature_names = "Happiness Score","Lower Confidence Interval","Economy (GDP per Capita)","Dystopia Residual"

happiness = len(df2["Happiness Score"])

confidence = len(df2["Lower Confidence Interval"])

economy = len(df2["Economy (GDP per Capita)"])

dystopia = len(df2["Dystopia Residual"])

feature_size = [happiness,confidence,economy,dystopia]

colors = ["red","green","blue","lightblue"]



circle = plt.Circle((0,0),0.4,color = "white")

plt.pie(feature_size, labels = feature_names, colors = colors)

p = plt.gcf()

p.gca().add_artist(circle)

plt.show()