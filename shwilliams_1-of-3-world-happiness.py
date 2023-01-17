# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import plotly.plotly as py
import plotly.tools as tls


sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml # data visualisation as well
import warnings
warnings.filterwarnings("ignore")
hap17 = pd.read_csv("../input/2017.csv", sep=",", header=0)
hap16 = pd.read_csv("../input/2016.csv", sep=",", header=0)
hap15 = pd.read_csv("../input/2015.csv", sep=",", header=0)
hap17.shape
hap16.shape

hap15.shape
hap17.head()
hap17.tail()
hap16.head()
hap16.tail()
hap15.head()
hap15.tail()
hap17.corr()
ml.figure(figsize=(20,10)) 
sn.heatmap(hap17.corr(), annot=True)
hap16.corr()
ml.figure(figsize=(20,10)) 
sn.heatmap(hap16.corr(), annot=True)
hap15.corr()
ml.figure(figsize=(20,10)) 
sn.heatmap(hap15.corr(), annot=True)
hap17.describe()
hap16.describe()
hap15.describe()
ml.figure(figsize=(15,10)) 
sn.stripplot(x="Region", y="Happiness Rank", data=hap16, jitter=True)
ml.xticks(rotation=-45)
ml.figure(figsize=(15,10)) 
sn.stripplot(x="Region", y="Happiness Rank", data=hap15, jitter=True)
ml.xticks(rotation=-45)
sn.pairplot(hap17[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.']])
sn.pairplot(hap16[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])
sn.pairplot(hap15[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])