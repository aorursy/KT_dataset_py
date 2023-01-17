# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#---------------------------------------------------------
# Stats Libs
#---------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#---------------------------------------------------------
# System Libs
#---------------------------------------------------------
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

#---------------------------------------------------------
# Plotting libs
#---------------------------------------------------------
import matplotlib.pyplot as plt
%matplotlib inline

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import cufflinks as cf

#---------------------------------------------------------
# Jupyter specific libs
#---------------------------------------------------------
from IPython.display import display, Markdown
df = pd.read_csv("../input/daily-inmates-in-custody.csv")
#df.info() # inspect the input data...noting DISCHARGED_DT column is extraneous
df.drop(columns="DISCHARGED_DT", inplace=True) # delete as nothing but N/A's
# Decrypting some of the data a bit
df.rename(columns={"RACE":"RACE_CODE"}, inplace=True)
df["RACE"] = df["RACE_CODE"]
df.RACE.replace({"A":"Asian","B":"Black","W":"White","O":"Other", "U":"Unknown","I":"Native"}, inplace=True)
# some quick peeks at the dataset
#print(df.RACE.unique())
#df.sample(5)
#display(Markdown("# Males >> Females"))
#df.groupby("GENDER").nunique()
# start looking at age, splitting by gender
display(Markdown("# Interesting hint of bimodal distribution at late-twenties & early-fifties"))

## interactive plot
data = [go.Histogram(x=df.loc[df['GENDER'] == 'M']["AGE"], name='Males', xbins=dict(start=df['AGE'].min(),end=df['AGE'].max(),size=1)), 
        go.Histogram(x=df.loc[df['GENDER'] == 'F']["AGE"], name='Females')]
iplot(data)

## static plot
#df.hist(column="AGE", bins=100, by='GENDER', figsize=(15, 5))
# add race into the split & utilize boxplots to get better idea of statistics
#df.groupby("GENDER").boxplot(column="AGE", by="RACE", figsize=(20, 10))
display(Markdown("# Clear Racial Representation Issues in NY Inmate Population"))
fig, ax = plt.subplots(figsize=(15,7))
ax = df.groupby(['GENDER','RACE']).count()['INMATEID'].plot.bar()
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# next look at:
# --mental health indictations -> bar or pie chart?
# --arrests across time -> time series