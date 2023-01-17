# This Python 3 environment comes with many helpful analytics libraries installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/India_states.csv")
df.head()
states = df['STATE.NAME'].unique()
print( "There are " , len(states), " States are in the data set")
# Removing records where District name and Statenme are same
df = df[df['DT.Code'] != 0]
# Take out the required columns for analysis remove rest all
df =  df[['STATE.NAME', 'DISTRICT.NAME', 'SUB.DISTRICT.NAME']]
dist_subdist = df[['DISTRICT.NAME', 'SUB.DISTRICT.NAME']]
dist_subdist = dist_subdist.drop_duplicates()
table1 = dist_subdist['DISTRICT.NAME'].value_counts().head(20)
table1
table1.iplot(kind = 'bar', theme = 'solar')
state_dist = df[['STATE.NAME', 'DISTRICT.NAME']]
state_dist = state_dist.drop_duplicates()
table2 = state_dist['STATE.NAME'].value_counts().head(20)
table2
table2.iplot(kind = 'bar', theme = 'solar')
