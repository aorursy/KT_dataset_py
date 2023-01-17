import pandas as pd

import plotly

plotly.__version__

import plotly.plotly as py

import plotly.graph_objs as go

import numpy as np

import os 

import sys



cwd=os.getcwd()

cwd

os.chdir('/kaggle/input/')

all_list=os.listdir()

if len(all_list)<12:

    os.chdir('/kaggle/input/earn-your-6-figure-prize/')

                  
FTHT6 = pd.read_csv('FT_HT6.csv')
ranks6 = pd.read_csv('ranks6.csv')
winrate6 = pd.read_csv('winrate6.csv')
country6 = pd.read_csv('country6.csv')
names6 = pd.read_csv('names6.csv')

fresults6 = pd.read_csv('fresults6.csv')

mask1 =  (FTHT6.iloc[:,1]>1.6) & (FTHT6.iloc[:,0]>2.4) & ((ranks6.iloc[:,0]-ranks6.iloc[:,1]).abs()>9)
mask2 = ((winrate6.iloc[:,0]-winrate6.iloc[:,1]).abs()>38) & ~((winrate6<35).all(1)) & (country6.iloc[:,0]!='England')
mask = mask1 & mask2
country6[mask]
fresults6[mask]
names6[mask]