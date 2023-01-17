# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version atleast>= 4.8.1
import cufflinks as cf
# For Notebooks
init_notebook_mode(connected=True)
# For offline use
cf.go_offline()
df=pd.read_excel("../input/covid.xlsx")


#df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())


df.head()

df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2.head()
df.iplot(kind='scatter',x='Total Confirmed cases*',y='Deaths**',mode='markers',size=10)
df.iplot(kind='bar',x='Name of State / UT',y='Total Confirmed cases*')
df.iplot(kind='bar',x='Name of State / UT',y='Deaths**',color="red")
df.iplot(kind='bar',x='Name of State / UT',y='Cured/Discharged/Migrated*',color="pink")

df.iplot(kind='bar',x='Name of State / UT',y='Active Cases*')
df.columns
df.drop(columns=["S. No.","Name of State / UT"]).iplot(kind='box')
#df = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df.drop(columns=["S. No.","Name of State / UT"]).iplot(kind='surface',colorscale='rdylbu')
df.drop(columns=["S. No.","Name of State / UT"]).iplot(kind='spread')
df['A'].iplot(kind='hist',bins=25)
df.iplot(kind='bubble',x='A',y='B',size='C')

