# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import cufflinks as cf

import plotly.offline as py
py.init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
import plotly.tools as tls 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df_Resources = pd.read_csv("../input/Resources.csv")
df_Schools = pd.read_csv("../input/Schools.csv")
df_Donors = pd.read_csv("../input/Donors.csv")
df_Donations = pd.read_csv("../input/Donations.csv")
df_Teachers = pd.read_csv("../input/Teachers.csv")
df_Projects = pd.read_csv("../input/Projects.csv")
# Any results you write to the current directory are saved as output.
df_Resources.head()
df_Resources.info()
#lets explore the na values
df_Resources.isnull().sum(axis=0)
df_Donors.head()
df_Donors.info()
len(df_Donors['Donor Zip'].unique())
df_Donors.isnull().sum(axis=0)
df_Donations.head()
df_Donations.info()
df_Donations.isnull().sum(axis=0)

df_Schools.head()
df_Schools.head()
df_Schools.isnull().sum(axis=0)
df_Schools['School State'].unique()
scCount = df_Schools.groupby(['School State']).count()

scCount[['School ID']].sort_values('School ID').iplot(kind='bar')
df_Teachers.head()
df_Teachers.info()
df_Teachers.isnull().sum(axis=0)

df_Projects.head()
df_Projects.info()
df_Projects.isnull().sum(axis=0)
df_Projects['Project Resource Category'].unique()
