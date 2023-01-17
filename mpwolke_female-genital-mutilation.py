# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsmutilationcsv/mutilation.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'mutilation.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
burkina_faso = df[(df['Country Name']=='Burkina Faso')].reset_index(drop=True)

burkina_faso.head()
Egypt = df[(df['Country Name']=='Egypt')].reset_index(drop=True)

Egypt.head()
Guinea = df[(df['Country Name']=='Guinea')].reset_index(drop=True)

Guinea.head()
Mali = df[(df['Country Name']=='Mali')].reset_index(drop=True)

Mali.head()
Somalia = df[(df['Country Name']=='Somalia')].reset_index(drop=True)

Somalia.head()
Sudan = df[(df['Country Name']=='Sudan')].reset_index(drop=True)

Sudan.head()
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(2*size,2))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("2014", "2014 Year", df,4)
plot_count("2018", "2018 Year", df,4)
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= '2006', data = Somalia, palette="cool",edgecolor="black")

plt.subplot(132)

sns.countplot(x= '2012', data = Guinea, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= '2014', data = Sudan, palette="Greens_r",edgecolor="black")

plt.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= '1999', data = burkina_faso, palette="bone",edgecolor="black")

plt.subplot(132)

sns.countplot(x= '2003', data = burkina_faso, palette="ocean",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= '2006', data = burkina_faso, palette="Purples",edgecolor="black")

plt.show()
df_grp = df.groupby(["Indicator Name","1996"])[["2001", "2004","2006","2009"]].sum().reset_index()

df_grp.head()
Egypt_grp = Egypt.groupby(["Indicator Name","1996"])[["2001", "2004","2006","2009"]].sum().reset_index()

Egypt_grp.head()
Guinea_grp = Guinea.groupby(["Indicator Name","1999"])[["2005", "2012","2016","2018"]].sum().reset_index()

Guinea_grp.head()
Mali_grp = Mali.groupby(["Indicator Name","2001"])[["2006", "2010", "2013", "2015", "2018"]].sum().reset_index()

Mali_grp.head()