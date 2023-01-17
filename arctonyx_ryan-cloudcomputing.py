# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot,plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.tools import FigureFactory as ff



from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.head()
df.info()
df2 = df.iloc[:10]

df2
JumlahPenjualanBerdasarkanGenre = df.groupby(['Genre'])['Platform'].count()

JumlahPenjualanBerdasarkanGenre



plt.barh(JumlahPenjualanBerdasarkanGenre.index, JumlahPenjualanBerdasarkanGenre)

plt.ylabel("Genre")

plt.title("Jumlah Penjualan")

plt.show()
df.notnull().all()
import missingno as msno



msno.bar(df)

msno.matrix(df)

msno.heatmap(df)

plt.show()
df['Year'] = df2['Year'].astype(int)

df.head()
sns.countplot(df.Genre)

plt.title("Genre",color = 'Red',fontsize=15)

plt.xticks(rotation=45)

plt.show()
df2=df.head(100)

df2=df.loc[:,["Year","Platform","NA_Sales","EU_Sales"]]

df2["index"]=np.arange(1,len(df2)+1)



fig = ff.create_scatterplotmatrix(df2, 

                                  diag='box', 

                                  index='index',

                                  colormap='Portland',

                                  colormap_type='seq',

                                  height=1000, width=1200)

iplot(fig)

plt.show()