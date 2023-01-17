# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import base64
wrp_reg = pd.read_csv('../input/wrp-reg/regional.csv')
wrp_reg
wrp_reg.head(3)
print(wrp_reg['region'].unique())
#fig = plt.figure(figsize=(10, 8))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.viridis_r

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(1)

christianity_year = wrp_reg.groupby(['year','region']).christianity_all.sum()

christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(20,10) , legend=False)

axes[0].set_title('Christianity Followers',y=1.08,size=12)

axes[0].set_ylabel('Billions', color='gray')



# plt.subplot(2)

islam_year = wrp_reg.groupby(['year','region']).islam_all.sum()

islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Islam Followers',y=1.08,size=12)

axes[1].set_ylabel('Billions', color='gray')



# plt.subplot(3)

judaism_year = wrp_reg.groupby(['year','region']).judaism_all.sum()

judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap , grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Judaism Adherents',y=1.08,size=12)

axes[2].set_ylabel('Billions', color='gray')



plt.tight_layout()

plt.show()
#fig = plt.figure(figsize=(10, 8))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.autumn_r

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(1)

hinduism_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).hinduism_all.sum()

hinduism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(20,10) , legend=False)

axes[0].set_title('Hindusim Followers',y=1.08,size=12)



# plt.subplot(2)

sikhism_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).sikhism_all.sum()

sikhism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Sikhism Followers',y=1.08,size=12)



# plt.subplot(3)

jainism_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).jainism_all.sum()

jainism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Jainism Followers',y=1.08,size=12)



plt.tight_layout()

plt.show()
#fig = plt.figure(figsize=(8, 5))

fig, axes = plt.subplots(nrows=1, ncols=3)

colormap = plt.cm.tab20_r

# fig = plt.figure(figsize=(20, 10))

# plt.subplot(1)

buddhist_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).buddhism_all.sum()

buddhist_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(20,10) , legend=False)

axes[0].set_title('Buddhist Followers',y=1.08,size=12)



# plt.subplot(2)

taoism_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).taoism_all.sum()

taoism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)

axes[1].set_title('Taoist Followers',y=1.08,size=12)



# plt.subplot(3)

shinto_year = wrp_reg[wrp_reg['region'] != 'Asia'].groupby(['year','region']).shinto_all.sum()

shinto_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])

axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},

           ncol=5, mode="expand", borderaxespad=0.)

axes[2].set_title('Shinto Followers',y=1.08,size=12)



plt.tight_layout()

plt.show()
wrp_glbe = pd.read_csv('../input/wp-glbe/wp_glbe.csv')