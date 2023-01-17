from __future__ import print_function

from ipywidgets import interact, interactive, fixed

import ipywidgets as widgets
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as plt_ticker

import seaborn as sns

%matplotlib inline
cols_to_keep = np.array(['AGEP','DECADE', 'SEX', 'SCHL'])

ss1 = pd.read_csv('../input/ss14pusa.csv',usecols=cols_to_keep)

ss2 = pd.read_csv('../input/ss14pusb.csv',usecols=cols_to_keep)

ss = pd.concat([ss1,ss2])
gte21 = ss.loc[:,'AGEP']>=21

gte21df = ss.loc[gte21]

gte21df.loc[:,('DECADE')] = np.floor(gte21df.loc[:,('AGEP')]/10)
#plt.hist(gte21df['SCHL'], bins=24)

plt.figure(figsize=(8, 6))

g = sns.countplot(x="SCHL", data=gte21df)

g.set_xticklabels(g.get_xticklabels(), rotation=45)
def createDecadeSubPlots(nrows, chartFilters):

    fig, axes = plt.subplots(nrows, 1, figsize=(7,15))

    chartFilters_list = chartFilters.tolist()

    for row in axes:

        d1 = chartFilters_list.pop()

        rowFilter = gte21df.loc[:,'DECADE']==d1

        x = gte21df.loc[rowFilter,'SCHL']

        title1 = "Decade %0d" % (d1)

        plot(row, x, title1)

    plt.tight_layout()

    plt.show()

    

def plot(axrow, x, title1):

    axrow.hist(x, color=np.random.rand( 3,1))

    axrow.set_title(title1)
chartFilters = np.arange(gte21df.loc[:,'DECADE'].min(),gte21df.loc[:,'DECADE'].max()+1,1)

createDecadeSubPlots(8, chartFilters)
def plotGender(decade, col='SCHL', binSize=10):

    male = gte21df.loc[(gte21df.loc[:,'DECADE']==decade) & (gte21df.loc[:,'SEX']==1.0)]

    female = gte21df.loc[(gte21df.loc[:,'DECADE']==decade) & (gte21df.loc[:,'SEX']==2.0)]

    fig = plt.figure(figsize=(7,12))

    fig.suptitle('Gender by Decade Hist of Schooling')

    ax1 = fig.add_subplot(2, 1, 1)

    ax2 = fig.add_subplot(2, 1, 2)

    

    ax1.hist(male.loc[:,col], color='blue', bins=binSize)

    ax2.hist(female.loc[:,col], color='green', bins=binSize)

    

    ax1.set_title('Males')

    ax2.set_title('Females')

    

    plt.show()
interact(plotGender, decade=widgets.FloatSlider(min=2.0,max=9.0,step=1.0,value=2.0));