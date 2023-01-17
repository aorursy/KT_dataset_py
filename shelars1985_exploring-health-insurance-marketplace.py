# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import seaborn as sns 

import numpy as np

import pandas as pd

import numpy as np

import random as rnd

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from numpy import genfromtxt

from scipy.stats import multivariate_normal

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score , average_precision_score

from sklearn.metrics import precision_score, precision_recall_curve

%matplotlib inline
CostShare_df = pd.read_csv("../input//BenefitsCostSharing.csv")
CostShare_df.head(n=10)
# Fill empty and NaNs values with NaN

CostShare_df = CostShare_df.fillna(np.nan)



# Check for Null values

CostShare_df.isnull().sum()
print ('Total records in file:%d' %CostShare_df.BenefitName.count())

print ('Unique benefits pesent in the file:%d' %CostShare_df.BenefitName.nunique())
### lets Summarize data

# Summary and statistics

CostShare_df.describe()
v_features = CostShare_df.ix[:,0:32].columns

for i, cn in enumerate(CostShare_df[v_features]):

    print(i,cn)

    print(CostShare_df[cn].describe())

    print("-"*40)
CostShare_df[["BusinessYear","BenefitName"]].groupby('BusinessYear').describe()
CostShare_df[["StateCode","BenefitName"]].groupby('StateCode').count().sort_values("BenefitName")

Unique_State = CostShare_df.StateCode.unique()

benefitarray = []



for state in Unique_State:

    state_benefit =  len(CostShare_df[CostShare_df["StateCode"] == state])    

    benefitarray.append(state_benefit)   


f, ax = plt.subplots(figsize=(15, 15)) 

ax.set_yticklabels(Unique_State, rotation='horizontal', fontsize='large')

g = sns.barplot(y = Unique_State,x=benefitarray)

plt.show()



# Set number of ticks for x-axis

"""

ax.set_xticks(x)

# Set ticks labels for x-axis

ax.set_xticklabels(Unique_State, rotation='vertical', fontsize='small')

plt.show()

"""


df = pd.DataFrame(

    {'state': Unique_State,

     'Count' : benefitarray

     })



df = df.sort_values("Count", ascending=False).reset_index(drop=True)



f, ax = plt.subplots(figsize=(15, 15)) 

ax.set_yticklabels(df.state, rotation='horizontal', fontsize='large')

g = sns.barplot(y = df.state, x= df.Count)

plt.show()



data = dict(type = 'choropleth',

           locations = df['state'],

           locationmode = 'USA-states',

           colorscale = 'YIOrRed',

            text = df['state'],

            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),

           z = df['Count'],

           colorbar = {'title':'No of Benefit plans'})



layout = dict(title = 'Benefit plan spread across state',

         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 



choromap2 = go.Figure(data = [data],layout=layout)

iplot(choromap2)
CostShare_df[["StateCode","BenefitName"]].groupby('StateCode').describe()


#Coinsurance

print('Coinsurance details')

print(CostShare_df.CoinsInnTier1.unique())

print('*'*50)

print(CostShare_df.CoinsInnTier2.unique())

print('*'*50)

print(CostShare_df.CoinsOutofNet.unique())

print('_'*50)

print('_'*50)



"""Copay

print('Copay details')

print(CostShare_df.CopayInnTier1.unique())

print('*'*50)

print(CostShare_df.CopayInnTier2.unique())

print('*'*50)

print(CostShare_df.CopayOutofNet.unique())

print('_'*50)

print('_'*50)

"""



CoinsInnTier1 = []

YearBusiness = []

StateCode = []

CoinsInnTier1_real = np.asarray(CostShare_df.CoinsInnTier1)

            

for i, cn in enumerate(CoinsInnTier1_real):

       if (str(cn) == 'nan' or str(cn) == '$0' or str(cn) == 'Not Applicable') :

             continue     

       else :

             if  cn.replace("%","").strip().split(' ')[0] != 'No' :   

                 CoinsInnTier1.append(cn.replace("%","").strip().split(' ')[0])

                 YearBusiness.append(CostShare_df.BusinessYear[i])

                 StateCode.append(CostShare_df.StateCode[i])

# 1) use Value_counts method

CoinsInnTier1

CoinsInnTier1 = pd.to_numeric(CoinsInnTier1, errors='coerce')

Codf = pd.DataFrame(

    {'Coinsurance1': CoinsInnTier1,

      'YearBusiness' : YearBusiness,

      'StateCode' : StateCode

     })

Codf['Coinsurance1'].value_counts().head(5)
Codf.groupby('YearBusiness').sum()
# 2) write complex function



Coinsarray = []



Unique_Coinsurance = Codf.Coinsurance1.unique()

for Coinsurance in Unique_Coinsurance:

    Freq_Coinsurance =  len(Codf[Codf["Coinsurance1"] == Coinsurance])    

    Coinsarray.append(Freq_Coinsurance) 


Coins_df = pd.DataFrame(

    {'Coinsurance': Unique_Coinsurance,

     'Coinsfrequency' : Coinsarray

     })



Coins_df = Coins_df.sort_values("Coinsfrequency", ascending=False).reset_index(drop=True)



Coins_df
#sns.distplot(Codf['Coinsurance1'],kde=False,bins=15)



fig, ax = plt.subplots(figsize=(15,10), ncols=3, nrows=2)



left   =  0.125  # the left side of the subplots of the figure

right  =  0.9    # the right side of the subplots of the figure

bottom =  0.1    # the bottom of the subplots of the figure

top    =  0.9    # the top of the subplots of the figure

wspace =  .8     # the amount of width reserved for blank space between subplots

hspace =  1.5    # the amount of height reserved for white space between subplots



# This function actually adjusts the sub plots using the above paramters

plt.subplots_adjust(

    left    =  left, 

    bottom  =  bottom, 

    right   =  right, 

    top     =  top, 

    wspace  =  wspace, 

    hspace  =  hspace

)



# The amount of space above titles

y_title_margin = 1.0



ax[0][0].set_title("Year 2014", y = y_title_margin)

ax[0][1].set_title("Year 2015", y = y_title_margin)

ax[0][2].set_title("Year 2016", y = y_title_margin)

ax[1][0].set_title("Box Plot", y = y_title_margin)

ax[1][1].set_title("violin Plot", y = y_title_margin)

ax[1][2].set_title("Strip Plot", y = y_title_margin)



sns.distplot(Codf[Codf['YearBusiness'] == 2014]['Coinsurance1'],kde=False,bins=15,ax=ax[0][0])

sns.distplot(Codf[Codf['YearBusiness'] == 2015]['Coinsurance1'],kde=False,bins=15,ax=ax[0][1])

sns.distplot(Codf[Codf['YearBusiness'] == 2016]['Coinsurance1'],kde=False,bins=15,ax=ax[0][2])

sns.boxplot(x='YearBusiness',y='Coinsurance1',data=Codf,ax=ax[1][0])

sns.violinplot(x='YearBusiness',y='Coinsurance1',data=Codf,ax=ax[1][1])

sns.stripplot(x='YearBusiness',y='Coinsurance1',data=Codf,jitter=True,ax=ax[1][2])

plt.tight_layout()
g = sns.FacetGrid(data=Codf,row='StateCode',col='YearBusiness')

g.map(sns.distplot,'Coinsurance1')