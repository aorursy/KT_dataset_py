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



df = pd.read_csv('/kaggle/input/cancer-incidence-data-in-korea-as-of-2017/cancer_data_upto2017.csv', encoding='euc-kr')



df1 = df[(df['gender']!='total') & (df['unit2']!='nopatient')]

df1.tail(10)
df2 = df1.melt(id_vars=["cancertype", "gender", "agegroup"], var_name='year', value_name='rate_per_100k', \

               value_vars=['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])



#최근 10년치 데이터만 사용

df2 = df2[(df2['year']>='2008')]

#합계 및 연령 미상 데이터 제거

df2 = df2[(df2['agegroup']!='unknown') & (df2['agegroup']!='total')]

#df2 = df2[(df2['agegroup']!='unknown')]





df2['age_b'] = df2['agegroup'].str.split('-').str[0]

df2['age_e'] = df2['agegroup'].str.split('-').str[1]



#df2['age_b'] = df2['agegroup'].where(df2['agegroup']=='total').replace('total', '99')

#df2['age_e'] = df2['agegroup'].where(df2['agegroup']=='total').replace('total', '99')



#85+

df2['age_b'] = df2['age_b'].replace('85+', '85')

df2['age_e'] = df2['age_e'].replace(np.nan, '89')







df2.age_b.astype(int)

df2.age_e.astype(int)



#



df2['gender'] = df2['gender'].replace('male', 1)

df2['gender'] = df2['gender'].replace('female', 2).astype(int)

df3 = df2

df2.tail(50)
#label encoding



import numpy as np



df3['cancertype_num'] = np.unique(df2['cancertype'], return_inverse=True)[1]

#df3['gender'] = np.unique(df2['gender'], return_inverse=True)[1]



'''

df_CAGR3 = df2[(df2['year']=='2017')].values / df2[(df2['year']=='2014')].values

df_CAGR3.head(20)

'''
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import squarify





pd.options.mode.chained_assignment = None



gender_map = {'male':1, 'female':2}



df_sum1 = df3[(df3['year']=='2017')].groupby(['cancertype', 'agegroup']).sum().reset_index()

df_sum1.head(10)



# Make data.

#@interact

def show_data(cancertype=df3['cancertype'].unique(), \

              gender=['male','female'], \

             angle_x=-30, \

             angle_y=50, \

             size=100):

    

    gndr = gender_map.get(gender)

    df4 = df3.loc[(df3['cancertype']==cancertype) & (df3['gender']==gndr)]

    df4['year'] = pd.to_numeric(df3['year'])

    df4['age_b'] = pd.to_numeric(df3['age_b'])

    fig = plt.figure(figsize=(10*size/100,7*size/100))

    ax = fig.gca(projection='3d')



    ax.plot_trisurf(df4.year, df4.age_b, df4.rate_per_100k, cmap=cm.bwr, linewidth=0, antialiased=False, shade=True, alpha=0.9)

    ax.view_init(angle_y, angle_x)

    plt.title(cancertype + ' / ' + str(gender) + ' / cancer rate by age and calendar year')

    plt.show()

    

    #plt.show()



#show_data(cancertype='all(C00-C96)', gender=1, angle_x=-30, angle_y=50)



interact(show_data, angle_x=(-100, 100, 10), angle_y=(-100, 100, 10), size=(100, 200, 10))



#암종별 성별/연령대별 비중 chart 추가
df_sum = df3[(df3['year']=='2017')].drop('cancertype_num', axis=1).groupby(['cancertype', 'gender']).sum().reset_index()







df_sum_male = df_sum[(df_sum['gender']==1)]

df_sum_female = df_sum[(df_sum['gender']==2)]



df_sum_male.sort_values(by=['rate_per_100k'], axis=0, ascending=False, inplace=True)

df_sum_female.sort_values(by=['rate_per_100k'], axis=0, ascending=False, inplace=True)



df_sum_male = df_sum_male[(df_sum_male['cancertype'] != 'all(C00-C96)')]

df_sum_female = df_sum_female[(df_sum_female['cancertype'] != 'all(C00-C96)')]



df_topx_male = df_sum_male.head(10)

df_topx_female = df_sum_female.head(10)



df_topx_female.head(20)



import matplotlib

import matplotlib.pyplot as plt

import squarify

import platform



# treemap parameters

x = 0.

y = 0.

width = 100.

height = 100.

cmap = matplotlib.cm.viridis



#Utilise matplotlib to scale our goal numbers between the min and max, then assign this scale to our values.

norm_male = matplotlib.colors.Normalize(vmin=min(df_topx_male.rate_per_100k), vmax=max(df_topx_male.rate_per_100k))

colors_male = [matplotlib.cm.Blues(norm_male(value)) for value in df_topx_male.rate_per_100k]



norm_female = matplotlib.colors.Normalize(vmin=min(df_topx_female.rate_per_100k), vmax=max(df_topx_female.rate_per_100k))

colors_female = [matplotlib.cm.Blues(norm_female(value)) for value in df_topx_female.rate_per_100k]



# labels for squares

labels_male = ["%s\n%d" % (label) for label in zip(df_topx_male.cancertype, df_topx_male.rate_per_100k)]

labels_female = ["%s\n%d" % (label) for label in zip(df_topx_female.cancertype, df_topx_female.rate_per_100k)]



# make plot

fig1 = plt.figure(figsize=(20, 15))

ax1 = fig1.add_subplot(121, aspect="equal")

ax1 = squarify.plot(df_topx_male.rate_per_100k, label=labels_male, ax=ax1, alpha=.6, color=colors_male, text_kwargs={'fontsize':9})

plt.axis('off')

#plt.show()

plt.title("Cancer diagnosis top 10 for male, 2017",fontsize=23,fontweight="bold")



ax2 = fig1.add_subplot(122, aspect="equal")

ax2 = squarify.plot(df_topx_female.rate_per_100k, label=labels_female, ax=ax2, alpha=.6, color=colors_female, text_kwargs={'fontsize':9})

plt.axis('off')

plt.title("Cancer diagnosis top 10 for female, 2017",fontsize=23,fontweight="bold")

plt.show()

# use this if you want to draw a border between rectangles

# you have to give both linewidth and edgecolor

# ax = squarify.plot(df2.superf, color=colors, label=labels, ax=ax, alpha=.7,

#                    bar_kwargs=dict(linewidth=1, edgecolor="#222222"))