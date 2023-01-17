# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import geopandas as gpd

%matplotlib inline

#hide warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load dataset

df_comp = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')

df_comp.sort_values(by='year',inplace=True)
#rename States

df_comp['State_UT'].replace({

    'MADHYA PRADESH':'Madhya Pradesh',

    'Pondicherry':'Puducherry',

    'Uttaranchal':'Uttar Pradesh'

},inplace=True)
columns_tolot = ['Primary_Only','Primary_with_U_Primary','Primary_with_U_Primary_Sec','Primary_with_U_Primary_Sec_HrSec','All Schools']
sns.set(font_scale = 1.11)

sns.set_style("white")



#filt India data

filt1 = (df_comp['State_UT'] =='All India')

ax = df_comp.loc[filt1][columns_tolot].plot.bar(figsize=(15,6))

sns.despine(left=True, bottom=True)

#label and title

ax.set_xticklabels(np.arange(3))

ax.set_title('Schools with computers India',size=18)

ax.set_xticklabels(list(df_comp.loc[filt1]['year']))

for tick in ax.get_xticklabels():

    tick.set_rotation(-0)



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

#adjust legend

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4)

    
filt_year = df_comp['year'] =='2015-16'

df_comp_latest = df_comp[filt_year]

df_comp_latest.sort_values(by='All Schools',ascending = False,inplace =True)
fig,ax = plt.subplots(figsize=(15,6))

sns.set(font_scale = 1.11)

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="All Schools",palette='Greens_r',data=df_comp_latest.head(12))

sns.despine(left=True, bottom=True)





#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Percentage of schools with computer',fontsize=12)



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

fig,ax = plt.subplots(figsize=(15,6))

sns.set(font_scale = 1.11)

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="All Schools",palette='Greens',data=df_comp_latest[::-1].head(12))

sns.despine(left=True, bottom=True)



#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Percentage of schools with computer',fontsize=12)



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')
def state_to_plot(state):

    sns.set(font_scale = 1.11)

    sns.set_style("white")



    #filt state

    filt_state = (df_comp['State_UT'] == state)

    ax = df_comp.loc[filt_state][columns_tolot].plot.bar(figsize=(15,6))

    sns.despine(left=True, bottom=True)

    #label and title

    ax.set_xticklabels(np.arange(3))

    ax.set_title(f'{state} in detail',size=21)

    ax.set_xticklabels(list(df_comp.loc[filt_state]['year']))

    for tick in ax.get_xticklabels():

        tick.set_rotation(-0)



    #annotations

    for p in ax.patches:

        ax.annotate(format(p.get_height(), '.2f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', 

                       xytext = (0, 9), 

                       textcoords = 'offset points')

    #adjust legend

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4)

#Delhi in detail

state_to_plot('Delhi')
#Uttar Pradesh in detail

state_to_plot('Uttar Pradesh')
#Kerala in detail

state_to_plot('Kerala')
#you can plot any state you wish!!!
#load the shp file

states = gpd.read_file('/kaggle/input/india-states/Igismap/Indian_States.shp')
#adjust the names of the SHP file and our dataset so that they match

states['st_nm'].replace({

    'Andaman & Nicobar Island':'Andaman & Nicobar Islands',

    'Arunanchal Pradesh':'Arunachal Pradesh',

    'NCT of Delhi':'Delhi',

    'Jammu & Kashmir':'Jammu And Kashmir',

    'Dadara & Nagar Havelli':'Dadra & Nagar Haveli'    

},inplace=True)



#change both column to the same name

df_comp.rename(columns={

    'State_UT':'state'

},inplace=True)



states.rename(columns={

    'st_nm':'state'

},inplace=True)
#filter the latest year data of our data-set and merge them together

latest_filt = df_comp['year'] =='2015-16'

states_op = states.merge(df_comp.loc[latest_filt].sort_values(by='state'),on='state')
fig,ax = plt.subplots(figsize=(15,9))

states_op.plot(column='All Schools',cmap='Greens',figsize=(9,9),

                    legend=True,ax=ax,legend_kwds={'label': "",'orientation': "vertical",'shrink': 0.6})



#hide axes and ticks

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.set_xticks([]) 

ax.set_yticks([]) 

for item in ax.get_xticklabels():

    item.set_visible(False)

for item in ax.get_yticklabels():

    item.set_visible(False) 

ax.set_facecolor('#c9c9c9')

fig.suptitle('Schools with Computers across India', fontsize=18)
fig,ax = plt.subplots(2,2,figsize=(18,18))

plt.style.use('seaborn')

states_op.plot(column='Primary_Only',cmap='Greens',figsize=(9,9),

                    legend=False,ax=ax[0,0])

ax[0,0].set_title('Primary ',fontsize=15)

states_op.plot(column='Primary_with_U_Primary',cmap='Greens',figsize=(9,9),

                    legend=False,ax=ax[0,1])

ax[0,1].set_title('Upper Primary and Primary',fontsize=15)

states_op.plot(column='Primary_with_U_Primary_Sec',cmap='Greens',figsize=(9,9),

                    legend=False,ax=ax[1,0])

ax[1,0].set_title('Secondary,Upper Primary and Primary',fontsize=15)

states_op.plot(column='Primary_with_U_Primary_Sec_HrSec',cmap='Greens',figsize=(9,9),

                    legend=True,ax=ax[1,1])

ax[1,1].set_title('Higher Secondary,Secondary,Upper Primary and Primary',fontsize=15)



#add background color

ax[0,0].set_facecolor('#e0e0e0')

ax[0,1].set_facecolor('#e0e0e0')

ax[1,0].set_facecolor('#e0e0e0')

ax[1,1].set_facecolor('#e0e0e0')



#hide axes and ticks

for i in np.arange(2):

    for j in np.arange(2):

        ax[i,j].spines['top'].set_visible(False)

        ax[i,j].spines['right'].set_visible(False)

        ax[i,j].spines['left'].set_visible(False)

        ax[i,j].spines['bottom'].set_visible(False)

        for item in ax[i,j].get_xticklabels():

            item.set_visible(False)

        for item in ax[i,j].get_yticklabels():

            item.set_visible(False) 

        ax[i,j].set_xticks([]) 

        ax[i,j].set_yticks([]) 

fig.suptitle('Further Breakdown of Schools with computer in India', fontsize=18)
#load dataset

df_enroll = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

df_enroll.sort_values(by='Year',inplace=True)



#rename States

df_enroll['State_UT'].replace({

    'MADHYA PRADESH':'Madhya Pradesh',

    'Pondicherry':'Puducherry',

    'Uttaranchal':'Uttar Pradesh'},inplace=True)

df_enroll.rename(columns={

    'State_UT':'state'},inplace=True)



#get rid of unwnated columns

df_enroll.drop(['Primary_Boys','Primary_Girls','Upper_Primary_Boys','Upper_Primary_Girls',

                'Secondary_Boys','Secondary_Girls','Higher_Secondary_Boys','Higher_Secondary_Girls'],axis=1,inplace=True)



df_comp.drop(['U_Primary_Only','U_Primary_With_Sec_HrSec','U_Primary_With_Sec','HrSec_Only','Sec_Only','Sec_with_HrSec.'],axis=1,inplace=True)
#filter the latest year data of our data-set and merge them together

latest_filt1 = df_enroll['Year'] =='2015-16'

latest_filt2 = df_comp['year'] =='2015-16'

df_enroll_comp = df_enroll.merge(df_comp.loc[latest_filt],on='state')

df_enroll_comp.drop(['year'],axis=1,inplace =True)
#filter typos,missing values and convert all columns to float

filt_data1 = df_enroll_comp['Higher_Secondary_Total']=='NR'

filt_data2 = df_enroll_comp['Higher_Secondary_Total']== '@'



df_enroll_comp = df_enroll_comp.loc[~(filt_data1^filt_data2)]

df_enroll_comp['Higher_Secondary_Total'] = df_enroll_comp['Higher_Secondary_Total'].astype('float')
sns.set_style('darkgrid')

sns.set(font_scale = 1.11)

# Divide the figure into a 2x1 grid, and give me the first section

fig,ax = plt.subplots(2,2,figsize=(18,18))

ax1 = sns.regplot("Primary_Only", "Primary_Total", data=df_enroll_comp,color='#a30000',

                     ax=ax[0,0])

ax1.set_title('Primary',fontsize=15)



ax2 = sns.regplot("Primary_with_U_Primary", "Upper_Primary_Total", data=df_enroll_comp,color='#a30000',

                    ax=ax[0,1])

ax2.set_title('Upper Primary',fontsize=15)



ax3 = sns.regplot("Primary_with_U_Primary_Sec", "Secondary_Total", data=df_enroll_comp,color='#a30000',

                     ax=ax[1,0])

ax3.set_title('Secondary',fontsize=15)

ax4 = sns.regplot("Primary_with_U_Primary_Sec_HrSec", "Higher_Secondary_Total", data=df_enroll_comp,color='#a30000',

                     ax=ax[1,1])

ax4.set_title('Higher Secondary',fontsize=15)



#hide axes and ticks

for i in np.arange(2):

    for j in np.arange(2):

        ax[i,j].set_xlabel('percent schools with computer') 

        ax[i,j].set_ylabel('GER') 