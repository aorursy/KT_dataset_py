# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import geopandas as gpd



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

df_enroll = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

df_enroll.sort_values(by='Year',inplace=True)

#rename States

df_enroll['State_UT'].replace({

    'MADHYA PRADESH':'Madhya Pradesh',

    'Pondicherry':'Puducherry',

    'Uttaranchal':'Uttar Pradesh'

},inplace=True)
#filter data

filt1 = (df_enroll['State_UT'] =='All India')
df_gre_total = df_enroll.loc[filt1]

boys_col = ['Primary_Boys','Upper_Primary_Boys','Secondary_Boys','Higher_Secondary_Boys']

girls_col = ['Primary_Girls','Upper_Primary_Girls','Secondary_Girls','Higher_Secondary_Girls']
#convert object to float

df_gre_total.loc[:,'Higher_Secondary_Boys']=df_gre_total.loc[:,'Higher_Secondary_Boys'].astype('float')

df_gre_total.loc[:,'Higher_Secondary_Girls']=df_gre_total.loc[:,'Higher_Secondary_Girls'].astype('float')
sns.set(font_scale = 1.11)

sns.set_style("white")

ax = df_gre_total[boys_col].plot.bar(figsize=(15,6))

sns.despine(left=True, bottom=True)



#label and title

ax.set_xticklabels(np.arange(3))

ax.set_title('Gross Enrollment Ratio of Boys in India',size=18)

ax.set_xticklabels(list(df_gre_total['Year']))

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
sns.set(font_scale = 1.11)

sns.set_style("white")

ax = df_gre_total[girls_col].plot.bar(figsize=(15,6))

sns.despine(left=True, bottom=True)

#label and title

ax.set_xticklabels(np.arange(3))

ax.set_title('Gross Enrollment Ratio of Girls in India',size=18)

ax.set_xticklabels(list(df_gre_total['Year']))

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
#filter latest data

filt_year = df_enroll['Year'] =='2015-16'

df_enroll_latest = df_enroll[filt_year]

df_enroll_latest.sort_values(by='Higher_Secondary_Total',ascending=False,inplace=True)

df_enroll_latest['Higher_Secondary_Total'] = df_enroll_latest['Higher_Secondary_Total'].astype('float')
fig,ax = plt.subplots(figsize=(15,6))

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="Primary_Total",palette='Purples_r',

                 data=df_enroll_latest.sort_values(by='Primary_Total',ascending=False).head(12))

sns.despine(left=True, bottom=True)



#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Higher Secondary GRE')



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')



#add title

fig.suptitle('Sates with highest GER in Primary', fontsize=18)
fig,ax = plt.subplots(figsize=(15,6))

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="Higher_Secondary_Total",palette='Blues_r',data=df_enroll_latest.head(12))

sns.despine(left=True, bottom=True)



#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Higher Secondary GRE')



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')



#add title

fig.suptitle('Sates with highest GER in Higher Secondary', fontsize=18)
fig,ax = plt.subplots(figsize=(15,6))

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="Primary_Total",palette='Purples',

                 data=df_enroll_latest.sort_values(by='Primary_Total',ascending=False)[::-1].head(12))

sns.despine(left=True, bottom=True)



#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Higher Secondary GRE')



#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')



#add title

fig.suptitle('Sates with lowest GER in Primary', fontsize=18)
fig,ax = plt.subplots(figsize=(15,6))

sns.set_style("white")

ax = sns.barplot(x="State_UT", y="Higher_Secondary_Total",palette='Blues',

                 data=df_enroll_latest[::-1].head(12))

sns.despine(left=True, bottom=True)

#adjust labels

for item in ax.get_xticklabels():

    item.set_rotation(90)

    item.set_fontsize(12)

ax.set_xlabel('')

ax.set_ylabel('Higher Secondary GRE')

#annotations

for p in ax.patches:

    ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

#add title

fig.suptitle('Sates with lowest GER in Higher Secondary', fontsize=18)    
def plot_state(state):    

    filt_state = (df_enroll['State_UT'] == state)

    df_gre_total = df_enroll.loc[filt_state]

    #convert values

    df_gre_total.loc[:,'Higher_Secondary_Boys']=df_gre_total.loc[:,'Higher_Secondary_Boys'].astype('float')

    df_gre_total.loc[:,'Higher_Secondary_Girls']=df_gre_total.loc[:,'Higher_Secondary_Girls'].astype('float')

    sns.set(font_scale = 1.111)

   

    #figures

    sns.set_style("white")

    fig= plt.figure(figsize=(18,12))

    

    # Divide the figure into a 2x1 grid, and give me the first section

    ax1 = fig.add_subplot(211)

    # Divide the figure into a 2x1 grid, and give me the second section

    ax2 = fig.add_subplot(212)

    df_gre_total[boys_col].plot.bar(ax=ax1)

    df_gre_total[girls_col].plot.bar(ax=ax2)

    sns.despine(left=True, bottom=True)

    

    #label and title

    

    #ax1

    ax1.set_xticklabels(np.arange(3))

    ax1.set_title('GRE of Boys',size=15)

    ax1.set_xticklabels(list(df_gre_total['Year']))

    for tick in ax1.get_xticklabels():

        tick.set_rotation(-0)

    fig.suptitle(f'{state}', fontsize=18)

    

    #ax2

    ax2.set_xticklabels(np.arange(3))

    ax2.set_title('GRE of Girls',size=15)

    ax2.set_xticklabels(list(df_gre_total['Year']))

    for tick in ax2.get_xticklabels():

        tick.set_rotation(-0)





    

    #annotations

    for p in ax1.patches:

        ax1.annotate(format(p.get_height(), '.2f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', 

                       xytext = (0, 9), 

                       textcoords = 'offset points')

    

    #annotations

    for p in ax2.patches:

        ax2.annotate(format(p.get_height(), '.2f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', 

                       xytext = (0, 9), 

                       textcoords = 'offset points')

        

        

    #adjust legend

    ax1.get_legend().remove()

    #custom legend

    import matplotlib.patches as mpatches

    

    primary_patch = mpatches.Patch(color='#29629e', label='Primary')

    upper_primary_patch = mpatches.Patch(color='#ff6d05', label='Upper Primary')

    secondary_patch = mpatches.Patch(color='#226908', label='Secondary')

    higher_secondary_patch = mpatches.Patch(color='#8a1111', label='Higher Secondary')    

    ax2.legend(handles=[primary_patch, upper_primary_patch,secondary_patch,higher_secondary_patch],

               loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4)

    

    
plot_state('Maharashtra')
plot_state('Delhi')
plot_state('Assam')
#You can plot any state you wish!!
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

df_enroll.rename(columns={

    'State_UT':'state'

},inplace=True)



states.rename(columns={

    'st_nm':'state'

},inplace=True)
#filter the latest year data of our data-set and merge them together

latest_filt = df_enroll['Year'] =='2015-16'

states_op = states.merge(df_enroll.loc[latest_filt][1:].sort_values(by='state'),on='state')
#convert the columns to 'float'

cols=['Primary_Boys','Primary_Girls','Primary_Total','Upper_Primary_Boys',

      'Upper_Primary_Girls','Upper_Primary_Total','Secondary_Boys','Secondary_Girls',

      'Secondary_Total','Higher_Secondary_Boys','Higher_Secondary_Girls','Higher_Secondary_Total']

states_op[cols] = states_op[cols].astype('float')
fig,ax = plt.subplots(2,2,figsize=(18,18))

plt.style.use('seaborn')

states_op.plot(column='Primary_Boys',cmap='OrRd',figsize=(9,9),

                    legend=False,ax=ax[0,0])

ax[0,0].set_title('Primary',fontsize=15)

states_op.plot(column='Upper_Primary_Boys',cmap='OrRd',figsize=(9,9),

                    legend=False,ax=ax[0,1])

ax[0,1].set_title('Upper Primary',fontsize=15)

states_op.plot(column='Secondary_Boys',cmap='OrRd',figsize=(9,9),

                    legend=False,ax=ax[1,0])

ax[1,0].set_title('Secondary',fontsize=15)

states_op.plot(column='Higher_Secondary_Boys',cmap='OrRd',figsize=(9,9),

                    legend=True,ax=ax[1,1])

ax[1,1].set_title('Higher Secondary',fontsize=15)



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

fig.suptitle('Distribution of GER of boys across India', fontsize=18)

fig,ax = plt.subplots(2,2,figsize=(18,18))

plt.style.use('seaborn')

states_op.plot(column='Primary_Girls',cmap='RdPu',figsize=(9,9),

                    legend=False,ax=ax[0,0])

ax[0,0].set_title('Primary',fontsize=15)

states_op.plot(column='Upper_Primary_Girls',cmap='RdPu',figsize=(9,9),

                    legend=False,ax=ax[0,1])

ax[0,1].set_title('Upper Primary',fontsize=15)

states_op.plot(column='Secondary_Girls',cmap='RdPu',figsize=(9,9),

                    legend=False,ax=ax[1,0])

ax[1,0].set_title('Secondary',fontsize=15)

states_op.plot(column='Higher_Secondary_Girls',cmap='RdPu',figsize=(9,9),

                    legend=True,ax=ax[1,1])

ax[1,1].set_title('Higher Secondary',fontsize=15)



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

fig.suptitle('Distribution of GER of girls across India', fontsize=18)
#extract the GDP Data from the abovr link

#Link ----->"https://en.wikipedia.org/wiki/List_of_Indian_states_and_union_territories_by_GDP"



df_gdp = pd.read_html('https://en.wikipedia.org/wiki/List_of_Indian_states_and_union_territories_by_GDP_per_capita')[2]
#cleaning up the data for processing

df_gdp.rename(columns={

    'State/Union territory':'state',

    'NSDP Per Capita (Nominal)(2018–19)[1][2]':'GDP_percapita'

},inplace=True)

df_gdp.drop(['Rank','NSDP Per Capita (Nominal)(2019–20)[1]',

         'NSDP Per Capita (Nominal)(2018–19 INT$)','NSDP Per Capita (Nominal)(2019–20 INT$)','NSDP Per capita (PPP)1(2018–19 INT$)[3]'], axis=1,inplace=True)



#further cleaning

def clean(num):

    a = num.strip('₹ ')[0:]

    b = a.replace(',','')

    return b

df_gdp['GDP_percapita'] = df_gdp['GDP_percapita'].apply(clean)

df_gdp['GDP_percapita'].replace('191736[4]','191736',inplace=True)



filt_state= df_gdp['state'] =='India2'

df_gdp =df_gdp.loc[~filt_state]

#still cleaning

states_op['state'].replace({

    'Andaman & Nicobar Islands':'Andaman and Nicobar Islands',

    'Jammu And Kashmir':'Jammu and Kashmir',

    'Maharashtra':'Maharastra'

    

},inplace=True)
#combine and create a new Data Frame

states_gpd = states_op.merge(df_gdp,on='state')

states_gpd = states_gpd.drop(['geometry','Year'],axis=1)

states_gpd.sort_values(by='state',inplace=True)
states_gpd['GDP_percapita'] = states_gpd['GDP_percapita'].astype('float')
#extract the GDP Data from the abovr link

#Link ----->"https://en.wikipedia.org/wiki/List_of_Indian_states_and_union_territories_by_GDP_per_capita"



df_gdp2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_Indian_states_and_union_territories_by_GDP')[1]
#cleaning....

df_gdp2.rename(columns={

    'Nominal GDP(trillion INR, lakh crore ₹)':'nominal_GDP',

    'State/UT':'state'

},inplace=True)

df_gdp2.drop(['Data year[3][4][5][6]','Rank','Comparable country[7]'],axis=1,inplace=True)



#still cleaning.......

def clean2(num):

    a = num.strip(' lakh')[1:6]

    b= float(a)

    bill = 10**12

    return b*bill

df_gdp2['nominal_GDP'] = df_gdp2['nominal_GDP'].apply(clean2)



#still cleaning...............

df_gdp2['state'].replace({

    'Maharashtra':'Maharastra',

    'Jammu & Kashmir':'Jammu and Kashmir',

    'NCT of Delhi':'Delhi',

    'Arunanchal Pradesh':'Arunachal Pradesh',

    'Andaman & Nicobar Island':'Andaman and Nicobar Islands'

},inplace=True)
#combine

states_gpd = states_gpd.merge(df_gdp2,on='state')

states_gpd.sort_values(by='state',inplace=True)
sns.set_style('darkgrid')

# Divide the figure into a 2x1 grid, and give me the first section

fig,ax = plt.subplots(2,2,figsize=(18,12))

ax1 = sns.regplot("Primary_Total", "GDP_percapita", data=states_gpd,color='#b80000',

                     ax=ax[0,0])

ax2 = sns.regplot("Upper_Primary_Total", "GDP_percapita", data=states_gpd,color='#b80000',

                    ax=ax[0,1])

ax3 = sns.regplot("Secondary_Total", "GDP_percapita", data=states_gpd,color='#b80000',

                     ax=ax[1,0])

ax4 = sns.regplot("Higher_Secondary_Total", "GDP_percapita", data=states_gpd,color='#b80000',

                     ax=ax[1,1])
sns.set_style('darkgrid')

# Divide the figure into a 2x1 grid, and give me the first section

fig,ax = plt.subplots(2,2,figsize=(18,12))

ax1 = sns.regplot("Primary_Total", "nominal_GDP", data=states_gpd,color='#008216',

                     ax=ax[0,0])

ax2 = sns.regplot("Upper_Primary_Total", "nominal_GDP", data=states_gpd,color='#008216',

                    ax=ax[0,1])

ax3 = sns.regplot("Secondary_Total", "nominal_GDP", data=states_gpd,color='#008216',

                     ax=ax[1,0])

ax4 = sns.regplot("Higher_Secondary_Total", "nominal_GDP", data=states_gpd,color='#008216',

                     ax=ax[1,1])
columns = ['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total','GDP_percapita','nominal_GDP']
corr_matrix = states_gpd[columns].corr()
corr_matrix.drop(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'],axis=1,inplace=True)

corr_matrix.drop(['GDP_percapita','nominal_GDP'],axis=0,inplace=True)
fig,ax = plt.subplots(figsize=(9,9))

ax = sns.heatmap(corr_matrix,annot=True,cmap="OrRd",

                 yticklabels=['Primary','Upper Primary','Secondary','Higher Secondary'],

                 annot_kws={'size': 18})

for item in ax.get_yticklabels():

    item.set_rotation(0)