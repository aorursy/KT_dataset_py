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
import matplotlib.pyplot as plt

import seaborn as sns





from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
# read data

df = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
# check for missing values

percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending = False)

missing_value_df_40 = missing_value_df[missing_value_df['percent_missing']>40]

missing_value_df_40.index
# drop columns based on amount of missing data

df = df.drop(['gsubname3', 'weapsubtype4_txt', 'weapsubtype4', 'weaptype4',

       'weaptype4_txt', 'claimmode3', 'claimmode3_txt', 'gsubname2', 'claim3',

       'guncertain3', 'gname3', 'divert', 'attacktype3', 'attacktype3_txt',

       'ransomnote', 'ransompaidus', 'ransomamtus', 'claimmode2',

       'claimmode2_txt', 'ransompaid', 'corp3', 'targsubtype3',

       'targsubtype3_txt', 'natlty3_txt', 'natlty3', 'target3', 'targtype3',

       'targtype3_txt', 'ransomamt', 'weapsubtype3_txt', 'weapsubtype3',

       'weaptype3_txt', 'weaptype3', 'claim2', 'guncertain2', 'gname2',

       'resolution', 'kidhijcountry', 'nhours', 'compclaim', 'gsubname',

       'attacktype2', 'attacktype2_txt', 'ndays', 'approxdate', 'corp2',

       'nreleased', 'targsubtype2', 'targsubtype2_txt', 'natlty2',

       'natlty2_txt', 'hostkidoutcome_txt', 'hostkidoutcome', 'target2',

       'targtype2_txt', 'targtype2', 'weapsubtype2', 'weapsubtype2_txt',

       'weaptype2', 'weaptype2_txt', 'nhostkidus', 'nhostkid', 'claimmode_txt',

       'claimmode', 'related', 'addnotes', 'alternative', 'alternative_txt',

       'propvalue', 'scite3', 'motive', 'location', 'propcomment',

       'propextent', 'propextent_txt', 'scite2', 'ransom'], axis = 1)
# drop columns with non-useful information for the analysis/missing values

df = df.drop(['latitude','longitude','summary','targsubtype1','guncertain1',

       'targsubtype1_txt','corp1','weapsubtype1', 'weapsubtype1_txt',

       'weapdetail','nkillus', 'nkillter','nwoundus','nperps', 'nperpcap', 'claimed',

       'nwoundte','scite1','INT_LOG','INT_IDEO', 'INT_MISC', 'INT_ANY','dbsource','provstate'], axis = 1)
# drop remaining nan values by rows and assign to new dataframe

df_clean = df.dropna()
df_clean.success.value_counts().plot(kind='pie')
plt.figure(figsize=(25,8))

plt.box(False)

plt.title("Terrorattacks by Region", fontweight="bold", fontsize = 14)

regionchart = sns.countplot(df_clean.region_txt, order = df_clean.region_txt.value_counts().index, palette="dark")

regionchart.set_xticklabels(regionchart.get_xticklabels(), rotation=45)
# plot terror attacks by year

year = df_clean['eventid'].groupby(df_clean['iyear']).count()

plt.figure(figsize=(20,8))

ax = sns.lineplot(data=year)

plt.box(False)

ax.set(xticks=year.index.values[::3])
# plot count of type of weapon used

plt.figure(figsize=(20,10))

weapon = sns.countplot(df_clean.weaptype1_txt, order = df_clean.weaptype1_txt.value_counts().index, palette="rocket")

weapon.set_xticklabels(weapon.get_xticklabels(), rotation=45)

plt.box(False)
# plot causalities by nationality

plt.figure(figsize=(20,10))

nat = sns.countplot(df_clean.natlty1_txt, order = df_clean.natlty1_txt.value_counts().index[:10])

nat.set_xticklabels(nat.get_xticklabels(), rotation=45)

plt.box(False)
# plot terror attacks by month

df_clean = df_clean[df_clean.imonth != 0] #remove some 0 values that have been in the data

df_clean['eventid'].groupby(df['imonth']).count().plot(kind='bar')
df_clean.suicide.value_counts().plot(kind="pie", title="Suicide attack Yes/No")
plt.figure(figsize=(25,8))

plt.box(False)

plt.title("Method of Attack",fontsize=15,fontweight="bold")

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

atttype= sns.countplot(df_clean.attacktype1_txt,order = df_clean.attacktype1_txt.value_counts().index, palette =  'gist_heat')

atttype.set_xticklabels(atttype.get_xticklabels(), rotation=45)
terror_region = pd.crosstab(df_clean.iyear,df_clean.region_txt)

terror_region.plot(color = sns.color_palette('tab10'))

fig = plt.gcf()

fig.set_size_inches(18,6)

plt.box(False)

plt.legend(title='Region')

plt.title("Development Terror by Region")

plt.show()
#df_clean[df_clean.ishostkid == -9] # checking if missing values occur systematic or random

df_clean = df_clean[df_clean.ishostkid != -9] #remove -9 values that have been in the data
plt.figure(figsize=(25,8))

plt.box(False)

plt.title("Target",fontsize=15,fontweight="bold")

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

targtype= sns.countplot(df_clean.targtype1_txt,order = df_clean.targtype1_txt.value_counts().index, palette =  'cividis')

targtype.set_xticklabels(targtype.get_xticklabels(), rotation=90)