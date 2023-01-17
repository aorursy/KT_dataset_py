# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-4:] == '.csv':

            filepath = os.path.join(dirname, filename)

            #print(filepath)

            try:

                df = pd.read_csv(filepath)

                if 'county' in df.columns:

                    print(filepath)

            except:

                print('The CSV made a boo boo', filepath)



# Any results you write to the current directory are saved as output.
soc = pd.read_csv('/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv')

nytests = pd.read_csv('/kaggle/input/uncover/UNCOVER/ny_dept_of_health/new-york-state-statewide-covid-19-testing.csv')

# County plus total population and estimate percentage cols

cols = ['county', 'e_totpop'] + [c for c in soc.columns if c[:3] == 'ep_']

tsoc = soc[soc['st_abbr'] == 'NY'][cols]



# Only interested in data of performed tests

tnytests = nytests[nytests['cumulative_number_of_tests_performed']>0]



combo = pd.merge(tnytests, tsoc, how='left')

#combo.head()
mapping = { 'new_positives': 'p_newpos',

            'cumulative_number_of_positives' : 'p_cpos', 

            'total_number_of_tests_performed' : 'p_tests',

            'cumulative_number_of_tests_performed' : 'p_ctests'}



for k in mapping:

    combo[mapping[k]] = (combo[k] / combo['e_totpop']) * 100

    

#combo.head()
cols = ['p_cpos'] + [c for c in combo.columns if 'ep_' in c]

corr = combo[combo.test_date==combo.test_date.max()][cols].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

#sns.heatmap(corr, mask= mask, cmap=cmap, vmax=.3, center=0,

#            square=True, linewidths=.5, cbar_kws={"shrink": .5})



temp = corr['p_cpos']

ax = temp[temp<1].sort_values().plot(kind='barh', 

                                     figsize=(15,10),

                                     title='Correlation of COVID-19 cases in NY with social vulnerability measures')
temp = corr['p_cpos']

top_corrs = temp[(abs(temp)>0.4) & (temp<1)].sort_values(ascending=False).index.values

print('Vulnerability measures with absolute correlation value above 0.4:')

for i in top_corrs:

    print('\t-', i)
cols = ['p_cpos'] + [c for c in top_corrs]

tdf = combo[combo.test_date==combo.test_date.max()].sort_values(by='p_cpos')

f, axes = plt.subplots(1,11, figsize=(30,30))

#f.suptitle('Banana')

for i,col in enumerate(cols):

    axt = axes[i].set_title(col)

    ax = axes[i].barh(tdf['county'], tdf[col])

    yl = axes[i].set_ylim(0,62)

    xg = axes[i].grid('both')

    #xt = axes[i].set_xticklabels([])

    if i>0:

        yt = axes[i].set_yticklabels([])
cc = pd.read_csv('/kaggle/input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv')

tcc = cc[cc.date==cc.date.max()].copy()

tsoc = soc[['state', 'county', 'e_totpop']+[c for c in top_corrs]].copy()

for col in ['state', 'county']:

    tcc[col] = tcc[col].str.lower()

    tsoc[col] = tsoc[col].str.lower()



ccsoc = pd.merge(tcc, tsoc, on=['state', 'county'], how='inner')



mapping = { 'cases' : 'p_cpos', 

            'deaths' : 'p_cdeaths'}



for k in mapping:

    ccsoc[mapping[k]] = (ccsoc[k] / ccsoc['e_totpop']) * 100

#ccsoc.head()





cols = ['p_cpos'] + [c for c in top_corrs]

#tdf = ccsoc[ccsoc.test_date==ccsoc.test_date.max()].sort_values(by='p_cpos')

tdf = ccsoc[~ccsoc['p_cpos'].isnull()].sort_values(by='p_cpos').reset_index(drop=True)

f2, axes2 = plt.subplots(1,11, figsize=(30,90))

#f.suptitle('Banana')

for i,col in enumerate(cols):

    axt = axes2[i].set_title(col)

    ax = axes2[i].barh(tdf['state'] + ', ' + tdf['county'], tdf[col])

    yl = axes2[i].set_ylim(0,len(tdf))

    #xg = axes2[i].grid('both')

    xt = axes[i].set_xticklabels([])

    if i>0:

        yt = axes2[i].set_yticklabels([])
hp= pd.read_csv('/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/definitive-healthcare-usa-hospital-beds.csv')

hppot = pd.DataFrame(hp.groupby(['state_name', 'county_nam']).potential.sum())

hppot['state'] = [str(i).lower() for i in hppot.index.get_level_values(0).values ]

hppot['county'] = [str(i).lower() for i in hppot.index.get_level_values(1).values ]

hpsoc = pd.merge(hppot, tsoc, on=['state', 'county'], how='inner')

hpsoc['p_hppotpop'] = (hpsoc['potential'] / hpsoc['e_totpop']) * 100





temp = hpsoc.groupby('state').p_hppotpop.sum().sort_values(ascending=False)

ax = temp.tail(20).plot(kind='barh',

                        figsize=(15,10),

                        title='States with the least "Free Hospital Beds to Population Ratio"')