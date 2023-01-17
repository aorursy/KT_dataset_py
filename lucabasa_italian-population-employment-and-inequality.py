import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import matplotlib.lines as mlines

from matplotlib.ticker import AutoMinorLocator

import matplotlib.gridspec as gridspec

import seaborn as sns

%matplotlib inline



pd.set_option('max_columns', 200)

pd.set_option('max_rows', 100)
df = pd.read_csv('../input/cusersmarildownloadspopolazionecsv/popolazione.csv', delimiter=';', 

                 encoding="ISO-8859-1", dtype={'value': np.int64}, thousands='.')



df.head()
df[['year', 'quarter']] = df['time'].str.split('-', expand=True)

df['year'] = df.year.astype(int)

df['quarter'] = df.quarter.fillna('total')

del df['time']  # we already have 2 columns about it

del df['seleziona periodo']  # redundant

del df['tipo dato']  # only one value

del df['tipo_dato_fol']  # only one value

del df['itter107']  # only a code for terriory (territorio)



df.head()
tmp = df.copy()

tmp['magnitude'] = np.log10(tmp['value']).astype(int)



tmp['max_value'] = tmp.groupby(['territorio', 'sesso', 'eta1', 

                                       'condizione_prof', 'condizione_prof_eu', 'year']).magnitude.transform('max')



tmp['difference'] = tmp['max_value'] - tmp['magnitude']  # this goes as high as 4



tmp['value_new'] = tmp['value'] * 10**tmp['difference'] 



df['value'] = tmp['value_new']



df.head()
pd.crosstab(df['condizione professionale'], df['condizione professionale europea'])
df = df[df['condizione professionale europea'] == 'totale'].copy()

del df['condizione professionale europea']

del df['condizione_prof_eu']
df[(df.quarter == 'total') & 

   (df.sesso == 'totale') & 

   (df['condizione professionale'] == 'totale')].groupby(['year','eta1','condizione_prof'], 

                                                                 as_index=False).value.sum()
tmp = df.copy()

tmp['eta1'] = tmp['eta1'].str.replace('_', '-')

tmp['condizione_prof'] = tmp['condizione_prof'].str.replace('_', '-')

prof_cond = tmp[['condizione_prof', 'condizione professionale']].drop_duplicates()



# Start manipulating

tmp = pd.pivot_table(tmp, index=['year', 'territorio'], 

                     columns=['sesso', 'eta1', 'condizione_prof', 'quarter'], values='value', fill_value=0) 

tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]



for gender in ['femmine', 'maschi', 'totale']:

    for con_prof in ['3-4', '99', '2', '1-2', '1', '3D', '3A-3B-3C']:

        for quarter in ['total', 'Q1', 'Q2', 'Q3', 'Q4']: # 

            tmp[gender+'_Y65-74_'+con_prof+'_'+quarter] = tmp[gender+'_Y15-74_'+con_prof+'_'+quarter] - tmp[gender+'_Y15-64_'+con_prof+'_'+quarter]

            tmp[gender+'_Y-GE75_'+con_prof+'_'+quarter] = tmp[gender+'_Y-GE15_'+con_prof+'_'+quarter] - tmp[gender+'_Y15-74_'+con_prof+'_'+quarter]

            tmp[gender+'_Y-GE65_'+con_prof+'_'+quarter] = tmp[gender+'_Y-GE15_'+con_prof+'_'+quarter] - tmp[gender+'_Y15-64_'+con_prof+'_'+quarter]

            try:

                tmp[gender+'_Total_'+con_prof+'_'+quarter] = tmp[gender+'_Y-GE15_'+con_prof+'_'+quarter] + tmp[gender+'_Y0-14_'+con_prof+'_'+quarter]

            except KeyError:

                tmp[gender+'_Total_'+con_prof+'_'+quarter] = tmp[gender+'_Y-GE15_'+con_prof+'_'+quarter]



tmp = tmp.reset_index()

tmp = pd.melt(tmp, id_vars=['year', 'territorio'])

tmp[['sesso', 'eta1', 'condizione_prof', 'quarter']] = tmp['variable'].str.split('_', expand=True)

del tmp['variable']



df_cleaned = pd.merge(tmp, prof_cond, on='condizione_prof', how='left')



df_cleaned.head()
df_cleaned[(df_cleaned.eta1=='Total') & 

           (df_cleaned.sesso=='totale') & 

           (df_cleaned.quarter == 'total') & 

           (df_cleaned.condizione_prof == '99')].groupby('year', as_index=False).value.sum()
by_year = df[(df.quarter == 'total') & 

         (df['condizione professionale'] == 'totale')].groupby(['year','eta1', 'sesso'], as_index=False).value.sum()



fig, ax = plt.subplots(1,2, figsize=(16, 6), facecolor='#f7f7f7')



men = by_year[(by_year.sesso == 'maschi') & (by_year.eta1 == 'Y_GE15')]

women = by_year[(by_year.sesso == 'femmine') & (by_year.eta1 == 'Y_GE15')]



ax[0].plot(men.year, men.value, label='Men', color='green')

ax[0].plot(women.year, women.value, label='Women', color='red')

ax[0].set_title('Population 15+', fontsize=14)

ax[0].legend()



men = by_year[(by_year.sesso == 'maschi') & (by_year.eta1 == 'Y0-14')]

women = by_year[(by_year.sesso == 'femmine') & (by_year.eta1 == 'Y0-14')]



ax[1].plot(men.year, men.value, label='Men', color='green')

ax[1].plot(women.year, women.value, label='Women', color='red')

ax[1].set_title('Population 0-14', fontsize=14)

ax[1].legend()



fig.suptitle('Pupulation in Italy 1996-2018', fontsize=18)



plt.show()
piv_year = pd.pivot_table(by_year, index='year', columns=['eta1', 'sesso'], values='value')

piv_year.columns = ['_'.join(col).strip() for col in piv_year.columns.values]



for gender in ['femmine', 'maschi', 'totale']:

    piv_year['Y65-74_'+gender] = piv_year['Y15-74_'+gender] - piv_year['Y15-64_'+gender]

    piv_year['Y_GE75_'+gender] = piv_year['Y_GE15_'+gender] - piv_year['Y15-74_'+gender]

    piv_year['All_'+gender] = piv_year['Y_GE15_'+gender] + piv_year['Y0-14_'+gender]



piv_year.head()
fig, ax = plt.subplots(5,1, figsize=(16, 24), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.95)



years = piv_year.index



i = 0

for sel in ['All', 'Y0-14', 'Y15-64', 'Y65-74', 'Y_GE75']:

    piv_year[sel + '_maschi'].plot(ax=ax[i], label='Men', color='green')

    piv_year[sel + '_femmine'].plot(ax=ax[i], label='Women', color='red')

    ax[i].legend()

    ax[i].set_xticks(years)

    i += 1



ax[0].set_title('Total population', fontsize=14)

ax[1].set_title('Population 0-14', fontsize=14)

ax[2].set_title('Population 15-64', fontsize=14)

ax[3].set_title('Population 65-74', fontsize=14)

ax[4].set_title('Population 75+', fontsize=14)





fig.suptitle('Pupulation in Italy 1996-2018', fontsize=18)



plt.show()
def newline(ax, p1, p2, color='black'):

    l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color=color)

    ax.add_line(l)

    return ax



def plot_pop_change(data, pop, title, range_pop='Full', prof_cond='totale'):

    fig, ax = plt.subplots(1,1,figsize=(14,14), facecolor='#f7f7f7')

    fig.subplots_adjust(top=0.95)



    by_year = data[(data.quarter == 'total') & (data.sesso == 'totale') & (data.eta1 == pop) &

             (data['condizione professionale'] == prof_cond)].groupby(['year','eta1', 'territorio'], as_index=False).value.sum()



    y_1996 = by_year[by_year.year == 1996].reset_index(drop=True)

    y_2018 = by_year[by_year.year == 2018].reset_index(drop=True)



    ax.scatter(y=y_1996['territorio'], x=y_1996['value'], s=80, color='#0e668b', alpha=0.5)

    ax.scatter(y=y_2018['territorio'], x=y_2018['value'], s=160, color='#ff0000', alpha=0.6)

    

    fig.suptitle(title, fontsize=18)



    for i, p1, p2 in zip(y_1996['territorio'], y_1996['value'], y_2018['value']):

        ax = newline(ax, [p1, i], [p2, i])

    

    if range_pop == 'Full':

        ax.set(xlim=(0,10500000), xlabel='Population')

        ax.vlines(x=2000000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

        ax.vlines(x=6000000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

        ax.set_xticks([0, 2000000, 4000000, 6000000, 8000000, 10000000])

        ax.set_xticklabels(['0', '2M', '4M', '6M', '8M', '10M'])

    elif range_pop == 'Reduced':

        ax.set(xlim=(0,1200000), xlabel='Population')

        ax.vlines(x=300000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

        ax.vlines(x=700000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

    elif range_pop == 'Employed':

        ax.set(xlim=(0,5000000), xlabel='Population')

        ax.vlines(x=1000000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

        ax.vlines(x=3000000, ymin='Abruzzo', ymax='Veneto', color='black', alpha=1, linewidth=1, linestyles='dotted')

    

    plt.show()



plot_pop_change(df_cleaned, 'Total', 'Variation in the Italian population by region, 1996 vs 2018')
plot_pop_change(df_cleaned, 'Y15-64', 'Variation in the 15-64 population by region, 1996 vs 2018')
plot_pop_change(df_cleaned, 'Y65-74', 'Variation in the 65-74 population by region, 1996 vs 2018', range_pop='Reduced')
plot_pop_change(df_cleaned, 'Y-GE75', 'Variation in the 75+ population by region, 1996 vs 2018', range_pop='Reduced')
cond_prof = {'forze lavoro': 'workforce',  # the sum of employed and unemployed

             'occupati': 'employed', 

             'disoccupati': 'unemployed', 

             'inattivi': 'inactive',  # whoever is not workforce, people that can't work or are not looking for a job

             'totale': 'total',  # the sum of workforce and inactive

             'non cercano e non disponibili a lavorare': 'not_looking',  # part of the inactives

             "zona grigia dell'inattività": 'inactive_greyzone'}  # honestly don't know, but it the other half of the inactive



df_cleaned['prof_cond'] = df_cleaned['condizione professionale'].map(cond_prof)
def group_year_profcond(data, gender, age='Y15-64', regions=False):

    

    if regions:

        by_year = data[(data.quarter == 'total') & 

           (data.sesso == gender) & 

           (data.eta1 == age)].groupby(['year', 'territorio', 'prof_cond']).value.sum().unstack(2).reset_index()

    else:

        by_year = data[(data.quarter == 'total') & 

           (data.sesso == gender) & 

           (data.eta1 == age)].groupby(['year', 'prof_cond']).value.sum().unstack()

    

    # some entries are inconsistent when we consider the gender split, we fix them here

    by_year['workforce'] = by_year['employed'] + by_year['unemployed']

    try:

        by_year['total'] = by_year['workforce'] + by_year['inactive']

        by_year['inactive_perc'] = by_year['inactive'] / by_year['total']

        by_year['unemployed_overInac'] = by_year['unemployed'] / by_year['inactive']

    except KeyError:

        pass

    

    

    # calculation of rates

    by_year['employed_overWF'] = by_year['employed'] / by_year['workforce']

    by_year['employed_perc'] = by_year['employed'] / by_year['total']

    by_year['unemployed_overWF'] = by_year['unemployed'] / by_year['workforce']

    by_year['unemployed_perc'] = by_year['unemployed'] / by_year['total']

    

    try:

        del by_year['inactive_greyzone']

        del by_year['not_looking']

        #del by_year['total']

    except KeyError:

        pass

    

    return by_year



total_year = group_year_profcond(df_cleaned, 'totale')

males_year = group_year_profcond(df_cleaned, 'maschi')

females_year = group_year_profcond(df_cleaned, 'femmine')



total_year.head()
def multiple_proportions(total_year, males_year, females_year, title, inactives=True):

    if inactives:

        fig, ax = plt.subplots(3,2, figsize=(15, 18), facecolor='#f7f7f7')

    else:

        fig, ax = plt.subplots(2,2, figsize=(15, 12), facecolor='#f7f7f7')

    fig.subplots_adjust(top=0.95)



    def plot_proportions(data, label, color, inactives=True):



        data['employed_perc'].plot(ax=ax[0][0], label=label, color=color)

        data['unemployed_perc'].plot(ax=ax[0][1], label=label, color=color)

        data['employed_overWF'].plot(ax=ax[1][0], label=label, color=color)

        data['unemployed_overWF'].plot(ax=ax[1][1], label=label, color=color)

        if inactives:

            data['inactive_perc'].plot(ax=ax[2][0], label=label, color=color)

            data['unemployed_overInac'].plot(ax=ax[2][1], label=label, color=color)



        return fig, ax



    fig, ax = plot_proportions(total_year, 'Total', 'black', inactives)

    fig, ax = plot_proportions(males_year, 'Males', 'green', inactives)

    fig, ax = plot_proportions(females_year, 'Females', 'red', inactives)



    ax[0][0].set_ylim((0, 1))

    ax[0][1].set_ylim((0, 1))

    ax[1][0].set_ylim((0, 1))

    ax[1][1].set_ylim((0, 1))

    ax[0][0].legend()

    ax[0][1].legend()

    ax[1][0].legend()

    ax[1][1].legend() 

    ax[0][0].set_title('Employed over Population', fontsize=14)

    ax[0][1].set_title('Unemployed over Population', fontsize=14)

    ax[1][0].set_title('Employed over Workforce', fontsize=14)

    ax[1][1].set_title('Unemployed over Workforce', fontsize=14)

    

    if inactives:

        ax[2][0].set_ylim((0, 1))

        ax[2][1].set_ylim((0, 1))

        ax[2][0].legend()

        ax[2][1].legend()

        ax[2][0].set_title('Inactives over Population', fontsize=14)

        ax[2][1].set_title('Unemployed over Inactives', fontsize=14)

        

    fig.suptitle(title, fontsize=18)



    plt.show()

    

    

multiple_proportions(total_year, males_year, females_year, 'Employed, Unemployed, and Inactives (15-64)')
fig, ax = plt.subplots(3,2, figsize=(15, 15), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.92)



ax[0][0].pie(total_year[total_year.index==1996][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

ax[0][1].pie(total_year[total_year.index==2018][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

ax[1][0].pie(males_year[males_year.index==1996][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

ax[1][1].pie(males_year[males_year.index==2018][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

ax[2][0].pie(females_year[females_year.index==1996][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

ax[2][1].pie(females_year[females_year.index==2018][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')



ax[0][0].set_title('Total Population - 1996', fontsize=14)

ax[0][1].set_title('Total Population - 2018', fontsize=14)

ax[1][0].set_title('Males - 1996', fontsize=14)

ax[1][1].set_title('Males - 2018', fontsize=14)

ax[2][0].set_title('Females - 1996', fontsize=14)

ax[2][1].set_title('Females - 2018', fontsize=14)



fig.suptitle('Employed, Unemployed, and Inactive - 15-64 - 1996 vs 2018', fontsize=18)



plt.show()
fig, ax = plt.subplots(3,1, figsize=(18, 16), facecolor='#f7f7f7')



total_year['employed_perc'].plot(ax=ax[0], color='k')

total_year['unemployed_perc'].plot(ax=ax[1], color='k')

total_year['inactive_perc'].plot(ax=ax[2], color='k')



ax[0].set_title('Employement', fontsize=14)

ax[1].set_title('Unemployement', fontsize=14)

ax[2].set_title('Inactive', fontsize=14)



for axes in ax:

    axes.axvspan(1996, 2001, facecolor='r', alpha=0.2)  # 3 goverments + 1 technical goverment

    axes.axvline(x=2001, linestyle='--')

    axes.axvspan(2001, 2006, facecolor='b', alpha=0.2)  # 2 goverments 

    axes.axvline(x=2006, linestyle='--')

    axes.axvspan(2006, 2008, facecolor='r', alpha=0.2)  # 1 goverment

    axes.axvline(x=2008, linestyle='--')

    axes.axvspan(2008, 2011, facecolor='b', alpha=0.2)  # 1 goverment

    axes.axvline(x=2011, linestyle='--')

    axes.axvline(x=2012, linestyle='--')

    axes.axvspan(2012, 2016, facecolor='r', alpha=0.2)  # 2 governments

    axes.axvline(x=2016, linestyle='--')

    axes.axvline(x=2018, linestyle='--')

    axes.set_yticklabels(['{:,.0%}'.format(x) for x in axes.get_yticks()])
macro_regions = {'Abruzzo': 'South', 

                 'Basilicata': 'South', 

                 'Calabria': 'South', 

                 'Campania': 'South', 

                 'Emilia-Romagna': 'North-East', 

                 'Friuli-Venezia Giulia': 'North-East', 

                 'Lazio': 'Center', 

                 'Liguria': 'North-West', 

                 'Lombardia': 'North-West', 

                 'Marche': 'Center', 

                 'Molise': 'South', 

                 'Piemonte': 'North-West', 

                 'Provincia Autonoma Bolzano / Bozen': 'North-East', 

                 'Provincia Autonoma Trento': 'North-East', 

                 'Puglia': 'South', 

                 'Sardegna': 'Islands', 

                 'Sicilia': 'Islands', 

                 'Toscana': 'Center', 

                 'Trentino Alto Adige / Südtirol': 'North-East', 

                 'Umbria': 'Center', 

                 "Valle d'Aosta / Vallée d'Aoste": 'North-West', 

                 'Veneto': 'North-East'}
total_regions = group_year_profcond(df_cleaned, 'totale', regions=True)



total_regions['macro_region'] = total_regions.territorio.map(macro_regions)



macros = total_regions.groupby(['year', 

                             'macro_region'], as_index=False)[['employed', 

                                                               'unemployed', 

                                                               'inactive', 

                                                               'total']].sum().set_index('year')



macros['employed_perc'] = macros['employed'] / macros['total']

macros['unemployed_perc'] = macros['unemployed'] / macros['total']

macros['inactive_perc'] = macros['inactive'] / macros['total']





fig, ax = plt.subplots(3,1, figsize=(18, 16), facecolor='#f7f7f7')



for region in macros.macro_region.unique():

    macros[macros.macro_region == region]['employed_perc'].plot(ax=ax[0], label=region, alpha=0.7)

    macros[macros.macro_region == region]['unemployed_perc'].plot(ax=ax[1], label=region, alpha=0.7)

    macros[macros.macro_region == region]['inactive_perc'].plot(ax=ax[2], label=region, alpha=0.7)

    

total_year['employed_perc'].plot(ax=ax[0], color='k', label='Total')

total_year['unemployed_perc'].plot(ax=ax[1], color='k', label='Total')

total_year['inactive_perc'].plot(ax=ax[2], color='k', label='Total')



ax[0].set_title('Employement', fontsize=14)

ax[1].set_title('Unemployement', fontsize=14)

ax[2].set_title('Inactive', fontsize=14)



for axes in ax:

    axes.legend()

    axes.set_yticklabels(['{:,.0%}'.format(x) for x in axes.get_yticks()])
fig, ax = plt.subplots(2,1, figsize=(18, 10), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.92)



for region in ['South', 'Islands']:

    macros[macros.macro_region == region]['employed_perc'].plot(ax=ax[0], label=region, alpha=0.7)

    macros[macros.macro_region == region]['unemployed_perc'].plot(ax=ax[1], label=region, alpha=0.7)

    

total_year['employed_perc'].plot(ax=ax[0], color='k', label='Total')

total_year['unemployed_perc'].plot(ax=ax[1], color='k', label='Total')



ax[0].set_title('Employement', fontsize=14)

ax[1].set_title('Unemployement', fontsize=14)



for axes in ax:

    axes.axvspan(1996, 2001, facecolor='r', alpha=0.2)  # 3 goverments + 1 technical goverment

    axes.axvline(x=2001, linestyle='--')

    axes.axvspan(2001, 2006, facecolor='b', alpha=0.2)  # 2 goverments 

    axes.axvline(x=2006, linestyle='--')

    axes.axvspan(2006, 2008, facecolor='r', alpha=0.2)  # 1 goverment

    axes.axvline(x=2008, linestyle='--')

    axes.axvspan(2008, 2011, facecolor='b', alpha=0.2)  # 1 goverment

    axes.axvline(x=2011, linestyle='--')

    axes.axvline(x=2012, linestyle='--')

    axes.axvspan(2012, 2016, facecolor='r', alpha=0.2)  # 2 governments

    axes.axvline(x=2016, linestyle='--')

    axes.axvline(x=2018, linestyle='--')

    axes.set_yticklabels(['{:,.0%}'.format(x) for x in axes.get_yticks()])

    axes.legend()

    

fig.suptitle('Cyclic decline in Employement for southern regions and governement changes', fontsize=18)

plt.show()
plot_pop_change(df_cleaned, 'Y15-64', 'Variation in the number of Employed by region, 1996 vs 2018', 

                prof_cond='occupati', range_pop='Employed')
def plot_pop_sidebyside(data, title, prof_cond='occupati'):

    fig, ax = plt.subplots(1,2,figsize=(16,6), facecolor='#f7f7f7')

    fig.subplots_adjust(top=0.87)



    by_year = data.groupby(['year', 'macro_region'], as_index=False).sum()



    y_1996 = by_year[by_year.year == 1996].reset_index(drop=True)

    y_2018 = by_year[by_year.year == 2018].reset_index(drop=True)



    ax[0].scatter(y=y_1996['macro_region'], x=y_1996['total'], s=80, color='#0e668b', alpha=0.5)

    ax[0].scatter(y=y_2018['macro_region'], x=y_2018['total'], s=160, color='#ff0000', alpha=0.6)

    ax[1].scatter(y=y_1996['macro_region'], x=y_1996[prof_cond], s=80, color='#0e668b', alpha=0.5)

    ax[1].scatter(y=y_2018['macro_region'], x=y_2018[prof_cond], s=160, color='#ff0000', alpha=0.6)



    fig.suptitle(title, fontsize=18)



    for i, p1, p2 in zip(y_1996['macro_region'], y_1996['total'], y_2018['total']):

        ax[0] = newline(ax[0], [p1, i], [p2, i])

    for i, p1, p2 in zip(y_1996['macro_region'], y_1996[prof_cond], y_2018[prof_cond]):

        ax[1] = newline(ax[1], [p1, i], [p2, i])

        

    ax[0].set(xlim=(0,10500000), xlabel='Population')

    ax[0].set_title('Population', fontsize=14)

    ax[1].set(xlim=(0,10500000), xlabel=prof_cond.capitalize())

    ax[1].set_title(f'{prof_cond.capitalize()}', fontsize=14)

    for axes in ax:

        axes.set_xticks([0, 2000000, 4000000, 6000000, 8000000, 10000000])

        axes.set_xticklabels(['0', '2M', '4M', '6M', '8M', '10M'])



    plt.show()

    

plot_pop_sidebyside(total_regions, 'Variation by Macro Regions (15-64), 1996 vs 2018', prof_cond='employed')
by_year = total_regions.groupby(['year', 'macro_region'], as_index=False).sum()



by_year['employed_perc'] = by_year['employed'] / by_year['total']

by_year['unemployed_perc'] = by_year['unemployed'] / by_year['total']

by_year['inactive_perc'] = by_year['inactive'] / by_year['total']



fig, ax = plt.subplots(5,2, figsize=(15, 25), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.95)



for axes, region in zip(ax[:,0], ['South', 'North-West', 'North-East', 'Islands', 'Center']):

    axes.pie(by_year[(by_year.year==1996) & 

                     (by_year.macro_region==region)][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

    axes.set_title(f'{region} - 1996', fontsize=14)



for axes, region in zip(ax[:,1], ['South', 'North-West', 'North-East', 'Islands', 'Center']):

    axes.pie(by_year[(by_year.year==2018) & 

                     (by_year.macro_region==region)][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

        labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

    axes.set_title(f'{region} - 2018', fontsize=14)



fig.suptitle('Employed, Unemployed, and Inactive - 15-64 - 1996 vs 2018', fontsize=18)



plt.show()
def region_overview(data, region):

    total_year = group_year_profcond(data[data.territorio==region], 'totale', regions=True)

    males_year = group_year_profcond(data[data.territorio==region], 'maschi', regions=True)

    females_year = group_year_profcond(data[data.territorio==region], 'femmine', regions=True)

    

    total_year = total_year[total_year.year < 2019]

    males_year = males_year[males_year.year < 2019]

    females_year = females_year[females_year.year < 2019]



    fig = plt.figure(figsize=(18, 10), facecolor='#f7f7f7') 

    fig.suptitle(region, fontsize=18)



    spec = fig.add_gridspec(ncols=3, nrows=2, height_ratios=[5, 3])



    ax0 = fig.add_subplot(spec[0,:])

    ax1 = fig.add_subplot(spec[1,0])

    ax2 = fig.add_subplot(spec[1,1])

    ax3 = fig.add_subplot(spec[1,2])



    ax0.set_title('Population trend by labour status (15-64)', fontsize=14)

    ax1.set_title('Employed, Unemployed, Inactive (2018)', fontsize=12)

    ax2.set_title('Males (15-64)', fontsize=12)

    ax3.set_title('Females (15-64)', fontsize=12)

    

    total_year.set_index('year').employed_perc.plot(ax=ax0, label='Employed')

    total_year.set_index('year').unemployed_perc.plot(ax=ax0, label='Unemployed')

    total_year.set_index('year').inactive_perc.plot(ax=ax0, label='Inactive')

    

    ax1.pie(total_year[total_year.year==2018][['employed_perc', 'unemployed_perc', 'inactive_perc']].values[0], 

            labels=['Employed', 'Unemployed', 'Inactive'], autopct='%.0f%%')

    

    males_year.set_index('year').employed.plot(ax=ax2, label='Empl.')

    males_year.set_index('year').unemployed.plot(ax=ax2, label='Unempl.')

    males_year.set_index('year').inactive.plot(ax=ax2, label='Inact.')

    females_year.set_index('year').employed.plot(ax=ax3, label='Empl.')

    females_year.set_index('year').unemployed.plot(ax=ax3, label='Unempl.')

    females_year.set_index('year').inactive.plot(ax=ax3, label='Inact.')

    

    for axes in [ax0, ax2, ax3]:

        axes.set_xticks([1997, 2002, 2007, 2012, 2017])

        

    ax0.legend()

    ax0.set_ylim((0,1))

    ax0.set_ylabel('Percentage')

    ax0.set_yticklabels(['{:,.0%}'.format(x) for x in ax0.get_yticks()])



    plt.show()
region_overview(df_cleaned, "Lombardia")
region_overview(df_cleaned, "Emilia-Romagna")
region_overview(df_cleaned, "Veneto")
region_overview(df_cleaned, "Friuli-Venezia Giulia")
region_overview(df_cleaned, "Lazio")
region_overview(df_cleaned, "Abruzzo")
region_overview(df_cleaned, "Campania")
region_overview(df_cleaned, "Sardegna")
region_overview(df_cleaned, "Sicilia")