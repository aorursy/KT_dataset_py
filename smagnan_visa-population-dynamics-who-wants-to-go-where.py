import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl # an other interface to matplotlib

from matplotlib.gridspec import GridSpec # more advanced subplots layout



# Loading data

df_2017 = pd.read_csv('../input/schengen-visa-stats/2017-data-for-consulates.csv')

df_2018 = pd.read_csv('../input/schengen-visa-stats/2018-data-for-consulates.csv')

df_2017['year'] = 2017

df_2018['year'] = 2018

# adding extra information: the year

df = pd.concat([df_2017, df_2018])

print(df.columns)




# Data cleaning

# dropping data we won't use

df.drop(

    [

        'Airport transit visas (ATVs) applied for ',

        ' ATVs issued (including multiple)',

        'Multiple ATVs issued',

        'ATVs not issued ',

        'Not issued rate for ATVs',

        'Not issued rate for ATVs and uniform visas ',

        'Not issued rate for uniform visas',

        'Total ATVs and uniform visas applied for',

        'Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) ',

        'Total ATVs and uniform visas not issued',

        'Not issued rate for uniform visas',

        'Share of MEVs on total number of uniform visas issued',



    ],

    axis=1,

    inplace=True

)

# renaming columns

df.rename(

    columns={

        'Schengen State': 'schengen_state',

        'Country where consulate is located': 'country',

        'Consulate': 'city',

        'Uniform visas applied for': 'applications',

        'Total  uniform visas issued (including MEV) \n': 'tuv_issued',

        'Multiple entry uniform visas (MEVs) issued': 'mev_issued',

        'Total LTVs issued': 'ltv_issued',

        'Uniform visas not issued': 'tuv_not_issued',

    },

    inplace=True

)



# fixing data format: to string to int/float when possible (for the fields where it makes sense)

def us_format_to_int(value):

    try:

        return int(value.replace(',', '')) if isinstance(value, str) else value

    except ValueError:

        return np.NaN





for col in ['applications', 'mev_issued', 'tuv_issued', 'ltv_issued', 'tuv_not_issued']:

    df[col] = df[col].apply(lambda r: us_format_to_int(r))



print(df.head())
# ---- who goes where ? / who wants to go where?

country2schengen_state_stats = df.groupby(['schengen_state', 'country']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['schengen_state', 'country'])

# for the two following, we sort by nb of applications since this will be the axis order for our next graph>

# other possible choice: sort by state/country name

schengen_state_stats = df.groupby(['schengen_state']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['applications'], ascending=False).reset_index()

country_stats = df.groupby(['country']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['applications'], ascending=False).reset_index()

print(country2schengen_state_stats)

print(schengen_state_stats.head())

print(country_stats.head())
# we now add an index to our sorted data: this index will be used to ensure everying is displayed in the correct order 

# over the x and y axis through the various graphs (ticks values)

schengen_state_order2index = {schengen_state_stats.schengen_state[i]: i for i in range(len(schengen_state_stats.schengen_state))}

country2index = {country_stats.country[i]: i for i in range(len(country_stats.country))}



country2schengen_state_stats['ss_index'] = country2schengen_state_stats.schengen_state.apply(lambda r: schengen_state_order2index[r])

country2schengen_state_stats['cc_index'] = country2schengen_state_stats.country.apply(lambda r: country2index[r])

schengen_state_stats['ss_index'] = schengen_state_stats.schengen_state.apply(lambda r: schengen_state_order2index[r])

country_stats['cc_index'] = country_stats.country.apply(lambda r: country2index[r])
def plot_visa_counts(country2schengen_state_stats, schengen_state_stats, country_stats, keys=None, extra_margin=9, figsize=(16, 16)):

    fig = pl.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')



    gs = GridSpec(4, 4)



    ax_joint = fig.add_subplot(gs[1:4, 0:3])



    ax_joint.set_xticks(range(len(schengen_state_stats.schengen_state)))

    ax_joint.set_xticklabels(schengen_state_stats.schengen_state, rotation=70)

    ax_joint.set_yticks(range(len(country_stats.country)))

    ax_joint.set_yticklabels(country_stats.country)

    ax_joint.set_xlim(-1, len(schengen_state_stats.schengen_state) + 1)

    ax_joint.set_ylim(-1 - extra_margin, len(country_stats.country) + 1)



    ax_marg_x = fig.add_subplot(gs[0, 0:3])

    ax_marg_x.set_xticks(range(len(schengen_state_stats.schengen_state)))

    ax_marg_x.set_xticklabels(schengen_state_stats.schengen_state)

    ax_marg_x.set_xlim(-1, len(schengen_state_stats.schengen_state) + 1)

    ax_marg_y = fig.add_subplot(gs[1:4, 3])



    ax_marg_y.set_yticks(range(len(country_stats.country)))

    ax_marg_y.set_yticklabels(country_stats.country)

    ax_marg_y.set_ylim(-1 - extra_margin, len(country_stats.country) + 1)



    keys = ['applications', 'mev_issued'] if keys is None else keys

    width = 0.7 / len(keys)

    for i, key in enumerate(['applications', 'mev_issued']):

        # for the size of the dots (s), we use square root: from the doc -> s :

        # scalar or array_like, shape (n, ), optional

        # The marker size in points**2. Default is rcParams['lines.markersize'] ** 2.

        ax_joint.scatter(country2schengen_state_stats['ss_index'], country2schengen_state_stats['cc_index'], s=1 * np.sqrt(country2schengen_state_stats[key]), alpha=0.3)

        ax_marg_x.bar(schengen_state_stats['ss_index'] + (i-0.5)*width, schengen_state_stats[key], width=width, alpha=0.5, label=key)

        ax_marg_y.barh(country_stats['cc_index'] + (i-0.5)*width, country_stats[key], height=width, alpha=0.5)



    # Turn off tick labels on marginals

    pl.setp(ax_marg_x.get_xticklabels(), visible=False)

    pl.setp(ax_marg_y.get_yticklabels(), visible=False)



    # Set labels on joint

    ax_joint.set_xlabel('Schengen States')

    ax_joint.set_ylabel('Applying Countries')



    # Set labels on marginals

    ax_marg_y.set_xlabel('VISA / Schengen state')

    ax_marg_x.set_ylabel('VISA / Applying country')

    

    ax_marg_x.legend(loc='upper right')

    pl.suptitle('Schengen VISA applications and issued (mev)', size=16)

    pl.tight_layout()

    pl.show()





plot_visa_counts(country2schengen_state_stats, schengen_state_stats, country_stats, keys=None, figsize=(18, 26))

schengen_state_stats_small = schengen_state_stats.head(16)

country_stats_small = country_stats.head(16)

country2schengen_state_stats_small = country2schengen_state_stats[country2schengen_state_stats.apply(lambda r: r.cc_index in set(country_stats_small.cc_index) and r.ss_index in set(schengen_state_stats_small.ss_index), axis=1)]

plot_visa_counts(country2schengen_state_stats_small, schengen_state_stats_small, country_stats_small, keys=None, extra_margin=1)