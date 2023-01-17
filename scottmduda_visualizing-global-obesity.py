import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons

from matplotlib.widgets import RadioButtons



import plotly_express as px

import plotly.graph_objs as go

from plotly.offline import iplot



import seaborn as sns



plt.rcParams["figure.figsize"] = (12,6)

plt.style.use('ggplot')
obesity_df = pd.read_csv('../input/obesity-among-adults-by-country-19752016/obesity-cleaned.csv', index_col=0)

pop_df = pd.read_csv('../input/world-population-19602018/population_total_long.csv')

region_df = pd.read_csv('../input/country-mapping-iso-continent-region/continents2.csv')
obesity_df.head()
obesity_df.columns = ['country', 'year', 'obesity', 'sex']



num_sex_cats = len(obesity_df.sex.unique())

min_year = min(obesity_df.year)

max_year = max(obesity_df.year)

total_years = max_year - min_year + 1

num_countries = int(obesity_df.shape[0] / num_sex_cats / total_years)



print('There are {} rows of data and {} columns.'.format(obesity_df.shape[0], obesity_df.shape[1]))

print('-----------------')

sex_vals = obesity_df.sex.value_counts()

print('Obesity data are reported for {} sex categories: {}, {}, and {}.'.format(num_sex_cats, sex_vals.index[0], sex_vals.index[1], sex_vals.index[2]))

print('-----------------')

print('Obesity data are reported from {} to {}, covering {} years.'.format(min_year, max_year, total_years))

print('-----------------')

print('Obesity data from {} countries are presented.'.format(num_countries))
obesity_df.isnull().any()
obesity_df.isna().any()
df_nodata = obesity_df.loc[obesity_df.obesity == 'No data']

print('There are {} rows with missing obesity data. Data are missing for {} countries:'.format(len(df_nodata), len(df_nodata.country.value_counts())))

for country in df_nodata.country.value_counts().index:

    print('\t' + country)

print('-----------------')

print('Each country is missing {} rows of data.'.format(df_nodata.country.value_counts().values[0]))

print('Missing data is cumulative for each year (i.e. "No data" is reported for all three sexes.)')

print('This means that each of these countries is missing {} years worth of data...'.format(int(df_nodata.country.value_counts().values[0] /  len(obesity_df.sex.unique()))))



num_countries -= len(df_nodata.country.value_counts())

print("Looks like we can remove these countries, dropping the total number of countries represented to {}.".format(num_countries))
obesity_df.drop(obesity_df[obesity_df.country.isin(['South Sudan', 'Sudan', 'San Marino', 'Monaco'])].index, inplace=True)
obesity_df['obesity_prev'] = obesity_df.obesity.apply(lambda x: float(x.split(' ')[0]))

obesity_df['obesity_cri_lower'] = obesity_df.obesity.apply(lambda x: float((x.split(' ')[1]).split('-')[0][1:]))

obesity_df['obesity_cri_upper'] = obesity_df.obesity.apply(lambda x: float((x.split(' ')[1]).split('-')[1][:-1]))
obesity_df['obesity_cri_width'] = obesity_df['obesity_cri_upper'] - obesity_df['obesity_cri_lower']
country_map = {

    'Bahamas, The': 'Bahamas',

    'Bolivia': 'Bolivia (Plurinational State of)',

    'Congo, Rep.': 'Congo',

    'Czech Republic':'Czechia',

    "Cote d'Ivoire":"Côte d'Ivoire",

    'Korea, Dem. People’s Rep.':"Democratic People's Republic of Korea", 

    'Congo, Dem. Rep.':'Democratic Republic of the Congo',

    'Egypt, Arab Rep.':'Egypt',

    'Gambia, The':'Gambia',

    'Iran, Islamic Rep.':'Iran (Islamic Republic of)',

    'Kyrgyz Republic':'Kyrgyzstan',

    'Lao PDR':"Lao People's Democratic Republic",

    'Micronesia, Fed. Sts.':'Micronesia (Federated States of)',

    'Korea, Rep.':'Republic of Korea',

    'Moldova':'Republic of Moldova',

    'North Macedonia':'Republic of North Macedonia',

    'St. Kitts and Nevis':'Saint Kitts and Nevis',

    'St. Lucia':'Saint Lucia',

    'St. Vincent and the Grenadines':'Saint Vincent and the Grenadines',

    'Slovak Republic':'Slovakia',

    'Sudan':'Sudan (former)',

    'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',

    'Tanzania':'United Republic of Tanzania',

    'United States':'United States of America',

    'Venezuela, RB':'Venezuela (Bolivarian Republic of)',

    'Vietnam':'Viet Nam',

    'Yemen, Rep.':'Yemen'

}



# population data was not available for Cook Islands and Niue 
pop_df.replace({'Country Name': country_map}, inplace=True)

obesity_df = obesity_df.merge(pop_df, how='left', left_on=['country', 'year'], right_on=['Country Name', 'Year']).drop(['Country Name', 'Year'], axis=1)

obesity_df.rename(columns={'Count': 'population'}, inplace=True)
obesity_df['obesity_prev_count'] = obesity_df['obesity_prev'] / 100 * obesity_df['population']

obesity_df['obesity_cri_min_count'] = obesity_df['obesity_cri_lower'] / 100 * obesity_df['population']

obesity_df['obesity_cri_max_count'] = obesity_df['obesity_cri_upper'] / 100 * obesity_df['population']
country_2_map = {

    'Bolivia (Plurinational State of)': 'Bolivia',

    'Bosnia and Herzegovina': 'Bosnia And Herzegovina',

    'Czechia': 'Czech Republic',

    "Côte d'Ivoire": "Côte D'Ivoire",

    "Democratic People's Republic of Korea": 'Korea, Republic of',

    'Democratic Republic of the Congo': 'Congo (Democratic Republic Of The)',

    'Guinea-Bissau': 'Guinea Bissau',

    'Iran (Islamic Republic of)': 'Iran',

    "Lao People's Democratic Republic": 'Laos',

    'Republic of Korea': 'South Korea',

    'Republic of Moldova': 'Moldova',

    'Republic of North Macedonia': 'Macedonia',

    'Russian Federation': 'Russia',

    'Sudan (former)': 'Sudan',

    'Syrian Arab Republic': 'Syria',

    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',

    'United Republic of Tanzania': 'Tanzania',

    'United States of America': 'United States',

    'Venezuela (Bolivarian Republic of)': 'Venezuela',

    'Viet Nam': 'Vietnam'

}
obesity_df['country_2'] = obesity_df['country'].values.copy()



obesity_df.replace({'country_2': country_2_map}, inplace=True)



region_df = region_df[['name', 'region', 'sub-region']]

obesity_df = obesity_df.merge(region_df, how='left', left_on=['country_2'], right_on=['name']).drop('name', axis=1)
obesity_df_male = obesity_df.loc[obesity_df.sex == 'Male', :].reset_index(drop=True)

obesity_df_male_pivot = obesity_df_male[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')



obesity_df_female = obesity_df.loc[obesity_df.sex == 'Female', :].reset_index(drop=True)

obesity_df_female_pivot = obesity_df_female[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')



obesity_df_both = obesity_df.loc[obesity_df.sex == 'Both sexes', :].reset_index(drop=True)

obesity_df_both_pivot = obesity_df_both[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')
mf_diff = obesity_df_female_pivot - obesity_df_male_pivot
high_obesity_countries = obesity_df_both.groupby('country').mean().sort_values(by='obesity_prev', ascending=False)['obesity_prev'][:20]

countries_high = high_obesity_countries.index.tolist()

values_high = high_obesity_countries.values.tolist()
ax = sns.barplot(x=countries_high, y=values_high, color='red')

ax.set(title='High-Obesity Countries (Top 20 Countries by Mean Obesity Prevalence, 1975-2016)', ylabel='Average Obesity Prevalence (%), 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
low_obesity_countries = obesity_df_both.groupby('country').mean().sort_values(by='obesity_prev', ascending=True)['obesity_prev'][:20].sort_values(ascending=False)

countries_low = low_obesity_countries.index.tolist()

values_low = low_obesity_countries.values.tolist()
ax = sns.barplot(x=countries_low, y=values_low, color='blue')

ax.set(title='Top 20 Countries With the Lowest Average Mean Obesity Prevalence, 1975-2016)', ylabel='Average Obesity Prevalence (%), 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
obesity_region = obesity_df_both.groupby(['region', 'year']).mean().reset_index(drop=False)

obesity_sub = obesity_df_both.groupby(['sub-region', 'year']).mean().reset_index(drop=False)



obesity_region_pivot = obesity_region[['region', 'year', 'obesity_prev']].pivot(index='region', columns='year', values='obesity_prev')

obesity_sub_pivot = obesity_sub[['sub-region', 'year', 'obesity_prev']].pivot(index='sub-region', columns='year', values='obesity_prev')



obesity_region_mean = obesity_df.groupby('region').mean().reset_index(drop=False)[['region', 'obesity_prev']].sort_values(by='obesity_prev', ascending=False)

obesity_subregion_mean = obesity_df.groupby('sub-region').mean().reset_index(drop=False)[['sub-region', 'obesity_prev']].sort_values(by='obesity_prev', ascending=False)
data = [go.Scatter(x=obesity_region_pivot.columns,

                   y=obesity_region_pivot.loc[region],

                   name=region) for region in obesity_region_pivot.index]



layout = go.Layout(

    title='Obesity Prevalence by Region (Both Sexes)',

    yaxis=dict(title='Obesity Prevalence (%)'),

    xaxis=dict(title='Year')

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
ax = sns.barplot(x=obesity_region_mean['region'], y=obesity_region_mean.obesity_prev, color='darkgreen')

ax.set(title='Mean Obesity Prevalence by Region, 1975-2016', ylabel='Mean Obesity Prevalence (%), 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
def plot_by_region(region_type, region, grid_rows, grid_cols, y_lim, figsize):

    

    if region_type == 'region':

        region_df = obesity_df.loc[obesity_df.region == region]

    elif region_type == 'sub-region':

        region_df = obesity_df.loc[obesity_df['sub-region'] == region]

    else:

        return "Please enter a valid region type ('region' or 'sub-region')."

        

    # obesity prevalence

    region_countries = region_df.country.unique().tolist()



    region_df_female = region_df.loc[region_df.sex == 'Female', ['country', 'year', 'obesity_prev']]

    region_df_male = region_df.loc[region_df.sex == 'Male', ['country', 'year', 'obesity_prev']]



    region_df_female_pivot = region_df_female.pivot(index='country', columns='year', values='obesity_prev')

    region_df_male_pivot = region_df_male.pivot(index='country', columns='year', values='obesity_prev')



    # credible intervals

    region_df_cri_min_female = region_df.loc[region_df.sex == 'Female', ['country', 'year', 'obesity_cri_lower']]

    region_df_cri_min_male = region_df.loc[region_df.sex == 'Male', ['country', 'year', 'obesity_cri_lower']]



    region_df_cri_min_female_pivot = region_df_cri_min_female.pivot(index='country', columns='year', values='obesity_cri_lower')

    region_df_cri_min_male_pivot = region_df_cri_min_male.pivot(index='country', columns='year', values='obesity_cri_lower')



    region_df_cri_max_female = region_df.loc[region_df.sex == 'Female', ['country', 'year', 'obesity_cri_upper']]

    region_df_cri_max_male = region_df.loc[region_df.sex == 'Male', ['country', 'year', 'obesity_cri_upper']]



    region_df_cri_max_female_pivot = region_df_cri_max_female.pivot(index='country', columns='year', values='obesity_cri_upper')

    region_df_cri_max_male_pivot = region_df_cri_max_male.pivot(index='country', columns='year', values='obesity_cri_upper')





    fig, ax = plt.subplots(grid_rows, grid_cols, sharex='col', sharey='row', figsize=figsize)



    country_index = 0

    for i in range(grid_rows):

        for j in range(grid_cols):

            if country_index >= len(region_countries):

                pass

            else:

                x_vals = region_df_male_pivot.iloc[country_index].index



                y_male = region_df_male_pivot.iloc[country_index].values

                y_female = region_df_female_pivot.iloc[country_index].values



                y_male_cri_min = region_df_cri_min_male_pivot.iloc[country_index].values

                y_male_cri_max = region_df_cri_max_male_pivot.iloc[country_index].values



                y_female_cri_min = region_df_cri_min_female_pivot.iloc[country_index].values

                y_female_cri_max = region_df_cri_max_female_pivot.iloc[country_index].values



                l1 = ax[i,j].plot(x_vals, y_male, 'blue')

                l2 = ax[i,j].plot(x_vals, y_female, 'red')



                ax[i,j].fill_between(x_vals, (y_male_cri_min), (y_male_cri_max), color='blue', alpha=0.1)

                ax[i,j].fill_between(x_vals, (y_female_cri_min), (y_female_cri_max), color='red', alpha=0.1)



                ax[i,j].set_ylim([0, y_lim])



                label_dict = {

                    'United Kingdom of Great Britain and Northern Ireland':'United Kingdom / Northern Ireland',

                    'Democratic Republic of the Congo':'Dem. Rep. of the Congo',

                    'Saint Vincent and the Grenadines':'St. Vincent & the Grenadines',

                    'Venezuela (Bolivarian Republic of)':'Venezuela'

                }



                if region_countries[country_index] in label_dict.keys():

                    label = label_dict[region_countries[country_index]]



                else: 

                    label = region_countries[country_index]



                ax[i,j].set(xlabel=label)

                country_index += 1



    fig.legend((l1[0], l2[0]), ('Male', 'Female'), loc='upper right')



    fig.text(0, 0.5, 'Obesity Prevalence (%)', ha='center', rotation='vertical')

    fig.suptitle('{} Obesity Prevalence'.format(region))

    fig.tight_layout()

    fig.subplots_adjust(top=0.95)



    plt.show()
def get_number_of_countries(region_type, region):

    if region_type == 'region':

        region_df = obesity_df.loc[obesity_df.region == region]

    elif region_type == 'sub-region':

        region_df = obesity_df.loc[obesity_df['sub-region'] == region]

    else:

        return "Please enter a valid region type ('region' or 'sub-region')."

    

    return len(region_df.country.unique())
get_number_of_countries(region_type='region', region='Oceania')
plot_by_region(region_type='region', region='Oceania', grid_rows=4, grid_cols=4, y_lim=80, figsize=(16,16))
inflection_countries = ['Australia', 'New Zealand']
get_number_of_countries(region_type='region', region='Europe')
plot_by_region(region_type='region', region='Europe', grid_rows=10, grid_cols=4, y_lim=40, figsize=(16,24))
inflection_countries.append(['Sweden', 'Switzerland', 'Spain', 'Slovakia', 'Serbia', 'Romania', 

                             'Republic of North Macedonia', 'Netherlands', 'Norway', 'Poland', 

                             'Portugal', 'Luxembourg', 'Malta', 'Montenegro', 'Italy',

                             'Ireland', 'Iceland' ,'Hungary', 'Greece', 'Germany',

                             'France', 'Czechia', 'Denmark', 'Finland', 'Estonia',

                             'United Kingdom of Great Britain and Northern Ireland',

                             'Bosnia and Herzegovina', 'Belgium', 'Bulgaria', 'Croatia',

                             'Albania', 'Andora', 'Austria'])
get_number_of_countries(region_type='region', region='Americas')
plot_by_region(region_type='region', region='Americas', grid_rows=7, grid_cols=5, y_lim=50, figsize=(16,24))
inflection_countries.append(['United States of America', 'Canada', 'Argentina'])
get_number_of_countries(region_type='region', region='Asia')
plot_by_region(region_type='region', region='Asia', grid_rows=12, grid_cols=4, y_lim=50, figsize=(16,32))
inflection_countries.append(['China', 'Cyprus', "Democratic People's Republic of Korea", 'Israel', 'Japan', 'Republic of Korea', 'Singapore'])
get_number_of_countries(region_type='region', region='Africa')
plot_by_region(region_type='region', region='Africa', grid_rows=11, grid_cols=5, y_lim=50, figsize=(16,32))
sub_region_list = obesity_df['sub-region'].unique()

print('There are {} sub-regions identified in the dataset:'.format(len(sub_region_list)))

for country in sub_region_list:

    print('\t{}'.format(country))
ax = sns.barplot(x=obesity_subregion_mean['sub-region'], y=obesity_subregion_mean.obesity_prev, color='lawngreen')

ax.set(title='Mean Obesity Prevalence by Sub-Region, 1975-2016', ylabel='Mean Obesity Prevalence (%), 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
obesity_sub_male = obesity_df_male.groupby(['sub-region', 'year']).mean().reset_index(drop=False)

obesity_sub_female = obesity_df_female.groupby(['sub-region', 'year']).mean().reset_index(drop=False)



obesity_sub_male_pivot = obesity_sub_male[['sub-region', 'year', 'obesity_prev']].pivot(index='sub-region', columns='year', values='obesity_prev')

obesity_sub_female_pivot = obesity_sub_female[['sub-region', 'year', 'obesity_prev']].pivot(index='sub-region', columns='year', values='obesity_prev')
fig, ax = plt.subplots(6, 3, sharex='col', sharey='row', figsize=(12, 18))



country_index = 0

sub_region_list = obesity_sub_male_pivot.index



for i in range(6):

    for j in range(3):

        if country_index >= len(sub_region_list):

            pass

        else:

            x_vals = obesity_sub_male_pivot.iloc[country_index].index



            y_male = obesity_sub_male_pivot.iloc[country_index].values

            y_female = obesity_sub_female_pivot.iloc[country_index].values



            l1 = ax[i,j].plot(x_vals, y_male, 'blue')

            l2 = ax[i,j].plot(x_vals, y_female, 'red')



            ax[i,j].set_ylim([0, 60])



            ax[i,j].set(xlabel=sub_region_list[country_index])

            country_index += 1



fig.legend((l1[0], l2[0]), ('Male', 'Female'), loc='upper right')



fig.text(0, 0.5, 'Obesity Prevalence (%)', ha='center', rotation='vertical')

fig.suptitle('Global Obesity Prevalence by Sub-Region')

fig.tight_layout()

fig.subplots_adjust(top=0.95)



plt.show()
mf_diff_mean = mf_diff.mean(axis=1).sort_values(ascending=False)



ax = sns.barplot(x=mf_diff_mean[:20].index, y=mf_diff_mean[:20].values, color='orange')

ax.set(title='20 Highest Countries - % Difference between Mean Female and Male Obesity Prevalence, 1975-2016', ylabel='% Difference Mean Obesity Prevalence, 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
mf_diff_mean = mf_diff_mean.sort_values(ascending=True)



ax = sns.barplot(x=mf_diff_mean[:20].index, y=mf_diff_mean[:20].values, color='orange')

ax.set(title='20 Lowest Countries - % Difference between Mean Female and Male Obesity Prevalence, 1975-2016', ylabel='% Difference Mean Obesity Prevalence, 1975-2016', xlabel='')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
# obesity prevalence

flip_countries = ['Switzerland', 'Denmark', 'Sweden', 'Austria', 'Iceland', 'Germany']

flip_df = obesity_df.loc[obesity_df.country.isin(flip_countries), ['country', 'year', 'obesity_prev', 'obesity_cri_lower', 'obesity_cri_upper', 'sex']]



flip_df_female = flip_df.loc[flip_df.sex == 'Female', ['country', 'year', 'obesity_prev']]

flip_df_male = flip_df.loc[flip_df.sex == 'Male', ['country', 'year', 'obesity_prev']]



flip_df_female_pivot = flip_df_female.pivot(index='country', columns='year', values='obesity_prev')

flip_df_male_pivot = flip_df_male.pivot(index='country', columns='year', values='obesity_prev')



# credible intervals

flip_df_cri_min_female = flip_df.loc[flip_df.sex == 'Female', ['country', 'year', 'obesity_cri_lower']]

flip_df_cri_min_male = flip_df.loc[flip_df.sex == 'Male', ['country', 'year', 'obesity_cri_lower']]



flip_df_cri_min_female_pivot = flip_df_cri_min_female.pivot(index='country', columns='year', values='obesity_cri_lower')

flip_df_cri_min_male_pivot = flip_df_cri_min_male.pivot(index='country', columns='year', values='obesity_cri_lower')



flip_df_cri_max_female = flip_df.loc[flip_df.sex == 'Female', ['country', 'year', 'obesity_cri_upper']]

flip_df_cri_max_male = flip_df.loc[flip_df.sex == 'Male', ['country', 'year', 'obesity_cri_upper']]



flip_df_cri_max_female_pivot = flip_df_cri_max_female.pivot(index='country', columns='year', values='obesity_cri_upper')

flip_df_cri_max_male_pivot = flip_df_cri_max_male.pivot(index='country', columns='year', values='obesity_cri_upper')





fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(14,12))



country_index = 0

for i in range(3):

    for j in range(2):

        

        x_vals = flip_df_male_pivot.iloc[country_index].index

        

        y_male = flip_df_male_pivot.iloc[country_index].values

        y_female = flip_df_female_pivot.iloc[country_index].values

        

        y_male_cri_min = flip_df_cri_min_male_pivot.iloc[country_index].values

        y_male_cri_max = flip_df_cri_max_male_pivot.iloc[country_index].values

        

        y_female_cri_min = flip_df_cri_min_female_pivot.iloc[country_index].values

        y_female_cri_max = flip_df_cri_max_female_pivot.iloc[country_index].values

        

        l1 = ax[i,j].plot(x_vals, y_male, 'blue')

        l2 = ax[i,j].plot(x_vals, y_female, 'red')

        

        ax[i,j].fill_between(x_vals, (y_male_cri_min), (y_male_cri_max), color='blue', alpha=0.1)

        ax[i,j].fill_between(x_vals, (y_female_cri_min), (y_female_cri_max), color='red', alpha=0.1)

        

        #ax[i,j].plot(, flip_df_female_pivot.iloc[country_index].values, color='red')

        ax[i,j].set_ylim([0, 30])

        ax[i,j].set(xlabel=flip_countries[country_index])

        country_index += 1



fig.legend((l1[0], l2[0]), ('Male', 'Female'), loc='lower right')



fig.text(0.04, 0.5, 'Obesity Prevalence (%)', ha='center', rotation='vertical')

fig.suptitle('Countries Where Mean Obesity Prevalence for Males > Females')

plt.show()

obesity_both_df = obesity_df.loc[obesity_df.sex == 'Both sexes', :]



px.choropleth(

    locations=obesity_both_df.country.astype(str), 

    color=obesity_both_df.obesity_prev.astype(float), 

    hover_name=obesity_both_df.country.astype(str), 

    animation_frame=obesity_both_df.year.astype(int),

    color_continuous_scale=px.colors.sequential.Rainbow,

    range_color=[0,50],

    locationmode='country names',

    height=500,

    width=700,

    title='Global Obesity Prevalence - Both Sexes',

    projection='natural earth'

)
obesity_female_df = obesity_df.loc[obesity_df.sex == 'Female', :]



px.choropleth(

    locations=obesity_female_df.country.astype(str), 

    color=obesity_female_df.obesity_prev.astype(float), 

    hover_name=obesity_female_df.country.astype(str), 

    animation_frame=obesity_female_df.year.astype(int),

    color_continuous_scale=px.colors.sequential.Rainbow,

    range_color=[0,50],

    locationmode='country names',

    height=500,

    width=700,

    title='Global Obesity Prevalence - Female',

    projection='natural earth'

)
obesity_male_df = obesity_df.loc[obesity_df.sex == 'Male', :]



px.choropleth(

    locations=obesity_male_df.country.astype(str), 

    color=obesity_male_df.obesity_prev.astype(float), 

    hover_name=obesity_male_df.country.astype(str), 

    animation_frame=obesity_male_df.year.astype(int),

    color_continuous_scale=px.colors.sequential.Rainbow,

    range_color=[0,50],

    locationmode='country names',

    height=500,

    width=700,

    title='Global Obesity Prevalence - Both Sexes',

    projection='natural earth'

)