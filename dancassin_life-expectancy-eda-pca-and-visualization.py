import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

import math

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.stats.mstats import winsorize

from sqlalchemy import create_engine

import warnings



import plotly.express as px
import pandas as pd

le_df = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")
le_df.info()



#output cut out some of the columns. hacked it.

le_df[:5].T.head(22)
# I'm going to adjust the columns names and remove extra spacing for ease of use

le_df.rename(columns = lambda x: x.strip().replace(' ', '_').lower(), inplace=True)



# one column doesn't match our underscoring convention or the Kaggle description. fixing.

le_df.rename(columns = {'thinness__1-19_years':'thinness_10-19_years'}, inplace=True)



print((f'Number of columns: {len(le_df.columns)}'))

le_df.columns

# that's better
# checking for nans

le_df.isnull().sum()
# will fill nans multiple ways to see what approach is best

le_df2 = le_df.copy() # will fill by mean

le_df3 = le_df.copy() # will fill mean by country

le_df4 = le_df.copy() # will fill by interpolation



countries = le_df2['country'].unique()

na_cols = ['life_expectancy', 'adult_mortality', 'alcohol', 'hepatitis_b',

       'bmi', 'polio', 'total_expenditure','diphtheria', 'gdp', 'population', 

        'thinness_10-19_years', 'thinness_5-9_years', 

        'income_composition_of_resources', 'schooling']



# fill with overall mean

for col in na_cols:

    le_df2[col].fillna(le_df2[col].mean(), inplace=True)



# mean by country

for col in na_cols:

    for country in countries:

        le_df3.loc[le_df3['country']== country, col] = le_df3.loc[le_df3['country'] == country, col].fillna(

            le_df3[le_df3['country'] == country][col].mean())

        

# interpolated by entire df

# due to missing values, I did not interpolate by country as there are too many missing values

for col in na_cols:

    le_df4.loc[:,col] = le_df4.loc[:,col].interpolate(limit_direction='both')

#printing nulls for each method

dfs = [le_df, le_df2, le_df3, le_df4]

df_names = ['le_df', 'le_df2', 'le_df3', 'le_df4']



for name, df in zip(df_names, dfs):

    print('_'*60)

    print(f'nulls for {name}')

    print('_'*60)

    print(df.isnull().sum())
#plotting each method by column

plt.figure(figsize=(13,60))



for i, col in enumerate(na_cols):

    df = pd.concat([le_df[col], le_df2[col], le_df3[col], le_df4[col]], axis=1)



    plt.subplot(len(na_cols), 3, i+1)

    plt.bar(['original', 'overall mean', 'mean by ctry', 'interpolate'],df.median(), color=('xkcd:cranberry') )

    plt.title(f'Mod of {col}')

    plt.xticks(rotation=45)



plt.tight_layout()

plt.show()
#let's look at the distributions of our continuous variables

num_cols = ['life_expectancy', 'adult_mortality',

       'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',

       'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure',

       'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_10-19_years',

       'thinness_5-9_years', 'income_composition_of_resources', 'schooling']



# detecting outliers

plt.figure(figsize=(20,60))



for i, col in enumerate(num_cols):

    plt.subplot(len(na_cols), 4, i+1)

    sns.boxplot(le_df4[col], color=('xkcd:lime'))

    plt.title(f'{col}', fontsize=18)

    plt.xlabel('')



plt.tight_layout()

plt.show()
# detecting outliers and distribution via histograms

plt.figure(figsize=(20,60))



for i, col in enumerate(num_cols):

    plt.subplot(len(na_cols), 4, i+1)

    sns.distplot(le_df4[col], color=('xkcd:green'))

    plt.title(f'Distribution of {col}', fontsize=18)

    plt.xlabel('')

    plt.axvline(le_df4.loc[:,col].mean(), color=('xkcd:cranberry')) #red line is the mean



plt.tight_layout()

plt.show()
# dropping mistakes in the data collection per the reasoning above

adj_le_df4 = le_df4[(le_df4.infant_deaths<1000) & (le_df4.measles<1000) & (le_df4['under-five_deaths']<1000)]



# winsorizations

adj_le_df4['winz_life_exp'] = winsorize(adj_le_df4['life_expectancy'], (0.10,0.0))

adj_le_df4['winz_tot_exp'] = winsorize(adj_le_df4['total_expenditure'], (0.0,0.10))

adj_le_df4['winz_adult_mort'] = winsorize(adj_le_df4['adult_mortality'], (0.0,0.10))

adj_le_df4['winz_polio'] = winsorize(adj_le_df4['polio'], (0.15,0.0))

adj_le_df4['winz_diph'] = winsorize(adj_le_df4['diphtheria'], (0.10,0.0))

adj_le_df4['winz_hepb'] = winsorize(adj_le_df4['hepatitis_b'], (0.15,0.0))

adj_le_df4['winz_thin_1019_yr'] = winsorize(adj_le_df4['thinness_10-19_years'], (0.0,0.10))

adj_le_df4['winz_thin_59_yr'] = winsorize(adj_le_df4['thinness_5-9_years'], (0.0,0.10))

adj_le_df4['winz_income_comp'] = winsorize(adj_le_df4['income_composition_of_resources'], (0.10,0.0))

adj_le_df4['winz_schooling'] = winsorize(adj_le_df4['schooling'], (0.10,0.05))

adj_le_df4['winz_under5_deaths'] = winsorize(adj_le_df4['under-five_deaths'], (0.0, 0.20))

adj_le_df4['winz_infant_deaths'] = winsorize(adj_le_df4['infant_deaths'], (0.0, 0.15))

adj_le_df4['winz_hiv/aids'] = winsorize(adj_le_df4['hiv/aids'], (0.0, 0.21))

adj_le_df4['winz_measles'] = winsorize(adj_le_df4['measles'], (0.0, 0.17))





# transformations

adj_le_df4['winz_log_gdp'] = winsorize(np.log(adj_le_df4['gdp']), (0.10, 0.0))

adj_le_df4['winz_log_population'] = winsorize(np.log(adj_le_df4['population']), (0.10, 0.0))

adj_le_df4['log_pct_exp'] = np.log(adj_le_df4['percentage_expenditure'])
# reinspecting to see how outliers were handled

adj_num_cols = [ 'winz_life_exp', 'winz_tot_exp',

       'winz_adult_mort', 'winz_polio', 'winz_diph', 'winz_hepb',

       'winz_thin_1019_yr', 'winz_thin_59_yr', 'winz_income_comp',

       'winz_schooling', 'winz_under5_deaths', 'winz_infant_deaths',

       'winz_log_gdp', 'winz_log_population', 'log_pct_exp', 'winz_hiv/aids',

       'winz_measles']



plt.figure(figsize=(20,90))



for i, col in enumerate(adj_num_cols):

    plt.subplot(len(adj_num_cols), 6, i+1)

    sns.boxplot(y=adj_le_df4[col], color=('xkcd:purple'))

    plt.title(f'{col}', fontsize=18)

    plt.ylabel('')



plt.tight_layout()

plt.show()

# all outliers have been dealt with
# correlation heat map

adj_corr = adj_le_df4[['year', 'status', 'country', 'winz_life_exp', 'winz_tot_exp',

       'winz_adult_mort', 'winz_polio', 'winz_diph', 'winz_hepb', 'bmi',

       'winz_thin_1019_yr', 'winz_thin_59_yr', 'winz_income_comp',

       'winz_schooling', 'winz_under5_deaths', 'winz_infant_deaths',

       'winz_log_gdp', 'winz_log_population', 'log_pct_exp', 'winz_hiv/aids',

       'winz_measles']].corr()



plt.figure(figsize=(15,12))

sns.heatmap(adj_corr, square=True, annot=True, cmap='viridis');

#Let's take a visual of these correlations

plt.figure(figsize=(8,6))



plt.subplot(2,2,1)

# Turns out winz_hiv/aids is actually a categorical variable, so I am using a bar plot

ax = sns.barplot(x='winz_hiv/aids', y='winz_life_exp', data=adj_le_df4, color='magenta')

ax.set_ylim(30,80)



plt.subplot(2,2,2)

ax1 = sns.scatterplot(x='winz_life_exp', y='winz_adult_mort', data=adj_le_df4, color='magenta')



plt.subplot(2,2,3)

ax2 = sns.scatterplot(x='winz_life_exp', y='winz_income_comp', data=adj_le_df4, color='magenta')



plt.subplot(2,2,4)

ax3 = sns.scatterplot(x='winz_life_exp', y='winz_schooling', data=adj_le_df4, color='magenta')



plt.tight_layout()

plt.show;
# Comparing Developed vs Developing

# In order to do this, we will need to correct the incorrectly labeled countries



# List of incorrectly classified countries

incorr_status = ['Canada', 'Chile', 'Greece', 'Finland', 'France', 'Israel', 'Republic of Korea']



# Loop through and change them to Developed

for country in incorr_status:

    adj_le_df4['status'].loc[adj_le_df4.country == country] = 'Developed'



# Verify

print(adj_le_df4[adj_le_df4['status']=='developed']['country'].unique()) 

# Verified;
# Plotting Developed v Developing

plt.figure(figsize=(8,5))



sns.barplot(y='status', x='winz_life_exp', data=adj_le_df4, orient='h', 

            palette = ['xkcd:cranberry','xkcd:neon green'], errcolor='grey');

plt.title('Life Expectancy by Country Status', fontsize=20)

plt.xlabel('Life Expectancy (Winsorized)')

plt.ylabel('');
# Differences in life expectancy seem drastic

# Running an independent T Test to see if results are significant

stats.ttest_ind(adj_le_df4[adj_le_df4['status']=='Developed']['winz_life_exp'],

               adj_le_df4[adj_le_df4['status']=='Developing']['winz_life_exp'])
# trying my hand at a sort of sickness ratio and health ratio

adj_le_df4['sickness_index'] = ((adj_le_df4['winz_infant_deaths']+adj_le_df4['winz_measles']+

                                adj_le_df4['winz_under5_deaths']+

                                 adj_le_df4['winz_hiv/aids'])/4)*((adj_le_df4['winz_thin_59_yr']+

                                                                   adj_le_df4['winz_thin_1019_yr'])/2)



adj_le_df4['health_index'] = (adj_le_df4['winz_measles']+adj_le_df4['winz_polio']+

                                 adj_le_df4['winz_diph'])/3



# Creating binary variable for Developed and Developing replacing Status

adj_le_df4 = pd.concat([adj_le_df4, pd.get_dummies(adj_le_df4.status)], axis=1)
# dropping non-transformed columns and low correlation items

adj_le_df4.drop(['status', 'life_expectancy', 'adult_mortality',

       'infant_deaths', 'percentage_expenditure', 'hepatitis_b',

       'measles', 'under-five_deaths', 'polio', 'total_expenditure',

       'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_10-19_years',

       'thinness_5-9_years', 'income_composition_of_resources', 'schooling', 'winz_tot_exp',

        'winz_hepb', 'Developing', 'winz_log_population', 'winz_measles',

        'winz_log_gdp', 'year', 'alcohol', 'bmi', 'winz_polio', 'winz_diph'], 

        axis=1, inplace=True)



plt.figure(figsize=(15,12))

suite_corr = adj_le_df4.corr()



sns.heatmap(suite_corr, square=True, annot=True);
PCA_df = adj_le_df4[['winz_life_exp', 'winz_adult_mort', 'winz_thin_1019_yr',

       'winz_thin_59_yr', 'winz_income_comp', 'winz_schooling',

       'winz_under5_deaths', 'winz_infant_deaths', 'winz_hiv/aids']]



stndzd_PCA_df = StandardScaler().fit_transform(PCA_df)



sklearn_PCA = PCA(n_components=4)



PCs = sklearn_PCA.fit_transform(stndzd_PCA_df)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA: \n',

    #sklearn_PCA.components_,

    sklearn_PCA.explained_variance_ratio_,

    '\n Eigenvalues of each component: \n',

    sklearn_PCA.explained_variance_

    

)
# Let's visualize the above values

fig, ax = plt.subplots(figsize=(10,5))



ax1 = plt.subplot(121)

plt.plot(sklearn_PCA.explained_variance_)

ax1.set_xticks([0,1,2,3])

ax1.set_xticklabels([1,2,3,4])

ax1.set_xlabel('Components')

ax1.set_ylabel('Eigenvalues')





ax2 = fig.add_subplot(122)

plt.plot(np.cumsum(sklearn_PCA.explained_variance_ratio_))

ax2.set_ylabel('% of Variance Explained')

ax2.set_xlabel('Components')

ax2.set_xticks([0,1,2,3])

ax2.set_xticklabels([1,2,3,4])

ax2.set_ylim(.6,1);



plt.tight_layout()
# Bear with me as I create a dictionary of ISO country codes for plotly

ISOs_Country = {

'ABW':'Aruba', 'AFG':'Afghanistan','AGO':'Angola','AIA':'Anguilla','ALA':'Åland Islands','ALB':'Albania',

'AND':'Andorra','ARE':'United Arab Emirates','ARG':'Argentina','ARM':'Armenia','ASM':'American Samoa',

'ATA':'Antarctica','ATF':'French Southern Territories','ATG':'Antigua and Barbuda','AUS':'Australia',

'AUT':'Austria','AZE':'Azerbaijan','BDI':'Burundi','BEL':'Belgium','BEN':'Benin',

'BES':'Bonaire, Sint Eustatius and Saba','BFA':'Burkina Faso','BGD':'Bangladesh','BGR':'Bulgaria',

'BHR':'Bahrain','BHS':'Bahamas','BIH':'Bosnia and Herzegovina','BLM':'Saint Barthélemy','BLR':'Belarus',

'BLZ':'Belize','BMU':'Bermuda','BOL':'Bolivia (Plurinational State of)','BRA':'Brazil','BRB':'Barbados',

'BRN':'Brunei Darussalam','BTN':'Bhutan','BVT':'Bouvet Island','BWA':'Botswana','CAF':'Central African Republic',

'CAN':'Canada','CCK':'Cocos (Keeling) Islands','CHE':'Switzerland','CHL':'Chile','CHN':'China',

'CIV':'Côte d\'Ivoire','CMR':'Cameroon','COD':'Democratic Republic of the Congo','COG':'Congo',

'COK':'Cook Islands','COL':'Colombia','COM':'Comoros','CPV':'Cabo Verde','CRI':'Costa Rica','CUB':'Cuba',

'CUW':'Curaçao','CXR':'Christmas Island','CYM':'Cayman Islands','CYP':'Cyprus','CZE':'Czechia',

'DEU':'Germany','DJI':'Djibouti','DMA':'Dominica','DNK':'Denmark','DOM':'Dominican Republic','DZA':'Algeria',

'ECU':'Ecuador','EGY':'Egypt','ERI':'Eritrea','ESH':'Western Sahara','ESP':'Spain','EST':'Estonia',

'ETH':'Ethiopia','FIN':'Finland','FJI':'Fiji','FLK':'Falkland Islands (Malvinas)','FRA':'France',

'FRO':'Faroe Islands','FSM':'Micronesia (Federated States of)','GAB':'Gabon',

'GBR':'United Kingdom of Great Britain and Northern Ireland','GEO':'Georgia','GGY':'Guernsey','GHA':'Ghana',

'GIB':'Gibraltar','GIN':'Guinea','GLP':'Guadeloupe','GMB':'Gambia','GNB':'Guinea-Bissau',

'GNQ':'Equatorial Guinea','GRC':'Greece','GRD':'Grenada','GRL':'Greenland','GTM':'Guatemala',

'GUF':'French Guiana','GUM':'Guam','GUY':'Guyana','HKG':'Hong Kong','HMD':'Heard Island and McDonald Islands',

'HND':'Honduras','HRV':'Croatia','HTI':'Haiti','HUN':'Hungary','IDN':'Indonesia','IMN':'Isle of Man',

'IND':'India','IOT':'British Indian Ocean Territory','IRL':'Ireland','IRN':'Iran (Islamic Republic of)',

'IRQ':'Iraq','ISL':'Iceland','ISR':'Israel','ITA':'Italy','JAM':'Jamaica','JEY':'Jersey','JOR':'Jordan',

'JPN':'Japan','KAZ':'Kazakhstan','KEN':'Kenya','KGZ':'Kyrgyzstan','KHM':'Cambodia','KIR':'Kiribati',

'KNA':'Saint Kitts and Nevis','KOR':'Republic of Korea','KWT':'Kuwait','LAO':'Lao People\'s Democratic Republic',

'LBN':'Lebanon','LBR':'Liberia','LBY':'Libya','LCA':'Saint Lucia', 'LIE':'Liechtenstein','LKA':'Sri Lanka',

'LSO':'Lesotho','LTU':'Lithuania','LUX':'Luxembourg','LVA':'Latvia','MAC':'Macao',

'MAF':'Saint Martin (French part)','MAR':'Morocco','MCO':'Monaco','MDA':'Republic of Moldova',

'MDG':'Madagascar','MDV':'Maldives','MEX':'Mexico','MHL':'Marshall Islands','MKD':'North Macedonia',

'MLI':'Mali','MLT':'Malta','MMR':'Myanmar','MNE':'Montenegro','MNG':'Mongolia','MNP':'Northern Mariana Islands',

'MOZ':'Mozambique','MRT':'Mauritania','MSR':'Montserrat','MTQ':'Martinique','MUS':'Mauritius','MWI':'Malawi',

'MYS':'Malaysia','MYT':'Mayotte','NAM':'Namibia','NCL':'New Caledonia','NER':'Niger','NFK':'Norfolk Island',

'NGA':'Nigeria','NIC':'Nicaragua','NIU':'Niue','NLD':'Netherlands','NOR':'Norway','NPL':'Nepal','NRU':'Nauru',

'NZL':'New Zealand','OMN':'Oman','PAK':'Pakistan','PAN':'Panama','PCN':'Pitcairn','PER':'Peru','PHL':'Philippines',

'PLW':'Palau','PNG':'Papua New Guinea','POL':'Poland','PRI':'Puerto Rico','PRK':'Democratic People\'s Republic of Korea',

'PRT':'Portugal','PRY':'Paraguay','PSE':'Palestine, State of','PYF':'French Polynesia','QAT':'Qatar','REU':'Réunion',

'ROU':'Romania','RUS':'Russian Federation','RWA':'Rwanda','SAU':'Saudi Arabia','SDN':'Sudan','SEN':'Senegal',

'SGP':'Singapore','SGS':'South Georgia and the South Sandwich Islands','SHN':'Saint Helena, Ascension and Tristan da Cunha',

'SJM':'Svalbard and Jan Mayen','SLB':'Solomon Islands','SLE':'Sierra Leone','SLV':'El Salvador','SMR':'San Marino',

'SOM':'Somalia','SPM':'Saint Pierre and Miquelon','SRB':'Serbia','SSD':'South Sudan','STP':'Sao Tome and Principe',

'SUR':'Suriname','SVK':'Slovakia','SVN':'Slovenia','SWE':'Sweden','SWZ':'Eswatini','SXM':'Sint Maarten (Dutch part)',

'SYC':'Seychelles','SYR':'Syrian Arab Republic','TCA':'Turks and Caicos Islands','TCD':'Chad','TGO':'Togo',

'THA':'Thailand','TJK':'Tajikistan','TKL':'Tokelau','TKM':'Turkmenistan','TLS':'Timor-Leste','TON':'Tonga',

'TTO':'Trinidad and Tobago','TUN':'Tunisia','TUR':'Turkey','TUV':'Tuvalu','TWN':'Taiwan, Province of China',

'TZA':'United Republic of Tanzania','UGA':'Uganda','UKR':'Ukraine','UMI':'United States Minor Outlying Islands',

'URY':'Uruguay','USA':'United States of America','UZB':'Uzbekistan','VAT':'Holy See','VCT':'Saint Vincent and the Grenadines',

'VEN':'Venezuela (Bolivarian Republic of)','VGB':'Virgin Islands (British)','VIR':'Virgin Islands (U.S.)','VNM':'Viet Nam',

'VUT':'Vanuatu','WLF':'Wallis and Futuna','WSM':'Samoa','YEM':'Yemen','ZAF':'South Africa','ZMB':'Zambia',

'ZWE':'Zimbabwe'}
# Turning above dict into df

ISO_df = pd.DataFrame.from_dict(ISOs_Country, orient='index', columns=['country'])
# Grouping WHO df by Country and aggregating the continuous variables we want to plot

reduced_adj_df = adj_le_df4.groupby('country')['winz_life_exp', 'Developed', 

                                               'winz_income_comp', 'winz_schooling'].agg(

    {'winz_life_exp':'mean', 'Developed':'min', 'winz_income_comp':'mean', 'winz_schooling':'mean'}) 

# Merging ISO and WHO dfs together

merged = pd.merge(reduced_adj_df, ISO_df, left_index=True, right_on='country')
# Plotting via the Plotly library

import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=4, cols=1,

    row_heights=[0.25, 0.25, 0.25, 0.25],

    vertical_spacing=0.025,

    subplot_titles=("World Life Expectancy", "Status of Countries", 

                    "Income Composition of Resources", "Highest Average Age of Schooling"),

    specs=[[{"type": "Choropleth", "rowspan": 1}],

           [{"type": "Choropleth", "rowspan": 1}],

          [{"type": "Choropleth", "rowspan": 1}],

           [{"type": "Choropleth", "rowspan": 1}]])



fig.add_trace( # Life Expectancy

    go.Choropleth(locations = merged.index,

                  z= merged['winz_life_exp'], 

                  text=merged['country'],

                  name='Life Expectancy',

                  colorbar={'title':'Life<br>Expectancy', 'len':.25, 'x':.99,'y':.896},

                  colorscale='spectral',),

    row=1,col=1

)



fig.add_trace( #Developed v Developing

    go.Choropleth(locations = merged.index,

                  z= merged['Developed'], 

                  text=merged['country'],

                  name='Status of Countries',

                  colorbar={'len':.227, 'x':.99,'y':.629, 'tickmode':'array','nticks':2,

                           'tickvals':[0,1], 'ticktext':('Developing', 'Developed')},

                  colorscale='burgyl_r'),

    row=2,col=1

)



fig.add_trace( # Income Comp

    go.Choropleth(locations = merged.index,

                  z= merged['winz_income_comp'], 

                  text=merged['country'],

                  name='Income Composition of Resources',

                  colorbar={'title':'Index', 'len':.24, 'x':.99,'y':.378},

                  colorscale='bluered',),

    row=3,col=1

)



fig.add_trace( #Schooling

    go.Choropleth(locations = merged.index,

                  z= merged['winz_schooling'], 

                  text=merged['country'],

                  name='Highest Average Age of Schooling',

                  colorbar={'len':.248, 'x':.99,'y':.1275, 'title':'Schooling<br>Age'},

                  colorscale='burgyl_r'),

    row=4,col=1

)



fig.update_layout(

    margin=dict(r=1, t=30, b=10, l=30),

    width=700,

    height=1400,

)



fig.show()