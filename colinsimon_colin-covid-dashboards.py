# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_seq_items = 2000





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import plotly.express as px

import plotly.graph_objs as go

# import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



# modeling

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso, LassoCV, RidgeCV



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.svm import SVR
covid = pd.read_csv('../input/bingcovid0613/Bing-COVID19-Data-6-13.csv',engine='python')
# OLD:

# covid = pd.read_csv('../input/bingcovid67/bing-covid-6-7.csv',engine='python')
# set up column names

covid.columns= ['ID','Date','Confirmed','ConfirmedChange','Deaths','DeathsChange','Recovered','RecoveredChange','Latitude','Longitude','ISO2','ISO3','Country','AdminRegion1','AdminRegion2']



covid['ID'] = covid['ID'].astype(int)



# Create function to change empty strings to 0:

def insert_zeros(series):

    return [(float(x) if x else 0) for x in series]



# execute function as needed:

covid['ConfirmedChange'] = insert_zeros(covid['ConfirmedChange'])

covid['Deaths'] = insert_zeros(covid['Deaths'])

covid['DeathsChange'] = insert_zeros(covid['DeathsChange'])

covid['Recovered'] = insert_zeros(covid['Recovered'])

covid['RecoveredChange'] = insert_zeros(covid['RecoveredChange'])

covid['Latitude'] = insert_zeros(covid['Latitude'])

covid['Longitude'] = insert_zeros(covid['Longitude'])



# set various cols to float data type:

covid[['ID',

       'Confirmed',

       'ConfirmedChange',

       'Deaths',

       'DeathsChange',

       'Recovered',

       'RecoveredChange',

       'Latitude',

       'Longitude']] = covid[['ID',

                  'Confirmed',

                  'ConfirmedChange',

                  'Deaths',

                  'DeathsChange',

                  'Recovered',

                  'RecoveredChange',

                  'Latitude',

                  'Longitude'

                 ]].astype(float)



# set date to datetime data type:

covid['Date'] = pd.to_datetime(covid['Date'])
start_date = '2020-03-01' # Start March 1

end_date = '2020-06-11' # End June 11



covid = covid[covid['Date'] >= start_date] 

covid = covid[covid['Date'] <= end_date] 
# get USA data

usa = covid.loc[covid['Country'] == 'United States']



# US State population and HDI score data

# get state population data

state_pop=pd.read_csv('../input/statedata/2019-state-pop.csv')

state_pop = state_pop[['NAME', 'POPESTIMATE2019']]

useless = ['United States', 'Northeast Region', 'Midwest Region', 'South Region','West Region','Puerto Rico']

state_pop = state_pop[~state_pop['NAME'].isin(useless)]

state_pop.rename(columns={'NAME': 'State', 'POPESTIMATE2019': '2019 State Population'}, inplace=True)



# map population data onto main dataframe

pop_mapping = dict(state_pop[['State', '2019 State Population']].values)

pd.options.mode.chained_assignment = None

usa['Population'] = usa.AdminRegion1.map(pop_mapping)





# get state HDI data from globaldatalab.org

state_hdi = pd.read_csv('../input/statedata/2018-state-hdi.csv') # 2018 is most recent.

state_hdi = state_hdi[['Region', '2018']] # obtain necessary columns



# set up useless rows

useless = ['Total'] 

state_hdi = state_hdi[~state_hdi['Region'].isin(useless)] # remove useless rows

state_hdi.rename(columns={'Region': 'State', '2018': '2018 HDI Score'}, inplace=True)



# map HDI scores onto main dataframe

hdi_mapping = dict(state_hdi[['State', '2018 HDI Score']].values)

usa['2018 HDI Score'] = usa.AdminRegion1.map(hdi_mapping)
statelist = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}





usa['Infection Rate'] = usa['Confirmed'] / usa['Population'] # add infection rate

usa['Mortality Rate'] = usa['Deaths'] / usa['Confirmed'] # add mortality rate

usa['State'] = usa['AdminRegion1']

usa.replace({'State': statelist}, inplace=True) # add state list

usa.dropna(subset = ['AdminRegion1'], inplace=True) # drop county rows

usa = usa.loc[usa['AdminRegion2'].isna()] # remove county subset data



# set negative changed values to 0

usa['ConfirmedChange'] = np.where((usa.ConfirmedChange < 0), 0 ,usa.ConfirmedChange)
# Drop state rows

world = covid.loc[covid['AdminRegion1'].isna()]

# drop "worldwide" row

world = world.loc[world['Country'] != 'Worldwide']

world = world[world.Country != 'Congo']
# Fix China and Congo names

world.loc[world['Country'] == 'China (mainland)', 'Country'] = 'China'

world.loc[world['Country'] == 'Hong Kong SAR', 'Country'] = 'Hong Kong'



# set negative changed values to 0

world['ConfirmedChange'] = np.where((world.ConfirmedChange < 0), 0 ,world.ConfirmedChange)

world['ConfirmedChange'] = world['ConfirmedChange'].fillna(0)



# latest date

world_current = world[world['Date'] == end_date]



# grab top 60 countries:

world_top = world[world['Date'] == end_date].sort_values('Confirmed').tail(60)

toplist = world_top['Country']

world_top = world[world['Country'].isin(toplist)]

# US map with slider

colscale = [

            [0, 'rgb(224,255,224)'],

            [0.02, 'rgb(31,120,0)'],

            [0.10, 'rgb(175,175,0)'],

            [1, 'rgb(227,26,28)']]



fig = px.choropleth(

                    usa,

                   locations='State',

                   color='Deaths',

                   hover_name='State',

                   locationmode= 'USA-states',

                    animation_frame=usa["Date"].dt.strftime('%Y-%m-%d'),

                    color_continuous_scale=colscale,

                    range_color = (0,10_000)

                   )

fig.update_layout(

    title_text = f'US COVID Deaths: {start_date} - {end_date}, 2020',

    geo_scope='usa')

fig.show()
# US map with slider

colscale = [[0, 'rgb(224,255,224)'],

            [0.02, 'rgb(31,120,0)'],

            [0.10, 'rgb(175,175,0)'],

            [1, 'rgb(227,26,28)']]



fig = px.choropleth(

                    usa,

                   locations='State',

                   color='Confirmed',

                   hover_name='State',

                   locationmode= 'USA-states',

                    animation_frame=usa["Date"].dt.strftime('%Y-%m-%d'),

                    color_continuous_scale=colscale,

                    range_color = (0,200_000)

                   )

fig.update_layout(

    title_text = f'US COVID-19 Confirmed Cases: {start_date} - {end_date}, 2020',

    geo_scope='usa')

fig.show()
bubs = px.scatter(usa,

                x = 'Confirmed',

                y = 'Deaths',

                size = (usa['Population']/500_000+3),

                color = usa['2018 HDI Score'],

                hover_name='State',

                animation_frame=usa["Date"].dt.strftime('%Y-%m-%d'),

                range_x=[1.3,6],

                range_y=[0,5]

#                 log_x=True,

#                 log_y=True

                   )

bubs.update_layout(

    title_text = f'US States by COVID-19 Cases and Deaths through {end_date}, 2020(log scale)',

    

    xaxis_type='log',

    yaxis_type='log'

)

bubs.show()
bubs = px.scatter(usa.sort_values('State', ascending=False),

                x = 'Date',

                y = 'State',

                size = 'ConfirmedChange',

                color = 'ConfirmedChange',

                color_continuous_scale = 'viridis',

                height = 1000,

                   )

bubs.update_layout(

    title_text = f'US States by New COVID-19 Cases: {start_date} - {end_date}',

    yaxis = dict(dtick = 1)

)

bubs.show()
bubbles = px.scatter(world_top.sort_values('Country', ascending=False),

                x = 'Date',

                y = 'Country',

                size = 'ConfirmedChange',

                color = 'ConfirmedChange',

                color_continuous_scale = 'viridis',

                height = 1500,

                   )

bubbles.update_layout(

    title_text = f'Daily Confirmed Cases (Top 60 Countries) {start_date} - {end_date}',

    yaxis = dict(dtick = 1)

)

bubbles.show()
fig = px.bar(world_current.sort_values('Confirmed').tail(15),

            x='Confirmed',

            y='Country',

            orientation='h', 

            width=700

            )

fig.update_layout(title=f'COVID-19: Top 15 Countries by Confirmed Cases (Cumulative), {end_date}', 

                      xaxis_title="", 

                      yaxis_title="", 

                      yaxis_categoryorder = 'total ascending',

                      uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
# World map with slider



colscale = [

            [0, 'rgb(224,255,224)'],

            [0.02, 'rgb(31,120,0)'],

            [0.10, 'rgb(175,175,0)'],

            [1, 'rgb(227,26,28)']]



fig = px.choropleth(

                    world,

                   locations='Country',

                   color='Deaths',

                   hover_name='Country',

                   locationmode= 'country names',

                    animation_frame=world["Date"].dt.strftime('%Y-%m-%d'),

                    color_continuous_scale=colscale,

                    range_color = (0,100_000)

                   )

fig.update_layout(

    title_text = f'World COVID Deaths: {start_date} - {end_date}, 2020',

)

fig.show()
# World map with slider



colscale = [

            [0, 'rgb(224,255,224)'],

            [0.02, 'rgb(31,120,0)'],

            [0.10, 'rgb(175,175,0)'],

            [1, 'rgb(227,26,28)']]



fig = px.choropleth(

                    world,

                   locations='Country',

                   color='Confirmed',

                   hover_name='Country',

                   locationmode= 'country names',

                    animation_frame=world["Date"].dt.strftime('%Y-%m-%d'),

                    color_continuous_scale=colscale,

                    range_color = (0,1_000_000)

                   )

fig.update_layout(

    title_text = f'Worldwide COVID-19 Confirmed Cases: {start_date} - {end_date}',

)

fig.show()
hdi = pd.read_csv('../input/hdiclean25/HDI-2.5.csv')

pd.set_option('display.max_columns',300)

pd.set_option('display.max_rows',300)

hdi.loc[hdi['Country'] == 'Congo', 'Country'] = 'Congo (DRC)'



# ideal number of features:

print(f'Maximum number of features: {np.sqrt(hdi.shape[0]).round(0)}')
def viewer(df):

    view = pd.DataFrame()

    view['dtypes'] = df.dtypes

    view['nunique'] = df.nunique()

    view['nans'] = df.isna().sum()

    view['mean'] = df.mean().round(2)

    view['std dev'] = df.std().round(2)

    return view.T
viewer(hdi)
# HDI feature correlation Heatmap:

data = hdi.drop(['Country'], axis=1)



corr = data.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool)) # Generate a mask for the upper triangle

f, ax = plt.subplots(figsize=(20, 16)) # Set up the matplotlib figure

cmap = sns.diverging_palette(220, 10, as_cmap=True) # Generate a custom diverging colormap



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='coolwarm',

#             vmax=.3, 

            center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Heatmap of HDI feature correlation');
# only include countries with 1 million pop or more:

hdi_large = hdi[hdi['Population millions'] >= 1.0]



# create union:

union = []

for i in list(world['Country'].unique()):

    if i in list(hdi_large['Country']):

        union.append(i)



# look at differences:

stragglers = []

for i in list(hdi_large['Country'].unique()):

    if i not in union:

        stragglers.append(i)

print(f'# of stragglers:{len(stragglers)}')

print(f'stragglers: {stragglers}')



# only include inner joins:

world_cut = world[world['Country'].isin(union)]

hdi_cut = hdi[hdi['Country'].isin(union)]

same_length = len(world_cut['Country'].unique()) == len(hdi_cut['Country'].unique())

length = len(hdi_cut['Country'])



# view results:

print(f'Same length? : {same_length}')

print(f'# of countries: {length}')
# merge world and hdi datasets:

combined = world_cut.merge(hdi_cut, on=['Country'])



# Clean dataset:

combined.drop(['AdminRegion1', 'AdminRegion2', 'ISO2', 'ISO3'], axis=1, inplace=True)

combined['Deaths'] = combined['Deaths'].fillna(0)

combined['DeathsChange'] = combined['DeathsChange'].fillna(0)

combined['Recovered'] = combined['Recovered'].fillna(0)

combined['RecoveredChange'] = combined['RecoveredChange'].fillna(0)



# Infection and mortality rates

combined['infection_rate'] = combined['Confirmed'] / combined['Population millions']

combined['mortality_rate'] = combined['Deaths'] / combined['Confirmed']

combined['mortality_rate'] = combined['mortality_rate'].fillna(0)
# Test Dates

mar1 = combined[combined['Date'] == '2020-03-01']

apr1 = combined[combined['Date'] == '2020-04-01']

may1 = combined[combined['Date'] == '2020-05-01']

jun1 = combined[combined['Date'] == '2020-06-01']
# MASS FEATURES LIST:



features = [

#         'index', 

#         'ID', 

#         'Date', 

#         'Confirmed', 

#         'ConfirmedChange', 

#         'Deaths',

#        'DeathsChange', 

#         'Recovered', 

#         'RecoveredChange', 

#         'Latitude', 

#         'Longitude',

#        'Country', 

        '2018 HDI category', 

        '2018 HDI', 

        'Life expectancy at birth',

       'Expected years of schooling',

        'Mean years of schooling',

       'Gross national income (GNI) per capita PPP $',

       'GNI per capita rank minus HDI rank', 

        '2018 HDI rank', 

#         '1990 HDI',

#        '2000 HDI', 

#         '2010 HDI', 

#         '2013 HDI', 

#         '2015 HDI', 

#         '2016 HDI', 

#         '2017 HDI',

#        '2018 HDI.1', 

#         'Change in HDI rank ',

#        'Average annual HDI growth 1990-2000',

#        'Average annual HDI growth 2000-2010',

#        'Average annual HDI growth 2010-2018',

#        'Average annual HDI growth 1990-2018',

       'Inequality-adjusted HDI (IHDI) ', 

#         'IHDI loss %',

#        'IHDI diff from HDI rank', 

        'Coefficient of human inequality',

       'Inequality in life expectancy %',

       'Inequality-adjusted life expectancy index',

       'Inequality in education %', 

        'Inequality-adjusted education index',

       'Inequality in income %', 

        'Inequality-adjusted income index',

       '% income share of poorest 40 pct', 

        '% income share of richest 10pct',

       'Gini Coefficient %', 

        'GDI Overall', 

        'HDI female', 

        'HDI male',

       'Life expectancy female', 

        'Life expectancy male',

       'Expected schooling female', 

        'Expected schooling male',

       'Mean schooling female', 

        'Mean schooling male',       #28

        'GNI per capita female',

       'GNI per capita male', 

        'Gender Inequality Index',

       'Gender Inequality Rank', 

        'Maternal mortality per 100,000',

       'Teenage maternity per 1000', 

        'Female % of parliament',

       'Secondary Education rate female', 

        'Secondary Education rate male',

       'Labor force participation females', 

        'Labor force participation males',

       'Population millions', 

        'Population millions 2030 estimate',

       'Avg annual growth 2005-2010', 

        'Avg annual growth 2015-2020',

       ' Urban %', 

        'Under age 5-millions', 

        'Ages 15–64 millions',

       'Ages 65 and older millions', 

        'Median age',

       'Young dependency rate per 100',    #49

        'Seniors dependency rate per 100',

       'fertility rate 2005-2010', 

        'Fertility rate 2015-2020',

       'Infants lacking DPT vaccine',       #53

        'Infants lacking measles vaccine',

       'Infant mortality per 1000', 

        'Under five mortality per 1000',

       'Female adult per 1000', 

        'Male adult per 1000',

       'Noninfectious disease mortality Female per 100k', #59

       'Noninfectious disease mortality male per 100k', 

        'Malaria per 1000',

       'Tuberculosis per 100,000', 

        'HIV positive %, adult',    #63

       'Healthy life expectancy at birth', #64

        'Healthcare % of GDP',

       'Pop % with secondary education', 

        'GDP billions', 

        'GDP per capita',

       'GDP per capita growth',  #64

        'Fixed capital % of gdp',

       'Domestic credit provided by financial sector',

       'Consumer price index 2018 vs 2010', 

        'Employment to population ratio',

       'Labour force participation rate', #69

        'agriculture % of employment',

       'Services % of employment', 

        'Unemployment %', 

        'Under 24 unemployment %',

       '% Working poor at PPP$3.20 a day', 

        'High-skill to low-skill ratio', #75

       'Birth registration', 

        'Refugees-thousands ',

       'Homeless per million natural disasters', 

        'Prisoners per 100k',

       'Suicides per 100k female', 

        'Suicides per 100k male',

       'Exports and imports %GDP', 

        'Foreign investment inflows %GDP',

       'Remittances, inflows %GDP',

        'Net migration per 1000',

       'Stock of immigrants %pop', 

        'International inbound tourists thousands',

       'Internet users pct', 

        'Mobile phone subscriptions per 100',

       'Mobile subscription pct chg', 

        'Physicians per 10k', #91

       'Vulnerable employment %', 

        'Rural population electricity access',

       'Population basic drinking water', 

        'Population basic sanitation',

       'Male-female birth ratio', 

        'Youth unemployment female to male',

       'F-M unemployment rate', 

        '% female in parliament', 

#         'infection_rate',

#        'mortality_rate'

]
# ****REDUCED FEATURES LIST****



features2 = [

# #         'index', 

# #         'ID', 

# #         'Date', 

# #         'Confirmed', 

# #         'ConfirmedChange', 

# #         'Deaths',

# #        'DeathsChange', 

# #         'Recovered', 

# #         'RecoveredChange', 

# #         'Latitude', 

# #         'Longitude',

# #        'Country', 

#         '2018 HDI category', 

#         '2018 HDI', 

#         'Life expectancy at birth',

#        'Expected years of schooling',

#         'Mean years of schooling',

#        'Gross national income (GNI) per capita PPP $',

#        'GNI per capita rank minus HDI rank', 

#         '2018 HDI rank', 

# #         '1990 HDI',

# #        '2000 HDI', 

# #         '2010 HDI', 

# #         '2013 HDI', 

# #         '2015 HDI', 

# #         '2016 HDI', 

# #         '2017 HDI',

# #        '2018 HDI.1', 

# #         'Change in HDI rank ',

# #        'Average annual HDI growth 1990-2000',

# #        'Average annual HDI growth 2000-2010',

# #        'Average annual HDI growth 2010-2018',

# #        'Average annual HDI growth 1990-2018',

#        'Inequality-adjusted HDI (IHDI) ', 

# #         'IHDI loss %',

# #        'IHDI diff from HDI rank', 

#         'Coefficient of human inequality',

#        'Inequality in life expectancy %',

#        'Inequality-adjusted life expectancy index',

#        'Inequality in education %', 

#         'Inequality-adjusted education index',

#        'Inequality in income %', 

#         'Inequality-adjusted income index',

#        '% income share of poorest 40 pct', 

#         '% income share of richest 10pct',

       'Gini Coefficient %',                #19

#         'GDI Overall', 

#         'HDI female', 

#         'HDI male',

#        'Life expectancy female', 

#         'Life expectancy male',

#        'Expected schooling female', 

#         'Expected schooling male',

#        'Mean schooling female', 

#         'Mean schooling male',       #28

#         'GNI per capita female',

#        'GNI per capita male', 

#         'Gender Inequality Index',

#        'Gender Inequality Rank', 

#         'Maternal mortality per 100,000',

#        'Teenage maternity per 1000', 

#         'Female % of parliament',

#        'Secondary Education rate female', 

#         'Secondary Education rate male',

#        'Labor force participation females', 

#         'Labor force participation males',

#        'Population millions', 

#         'Population millions 2030 estimate',

#        'Avg annual growth 2005-2010', 

#         'Avg annual growth 2015-2020',

#        ' Urban %', 

#         'Under age 5-millions', 

#         'Ages 15–64 millions',

#        'Ages 65 and older millions', 

#         'Median age',

#        'Young dependency rate per 100',          #49

#         'Seniors dependency rate per 100',

#        'fertility rate 2005-2010', 

#         'Fertility rate 2015-2020',

#        'Infants lacking DPT vaccine',       #53

#         'Infants lacking measles vaccine',

#        'Infant mortality per 1000', 

#         'Under five mortality per 1000',

#        'Female adult per 1000', 

#         'Male adult per 1000',

#        'Noninfectious disease mortality Female per 100k',    #59

#        'Noninfectious disease mortality male per 100k', 

#         'Malaria per 1000',

#        'Tuberculosis per 100,000', 

#         'HIV positive %, adult',        #63

#        'Healthy life expectancy at birth',     #64

#         'Healthcare % of GDP',

#        'Pop % with secondary education', 

#         'GDP billions', 

#         'GDP per capita',

#        'GDP per capita growth',           #64

#         'Fixed capital % of gdp',

#        'Domestic credit provided by financial sector',

#        'Consumer price index 2018 vs 2010', 

#         'Employment to population ratio',

#        'Labour force participation rate',         #69

#         'agriculture % of employment',

#        'Services % of employment', 

#         'Unemployment %', 

#         'Under 24 unemployment %',

#        '% Working poor at PPP$3.20 a day', 

#         'High-skill to low-skill ratio',         #75

#        'Birth registration', 

#         'Refugees-thousands ',

#        'Homeless per million natural disasters', 

#         'Prisoners per 100k',

#        'Suicides per 100k female', 

#         'Suicides per 100k male',

#        'Exports and imports %GDP', 

#         'Foreign investment inflows %GDP',

#        'Remittances, inflows %GDP',

#         'Net migration per 1000',

#        'Stock of immigrants %pop', 

        'International inbound tourists thousands',   #87

#        'Internet users pct', 

#         'Mobile phone subscriptions per 100',

#        'Mobile subscription pct chg', 

        'Physicians per 10k',             #91

#        'Vulnerable employment %', 

#         'Rural population electricity access',

#        'Population basic drinking water', 

#         'Population basic sanitation',

#        'Male-female birth ratio', 

#         'Youth unemployment female to male',

#        'F-M unemployment rate', 

#         '% female in parliament', 

# #         'infection_rate',

# #        'mortality_rate'

]
# set up variables

X = jun1[features]

y = jun1['mortality_rate']



# train-test-split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



# Scale data

ss = StandardScaler()

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)



# instantiate and fit Linear Regression

lr = LinearRegression()

lr.fit(X_train, y_train)

# score linear Regression

lr_cvs = cross_val_score(lr, X_train, y_train, cv=5)

lr_train_score = lr.score(X_train, y_train)

lr_test_score = lr.score(X_test, y_test)

lr_preds = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_preds)



# instantiate and fit Lasso

l_alphas = np.logspace(-3, 2, 100)

lasso = LassoCV(

    alphas = l_alphas,

    cv = 5,

    max_iter=10_000)

lasso.fit(X_train, y_train)

# score LASSO

lasso_cvs = cross_val_score(lasso, X_train, y_train, cv=5)

lasso_train_score = lasso.score(X_train, y_train)

lasso_test_score = lasso.score(X_test, y_test)

lasso_preds = lasso.predict(X_test)

lasso_r2 = r2_score(y_test, lasso_preds)



# svr

svr = SVR(C=1., epsilon = 0.1, kernel = 'rbf')

svr.fit(X_train, y_train)

svr_cv = cross_val_score(svr, X_train, y_train, cv=5)

svr_preds = svr.predict(X_test)

svr_r2 = r2_score(y_test, svr_preds)



print(f'lr cv score: {lr_cvs.mean()}')

print(f'lr train score: {lr_train_score}')

print(f'lr test score: {lr_test_score}')

print(f'lr r2 value: {lr_r2}')

print('-'*18)

print(f'lasso best alpha: {lasso.alpha_}')

print(f'lasso cv score: {lasso_cvs.mean()}')

print(f'lasso train score: {lasso_train_score}')

print(f'lasso test score: {lasso_test_score}')

print(f'lasso r2 value: {lasso_r2}')

print('-'*18)

print(f'svr cv score: {svr_cv.mean()}')

print(f'svr r2 value: {svr_r2}')
# plot residual errors and observe

lr_residuals = y_test - lr_preds

lasso_residuals = y_test - lasso_preds

svr_residuals = y_test - svr_preds

plt.scatter(lr_preds, lr_residuals, color = 'green')

plt.scatter(lasso_preds, lasso_residuals, color = 'orange')

plt.scatter(svr_preds, svr_residuals, color = 'blue');

plt.scatter(lasso_preds, lasso_residuals);
# Examine coefficients:

for i in list(enumerate(lasso.coef_)):

    print(i)
# look at significant coefficients:

coef_set = list(enumerate(lasso.coef_))

sig_coefs = []

for i in lasso.coef_:

    if abs(i) > 0.01:

        sig_coefs.append(i)

len(sig_coefs)
for key, val in coef_set:

    if abs(val) > 0.01:

        print(key)
# Function to observe features

def regression(df, x_var, y_var, dates):

    # empty list for results

    results = []



    # iterate through dates

    for date in dates:

        data = df[df['Date'] == date]

        x = data[[x_var]]

        y = data[y_var]



        lr = LinearRegression()

        lr.fit(x, y)

        score = lr.score(x, y)

        results.append([date,score])

    return results  
datelist = combined['Date'].unique()



# Create Arrays to run

gini = combined[['Date', 'mortality_rate', 'Gini Coefficient %']]

docs = combined[['Date', 'mortality_rate', 'Physicians per 10k']]

tour = combined[['Date', 'mortality_rate', 'International inbound tourists thousands']]



# Run function and extract results

gini_results = regression(gini, 'Gini Coefficient %', 'mortality_rate', datelist)

docs_results = regression(docs, 'Physicians per 10k', 'mortality_rate', datelist)

tour_results = regression(tour, 'International inbound tourists thousands', 'mortality_rate', datelist)



# Merge Dataframes

gini_df = pd.DataFrame.from_records(gini_results)

gini_df.columns = ['Date', 'Gini Coefficient']

docs_df = pd.DataFrame.from_records(docs_results)

docs_df.columns = ['Date', 'Physicians Per Capita']

tour_df = pd.DataFrame.from_records(tour_results)

tour_df.columns = ['Date', "Int'l Inbound Tourists"]

final = gini_df.merge(docs_df, left_on='Date', right_on='Date')

final = final.merge(tour_df, left_on='Date', right_on='Date')
# Plotly Express Graph:

fig = px.line(final, x='Date', y =['Gini Coefficient','Physicians Per Capita',"Int'l Inbound Tourists"], labels={'variable':'2018 HDI Attributes'})

# fig.add_trace(x=docs_df['Date'], y=docs_df['Physicians Per Capita'])

fig.update_layout(

    title_text = 'Global R-Squared Values Targeting COVID-19 Mortality Rates Over Time'

)

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_text='R-Squared Values Against Mortality Rates')

fig.show()