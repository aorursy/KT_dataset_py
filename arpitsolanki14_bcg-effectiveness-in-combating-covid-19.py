# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from datetime import datetime

import plotly.express as px

import datetime 

from scipy import stats



import re

from plotly.subplots import make_subplots



import plotly.graph_objects as go



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input/johnshopkinscovid/csse_covid_19_data/csse_covid_19_daily_reports/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



covid_data=pd.read_csv('/kaggle/input/hackathon/task_2-owid_covid_data-21_June_2020.csv')#OWID Covid cases data till 20 June

germany_province_data=pd.read_csv('/kaggle/input/hackathon/task_2-Gemany_per_state_stats_20June2020.csv',thousands=',')#Covid stats for Germany provinces

#tb_data=pd.read_csv('/kaggle/input/hackathon/task_2-Tuberculosis_infection_estimates_for_2018.csv')

bcg_world_atlas=pd.read_csv('/kaggle/input/hackathon/BCG_world_atlas_data-2020.csv')#BCG World Atlas dataset

#pop_age_group=pd.read_csv('/kaggle/input/country-population-by-age-group/population-by-broad-age-group.csv')

spain_age_group=pd.read_csv('/kaggle/input/portugal-spain-covid-cases-by-age-group/spain_cases_age_group.csv',thousands=",")#Covid cases by age group in Spain

portugal_age_group=pd.read_csv('/kaggle/input/portugal/portugal_cases_age_group.csv')#Covid cases by age group in Portugal

#ni_ireland_covid_data=pd.read_csv('/kaggle/input/ireland-covid-data/Covid_19_Northern_Ireland_Daily_Indicators.csv',parse_dates=True, dayfirst=True)

ireland_data=pd.read_csv('/kaggle/input/ireland-covid-data/ireland_cso_data.csv')#Covid stats from Ireland counties

bcg_pnas=pd.read_csv('../input/pnas-bcg-data/bcg_data_pnas_europe.csv')#BCG Coverage dataset for Europe from PNAS

de_covid_data=pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')#Germany covid related stats

unicef_bcg_coverage=pd.read_csv('../input/unicef-bcg-coverage/unicef_bcg_data.csv')#UNICEF BCG coverage related stats

stringency_index=pd.read_csv('../input/owid-covid-stringency-index/covid-stringency-index.csv')
bcg_world_atlas = bcg_world_atlas.rename(columns={'Contry Name (Mandatory field)': 'location', 'Is it mandatory for all children?': 'mandatory'})

bcg_world_atlas_col=bcg_world_atlas[['location','mandatory','BCG Strain ']]

bcg_world_atlas_col['mandatory']=bcg_world_atlas_col['mandatory'].fillna('Unknown')

bcg_world_atlas_col



bcg_world_atlas_col['BCG Strain ']=bcg_world_atlas_col['BCG Strain '].fillna('NA')

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Danish|Staten|SSI',regex=True),'BCG Strain ']='Danish'

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Japan|Tokyo',regex=True),'BCG Strain ']='Japan'

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Pasteur',regex=True),'BCG Strain ']='Pasteur'
stringency_index['Date']= pd.to_datetime(stringency_index['Date']) 

stringency_index.columns=['location','code','Date','stringency_index']



covid_data['date']=pd.to_datetime(covid_data['date']).dt.date

covid_data['Fatality_Rate']=covid_data['total_deaths']/covid_data['total_cases']



covid_data_no_world=covid_data.loc[covid_data['iso_code']!='OWID_WRL']



fig = px.line(covid_data_no_world, x="date", y="total_cases", color='location')

fig.update_layout(title_text='Total Covid confirmed Cases ',xaxis_title_text='date',yaxis_title_text='Total Cases',width=800,height=400)

fig.show()



fig = px.line(covid_data_no_world, x="date", y="total_deaths", color='location')

fig.update_layout(title_text='Total Covid Deaths',xaxis_title_text='date',yaxis_title_text='Daily New Cases',width=800,height=400)

fig.show()
covid_data_no_world=covid_data.loc[covid_data['iso_code']!='OWID_WRL']



fig = px.choropleth(covid_data_no_world, locations="iso_code",

                    color="total_deaths", # Death is a column of gapminder

                    hover_name="location", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title='Covid Deaths across Countries')

fig.show()
covid_data_no_world_jun=covid_data_no_world.loc[covid_data_no_world['date']==datetime.date(year=2020,month=6,day=20)]

covid_data_no_world_2000=covid_data_no_world_jun.loc[covid_data_no_world['total_cases']>=10000]

fig = px.scatter(covid_data_no_world_2000, x="total_cases_per_million", y="Fatality_Rate",log_x=True, color="continent",hover_name="location",text="location")

fig.update_traces(textposition='top center')

fig.update_layout(

    height=600,

    title_text='Total Cases vs Fatality Rate',

    yaxis_tickformat = '%'

)

fig.show()
tests=covid_data_no_world_2000.loc[covid_data_no_world_2000['total_tests'].notna()]

fig = px.scatter(tests, x="total_cases_per_million", y="Fatality_Rate",log_x=True,size='total_tests_per_thousand', color="continent",hover_name="location",text="location",size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=600,

    title_text='Total Cases vs Fatality Rate - Bubble size indicates tests_per_thousand',

    yaxis_tickformat = '%'

)

fig.show()
#Correl Matrix different variables

import seaborn as sns

covid_data_sel=covid_data_no_world_jun[['date','total_cases','total_deaths','Fatality_Rate','total_cases_per_million','gdp_per_capita','diabetes_prevalence','total_tests','total_tests_per_thousand','population_density','population','median_age','aged_65_older','hospital_beds_per_thousand','total_tests','total_tests_per_thousand','stringency_index','population_density']]

covid_data_sel['total_tests']=covid_data_sel['total_tests'].fillna(0)

#covid_data_sel['Test Positivity Rate']=covid_data_sel['total_cases']/covid_data_sel['total_tests']

covid_data_sel_cor=covid_data_sel.fillna(covid_data_sel.mean())

corr = covid_data_sel_cor.corr(method='spearman')

corr

ax = sns.heatmap(corr,linewidths=.5,cmap='YlGnBu')
#Sort dataframe on a column

covid_data_no_world_jun_rel=covid_data_no_world_jun.loc[covid_data_no_world_jun['total_cases']>1000]

covid_data_no_world_jun_rel_sort=covid_data_no_world_jun_rel.sort_values('Fatality_Rate',ascending=False)

max_fat=covid_data_no_world_jun_rel_sort.head(5)

min_fat=covid_data_no_world_jun_rel_sort.tail(5)

max_fat

fig = make_subplots(rows=1, cols=2,subplot_titles=("Maximum Fatality Rate", "Minimum Fatality Rate"))



fig.add_trace(

    go.Bar(x=max_fat['location'], y=max_fat['Fatality_Rate'],name='Countries'),

    row=1, col=1

)



fig.add_trace(

    go.Bar(x=min_fat['location'], y=min_fat['Fatality_Rate'],name='Countries'),

    row=1, col=2

)



fig.update_layout(height=400, width=800, title_text="Countries with highest & lowest Fatality Rate",yaxis_tickformat = '%')

fig.show()
#In the top 5 countries, have a look at the fatality rate. 

covid_data_no_world

fig = go.Figure()



country_list=['France','Belgium','Italy','United Kingdom','Hungary']

for i in range(5):

    fig.add_trace(

    go.Scatter(x=covid_data_no_world[covid_data_no_world['location']==country_list[i]]['date'], y=covid_data_no_world[covid_data_no_world['location']==country_list[i]]['Fatality_Rate'],name=country_list[i]),

    )

fig.update_layout(title='Countries with Highest Fatality Rates',yaxis_tickformat = '%',height=400)

fig.show()





fig1=go.Figure()

country_list=['Uzbekistan','Bahrain','Nepal','Qatar','Singapore']

for i in range(5):

    fig1.add_trace(

    go.Scatter(x=covid_data_no_world[covid_data_no_world['location']==country_list[i]]['date'], y=covid_data_no_world[covid_data_no_world['location']==country_list[i]]['Fatality_Rate'],name=country_list[i],mode='lines'),

    )

fig1.update_layout(title='Countries with Lowest Fatality Rates',yaxis_tickformat = '%',height=400)



fig1.show()
unicef_bcg_data=unicef_bcg_coverage[['Geographic area','TIME_PERIOD','OBS_VALUE']]

unicef_bcg_data.columns=['location','year','% BCG Coverage']

country_continent=covid_data[['location','continent']].drop_duplicates(keep='first')

unicef_bcg_data_continent=pd.merge(unicef_bcg_data,country_continent,how='inner')

#unicef_bcg_data_continent

unicef_bcg_data_2015=unicef_bcg_data_continent.loc[unicef_bcg_data_continent['year']==2015]

#unicef_bcg_data_2015

fig = px.histogram(unicef_bcg_data_2015, x="% BCG Coverage",title='Distribution of BCG coverage in 2015 across Countries',height=400)

fig.show()

unicef_bcg_data_2015[unicef_bcg_data_2015['% BCG Coverage']<=60]



unicef_bcg_data_continent_gp=unicef_bcg_data_continent.groupby(['continent','year']).agg({'% BCG Coverage':'median'}).reset_index()

fig = px.line(unicef_bcg_data_continent_gp, x="year", y="% BCG Coverage", title='BCG Coverage trends by Continent',color='continent')

fig.show()
country_continent=covid_data[['location','continent']]

country_continent=country_continent.drop_duplicates(subset=['location'], keep='first')

bcg_world_atlas['BCG Strain ']=bcg_world_atlas['BCG Strain '].fillna('NA')



bcg_world_atlas_country=pd.merge(bcg_world_atlas,country_continent,how="left",left_on='location',right_on='location')



#bcg_world_atlas_country['BCG Strain '].str.contains('Danish')



bcg_world_atlas_country.loc[bcg_world_atlas_country['BCG Strain '].str.contains('Danish|Staten|SSI',regex=True),'BCG Strain ']='Danish'

bcg_world_atlas_country.loc[bcg_world_atlas_country['BCG Strain '].str.contains('Japan|Tokyo',regex=True),'BCG Strain ']='Japan'

bcg_world_atlas_country.loc[bcg_world_atlas_country['BCG Strain '].str.contains('Pasteur',regex=True),'BCG Strain ']='Pasteur'



# # #bcg_world_atlas_country.loc[gp['BCG Strain'].str.contains('Danish'|'Staten'),'BCG Strain']='Danish'



gp=bcg_world_atlas_country.groupby(['BCG Strain ','continent']).agg({'location':'nunique'}).reset_index(drop=False)

gp.columns=['BCG Strain','continent','location']

gp.sort_values('location',ascending=False,inplace=True)

gp = gp[gp['BCG Strain']!='NA']

k=gp.pivot(index='BCG Strain', columns='continent', values='location').reset_index()

k.fillna(0, inplace=True)

k['total']=k.iloc[:,1:6].sum(axis=1)

k.sort_values('total',ascending=False).head(5)
bcg_world_atlas = bcg_world_atlas.rename(columns={'Contry Name (Mandatory field)': 'location', 'Is it mandatory for all children?': 'mandatory'})

bcg_world_atlas_col=bcg_world_atlas[['location','mandatory','BCG Strain ']]

bcg_world_atlas_col['mandatory']=bcg_world_atlas_col['mandatory'].fillna('Unknown')

bcg_world_atlas_col



bcg_world_atlas_col['BCG Strain ']=bcg_world_atlas_col['BCG Strain '].fillna('NA')

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Danish|Staten|SSI',regex=True),'BCG Strain ']='Danish'

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Japan|Tokyo',regex=True),'BCG Strain ']='Japan'

bcg_world_atlas_col.loc[bcg_world_atlas_col['BCG Strain '].str.contains('Pasteur',regex=True),'BCG Strain ']='Pasteur'


#bcg_world_atlas_col.loc[bcg_world_atlas_col['mandatory']=='Unknown','mandatory']='Z'

bcg_world_atlas_col=bcg_world_atlas_col.sort_values('mandatory',ascending=True)

bcg_world_atlas_col_unique=bcg_world_atlas_col.drop_duplicates(subset=['location'],keep='first')

#bcg_world_atlas_col.loc[bcg_world_atlas_col['mandatory']=='Z','mandatory']='Unknown'



covid_data_no_world.loc[covid_data_no_world['location']=='United States','location']='United States of America'

bcg_world_atlas_col_unique.loc[bcg_world_atlas_col_unique['location']=='France','mandatory']='no'

covid_data_no_world_bcg=pd.merge(left=covid_data_no_world,right=bcg_world_atlas_col_unique,how='inner')

covid_data_no_world_bcg.groupby(['mandatory']).agg({'location':'nunique'}).reset_index()
covid_data_no_world_bcg[covid_data_no_world_bcg['mandatory']=='Unknown']['location'].unique()

#covid_data_no_world_bcg.location.unique()
k=covid_data_no_world_bcg.groupby(['date','mandatory']).agg({'total_deaths':['sum'],'total_cases':['sum']}).reset_index()

k.columns=['date','mandatory','total_deaths','total_cases']

k['Fatality_Rate']=k['total_deaths']/k['total_cases']

fig = px.line(k, x="date", y="Fatality_Rate", color='mandatory')

fig.update_layout(title_text='Comparison of Fatality Rate by BCG Vaccine Status',yaxis_tickformat = '%')

fig.show()



yes_sample=covid_data_no_world_bcg.loc[(covid_data_no_world_bcg['mandatory'].isin(['yes']))&(covid_data_no_world_bcg['date']==datetime.date(year=2020,month=6,day=20))]

no_sample=covid_data_no_world_bcg.loc[(covid_data_no_world_bcg['mandatory'].isin(['no']))&(covid_data_no_world_bcg['date']==datetime.date(year=2020,month=6,day=20))]



yes_sample=yes_sample['Fatality_Rate']

no_sample=no_sample['Fatality_Rate']

#yes_sample



stats.ttest_ind(yes_sample,no_sample)
covid_data_no_world_bcg_1000=covid_data_no_world_bcg.loc[(covid_data_no_world_bcg['total_cases']>=1000)&(covid_data_no_world_bcg['date']==datetime.date(year=2020,month=6,day=20))]

bcg_yes_no_comp=covid_data_no_world_bcg_1000.groupby('mandatory').agg({'population':'median','total_cases':'median','total_cases_per_million':'median','total_tests':'median','total_tests_per_thousand':'median','extreme_poverty':'mean','diabetes_prevalence':'mean','cvd_death_rate':'median','hospital_beds_per_thousand':'mean','Fatality_Rate':'median','aged_65_older':'median','life_expectancy':'mean','female_smokers':'mean','male_smokers':'mean'})

bcg_yes_no_comp
tests=covid_data_no_world_bcg_1000.loc[covid_data_no_world_bcg_1000['total_tests'].notna()]

tests=tests.loc[covid_data_no_world_bcg_1000['mandatory'].notna()]



fig = px.scatter(tests, x="total_cases_per_million", y="Fatality_Rate",log_x=True,size='total_tests_per_thousand', color="mandatory",hover_name="location",text="location",size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=600,

    title_text='BCG Status - Total Cases vs Fatality Rate - Bubble Size indicates total_tests_per_thousand',

    yaxis_tickformat = '%'

)

fig.show()
fig = px.violin(covid_data_no_world_bcg_1000, x='mandatory', color='mandatory',y='Fatality_Rate',box=True) 

fig.update_layout(template='seaborn',title='Distribution of Fatality Rate for Countries divided by BCG Status',legend_title_text='State',yaxis_tickformat = '%')



fig.show()
bcg_yes=covid_data_no_world_bcg_1000.loc[covid_data_no_world_bcg_1000['mandatory']=='yes']



fig = px.violin(bcg_yes, x='BCG Strain ',y='Fatality_Rate',box=True) 

fig.update_layout(template='seaborn',title='Distribution of Fatality Rate for Countries divided by BCG Strain',legend_title_text='State',yaxis_tickformat = '%')



fig.show()
bcg_pnas_col=bcg_pnas[['Country','Age oldest  vaccinated','BCG Coverage']]



bcg_merge=pd.merge(bcg_pnas_col,covid_data_no_world_bcg_1000,left_on='Country',right_on='location',how='inner')



fig = px.scatter(bcg_merge, x="Age oldest  vaccinated", y="Fatality_Rate", trendline="ols",hover_name="location",text="location")

fig.update_traces(textposition='top center')

fig.update_layout(

    height=300,

    title_text='Age Oldest Vaccinated vs Fatality Rate',

    yaxis_tickformat = '%'

)

fig.show()

results = px.get_trendline_results(fig)

results.px_fit_results.iloc[0].summary()
bcg_yes=covid_data_no_world_bcg.loc[covid_data_no_world_bcg['mandatory']=='yes']

bcg_fat_max=bcg_yes.loc[bcg_yes['date']==datetime.date(year=2020,month=6,day=20)]

bcg_fat_max=bcg_fat_max.loc[bcg_fat_max['total_cases']>=1000]



bcg_fat_max=bcg_fat_max.sort_values('Fatality_Rate',ascending=False)

head=bcg_fat_max.head(5)

#head



bcg_no=covid_data_no_world_bcg.loc[covid_data_no_world_bcg['mandatory']=='no']

bcg_no_fat_max=bcg_no.loc[bcg_no['date']==datetime.date(year=2020,month=6,day=20)]

bcg_no_fat_max=bcg_no_fat_max.loc[bcg_no_fat_max['total_cases']>=1000]

bcg_no_fat_max=bcg_no_fat_max.sort_values('Fatality_Rate',ascending=False)


bcg_fatality_top5=covid_data_no_world_bcg[covid_data_no_world_bcg['location'].isin(['Hungary','Mexico','Algeria','Ireland','Niger'])]

fig = px.line(bcg_fatality_top5, x="date", y="Fatality_Rate", color='location')

fig.update_xaxes(nticks=10)

fig.update_layout(title='BCG Mandatory Status Countries - Highest Fatality Rate',yaxis_tickformat = '%',height=400)



fig.show()



bcg_fatality_bottom5=covid_data_no_world_bcg[covid_data_no_world_bcg['location'].isin(['Oman','Maldives','Qatar','Nepal','Singapore'])]

fig = px.line(bcg_fatality_bottom5, x="date", y="Fatality_Rate", color='location')

fig.update_layout(title='BCG Mandatory Status Countries - Lowest Fatality Rate',yaxis_tickformat = '%',height=400)

fig.show()
bcg_fatality_top5.groupby('location').agg({'total_cases':'max','total_cases_per_million':'median','population':'median','total_tests':'max','total_tests_per_thousand':'max','extreme_poverty':'mean','diabetes_prevalence':'mean','cvd_death_rate':'median','hospital_beds_per_thousand':'mean','Fatality_Rate':'median','aged_65_older':'median','life_expectancy':'mean','female_smokers':'mean','male_smokers':'mean'})
bcg_fatality_bottom5.groupby('location').agg({'total_cases':'max','total_cases_per_million':'median','population':'median','total_tests':'max','total_tests_per_thousand':'max','extreme_poverty':'mean','diabetes_prevalence':'mean','cvd_death_rate':'median','hospital_beds_per_thousand':'mean','Fatality_Rate':'median','aged_65_older':'median','life_expectancy':'mean','female_smokers':'mean','male_smokers':'mean'})
sppo=covid_data_no_world_bcg.loc[covid_data_no_world_bcg['location'].isin(['Spain','Portugal'])]



sp=sppo.loc[sppo['date']>=datetime.date(year=2020,month=3,day=10)]



sp['cases_per_million']=(sp['total_cases']/sp['population'])*1000000



fig = px.line(sp, x="date", y="total_cases", color='location')

fig.update_layout( title_text="Spain vs Portugal - Total Covid Cases",height=400)

fig.show()



fig = px.line(sp, x="date", y="cases_per_million", color='location')

fig.update_layout( title_text="Spain vs Portugal - Covid Cases per Million",height=400)

fig.show()



fig = px.line(sp, x="date", y="Fatality_Rate", color='location')

fig.update_layout( title_text="Spain vs Portugal - Fatality Rate",height=400,yaxis_tickformat = '%')

fig.show()



splo=stringency_index[(stringency_index.location.isin(['Spain','Portugal']))&(stringency_index['Date']>='2020-03-15')&(stringency_index['Date']<='2020-06-20')]



fig = px.line(splo, x="Date", y="stringency_index", color='location')

fig.update_layout( title_text="Spain vs Portugal - Lockdown Stringency Index",height=400)

fig.show()
spain_age_group.loc[spain_age_group['Age Group']=='80-89','Age Group']='80+'

spain_age_group.loc[spain_age_group['Age Group']=='?90','Age Group']='80+'



spain_age_group=spain_age_group.groupby('Age Group').agg({'Cases':'sum','Hospitalization':'sum','ICU':'sum','Deaths':'sum'}).reset_index()



spain_age_group.columns=['Age Group','Cases','Hospitalization','ICU','Deaths']



spain_age_group['percent_hospitalization']=spain_age_group['Hospitalization']/spain_age_group['Cases']

spain_age_group['percent_ICU']=spain_age_group['ICU']/spain_age_group['Cases']

spain_age_group['percent_death']=spain_age_group['Deaths']/spain_age_group['Cases']
fig = make_subplots(rows=3, cols=1)



fig.append_trace(go.Bar(

   x=spain_age_group['Age Group'], y=spain_age_group['percent_hospitalization'],name='Percent_Hospitalization'

), row=1, col=1)



fig.append_trace(go.Bar(

   x=spain_age_group['Age Group'], y=spain_age_group['percent_ICU'],name='Percent_ICU'

), row=2, col=1)



fig.append_trace(go.Bar(

   x=spain_age_group['Age Group'], y=spain_age_group['percent_death'],name='Percent_Deaths'

), row=3, col=1)





fig.update_layout(height=600, width=800, title_text="Spain - Covid Stats by Age Group",yaxis_tickformat = '%')

fig.show()
portugal_age_group.loc[portugal_age_group['Age Group']=='Oct-19','Age Group']='10-19'

portugal_age_group['percent_deaths']=portugal_age_group['Deaths']/portugal_age_group['Cases']



spain_age_group['Age Group']

spain_age_group.loc[spain_age_group['Age Group']=='?90','Age Group']='80+'





fig = go.Figure(data=[

    go.Bar(name='Portugal', x=portugal_age_group['Age Group'], y=portugal_age_group['percent_deaths']),

    go.Bar(name='Spain', x=spain_age_group['Age Group'], y=spain_age_group['percent_death'])

])

# Change the bar mode

fig.update_layout(barmode='group',title='Spain vs Portugal - Percentage of Positive cases by Age Group leading to Deaths')

fig.show()
covid_data_no_world_bcg.loc[(covid_data_no_world_bcg['location'].isin(['Spain','Portugal']))& (covid_data_no_world_bcg['date']==datetime.date(year=2020,month=6,day=20))]
ireland_county_data=pd.read_csv('../input/ireland-covid-data/ireland_cso_data.csv')

ireland_county_daily_data=pd.read_csv('../input/ireland-covid-data/Covid19CountyStatisticsHPSCIreland.csv')

ireland_covid_stats=pd.read_csv('../input/ireland-covid-data/Covid19CountyStatisticsHPSCIrelandOpenData.csv')

ireland_covid_stats=ireland_covid_stats[['CountyName','PopulationCensus16','Density']].drop_duplicates(keep='first')

ireland_county_data.columns=['County', 'total_deaths', 'median_age_deaths', 'total_cases','median_age_cases']

ireland_county_data=ireland_county_data.sort_values('total_cases',ascending=False)

ireland_county_data=pd.merge(ireland_county_data,ireland_covid_stats,left_on='County',right_on='CountyName',how='inner')



ireland_county_data['total_deaths']=pd.to_numeric(ireland_county_data['total_deaths'],errors='coerce').fillna(0)

ireland_county_data['Fatality_Rate']=ireland_county_data['total_deaths']/ireland_county_data['total_cases']

ireland_county_data['cases_per_million']=(ireland_county_data['total_cases']/ireland_county_data['PopulationCensus16'])*10000000

ireland_county_data['deaths_per_million']=(ireland_county_data['total_deaths']/ireland_county_data['PopulationCensus16'])*10000000



# fig = px.bar(ireland_county_data, x='CountyName', y='ConfirmedCovidCases')

# fig.update_layout(title='Ireland - Confirmed Cases by County')

# fig.show()



colors = ['lightslategray',] * 26

colors[1] = 'blue'

colors[18] = 'crimson'





fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed Covid Cases", "Covid Fatality Rate","Population Density"))



fig.add_trace(

    go.Bar(

    x=ireland_county_data['County'],

    y=ireland_county_data['total_cases'],text=ireland_county_data['total_cases'],

    marker_color=colors # marker color can be a single color value or an iterable

),

    row=1, col=1

)



fig.add_trace(

    go.Bar(

    x=ireland_county_data['County'],

    y=ireland_county_data['Fatality_Rate'],text=ireland_county_data['Fatality_Rate'],

    marker_color=colors # marker color can be a single color value or an iterable

),

    row=2, col=1

)



fig.add_trace(

    go.Bar(

    x=ireland_county_data['CountyName'],

    y=ireland_county_data['Density'],text=ireland_county_data['Density'],

    marker_color=colors # marker color can be a single color value or an iterable

),

    row=3, col=1

)



#fig.update_layout(height=400, width=800, title_text="Fatailty_Rate")







fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(title_text='Ireland - Confirmed Cases & Population Density by County',height=1000)



fig.show()
ireland_county_daily_data['date']=pd.to_datetime(ireland_county_daily_data['TimeStamp']).dt.date

ireland_county_daily_data['cases_per_million']=(ireland_county_daily_data['ConfirmedCovidCases']/ireland_county_daily_data['PopulationCensus16'])*10000000

ireland_county_daily_data_cork_kerry=ireland_county_daily_data.loc[(ireland_county_daily_data['CountyName']=='Cork')|(ireland_county_daily_data['CountyName']=='Kerry')]



fig = px.line(ireland_county_daily_data_cork_kerry, x="date", y="cases_per_million", color='CountyName')

fig.update_layout( title_text="Cork vs Kerry - Cases per Million",height=300,width=800)

fig.show()



k=ireland_county_data.loc[(ireland_county_data['County']=='Cork')|(ireland_county_data['County']=='Kerry')]

fig1 = px.bar(k, x='County', y='deaths_per_million')

fig1.update_layout(title='Cork vs Kerry - Deaths per Million',height=300)

fig1.show()

ireland_county_data.loc[(ireland_county_data['County']=='Cork')|(ireland_county_data['County']=='Kerry')]
fig=go.Figure()

fig.add_trace(go.Violin(y=ireland_county_data['median_age_cases'],name='median_age_cases',box_visible=True,meanline_visible=True))

fig.add_trace(go.Violin(y=ireland_county_data['median_age_deaths'],name='median_age_deaths',box_visible=True,meanline_visible=True))

fig.update_layout(title='Median age for Cases & Deaths across Counties in Ireland')

fig.update_yaxes(title="Median Age")



fig.show()
ireland_county_data[['Density','total_cases','Fatality_Rate','PopulationCensus16','cases_per_million','deaths_per_million','median_age_cases','median_age_deaths']].corr()
de_covid_data=pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')

germany_province_data=pd.read_csv('/kaggle/input/hackathon/task_2-Gemany_per_state_stats_20June2020.csv')

id_mapping=pd.read_csv('/kaggle/input/idmapping/state_id_mapping_de.csv')





uniques=germany_province_data[['State in Germany (German)','East/West','Population']].drop_duplicates()



de_covid_data['date']=pd.to_datetime(de_covid_data['date']).dt.date

de_covid_data

#germany_province_data

#de_covid_data.state.unique()



de_covid_data_ew=pd.merge(de_covid_data,uniques,left_on='state',right_on='State in Germany (German)',how='left')

#de_covid_data_ew.state.unique()



de_covid_data_ew.loc[de_covid_data_ew.state.str.contains('Baden'),'East/West']='West'

de_covid_data_ew.loc[de_covid_data_ew.state.str.contains('Thueringen'),'East/West']='East'

de_covid_data_ew.loc[de_covid_data_ew.state.str.contains('Baden'),'Population']=10879618

de_covid_data_ew.loc[de_covid_data_ew.state.str.contains('Thueringen'),'Population']=2170714



de_sum=de_covid_data_ew.groupby(['state','East/West','date','Population']).agg({'cases':'sum','deaths':'sum','recovered':'sum'}).reset_index()

de_sum.columns=['state','East/West','date','Population','cases','deaths','recovered']

#de_sum.state.unique()



de_sum['total_cases']=de_sum.groupby(by=['state','East/West','Population'])['cases'].cumsum()

de_sum['total_deaths']=de_sum.groupby(by=['state','East/West','Population'])['deaths'].cumsum()

#de_sum



de_sum['Fatality_Rate']=de_sum['total_deaths']/de_sum['total_cases']

de_sum=pd.merge(de_sum,id_mapping,how='left')

de_304=de_sum.loc[de_sum['date']==datetime.date(year=2020,month=4,day=30)]

de_305=de_sum.loc[de_sum['date']==datetime.date(year=2020,month=5,day=30)]

de_236=de_sum.loc[de_sum['date']==datetime.date(year=2020,month=6,day=23)]
#germany_province_data[['State in Germany (German)','East/West']].drop_duplicates()

#de_sum

from urllib.request import urlopen

import json





with urlopen('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/master/2_bundeslaender/2_hoch.geo.json') as response:

    geojson = json.load(response)



fig = px.choropleth(de_304, geojson=geojson, color="Fatality_Rate",

                    locations="id",color_continuous_scale="Viridis"

                    ,hover_name="East/West",

                   )

fig.update_geos(fitbounds="locations")

fig.update_layout(title='Germany - 30 April Covid Fatality Rate',height=400)

fig.show()



fig1 = px.choropleth(de_305, geojson=geojson, color="Fatality_Rate",

                    locations="id",color_continuous_scale="Viridis"

                    ,hover_name="East/West"

                   )

fig1.update_geos(fitbounds="locations")

fig1.update_layout(title='Germany - 30 May Covid Fatality Rate',height=400)



fig1.show()





fig2 = px.choropleth(de_236, geojson=geojson, color="Fatality_Rate",

                    locations="id",color_continuous_scale="Viridis"

                    ,hover_name="East/West"

                   )

fig2.update_geos(fitbounds="locations")

fig2.update_layout(title='Germany - 23 June Covid Fatality Rate',height=400)



fig2.show()
germany_concat=pd.concat([de_304,de_305,de_236]).reset_index()

germany_concat.date.unique()

fig = px.violin(germany_concat, x='date', color='East/West', y='Fatality_Rate',box=True, hover_name='state') 

fig.update_layout(template='seaborn',title='Distribution of Fatality Rates Across East/West Germany',legend_title_text='Region',xaxis = {

   'tickformat': '%d-%m',

   'tickmode': 'auto',

#   'nticks': value, [where value is the max # of ticks]

#   'tick0': value, [where value is the first tick]

#   'dtick': value [where value is the step between ticks]

},yaxis_tickformat = '%')



fig.show()
group=de_sum.groupby(['East/West','date']).agg({'total_deaths':'sum','total_cases':'sum'}).reset_index()

reg_pop=germany_province_data.groupby('East/West').agg({'Population':'sum'}).reset_index()

group=pd.merge(group,reg_pop,how='inner')



group['Fatality Rate']=group['total_deaths']/group['total_cases']

group['Deaths_per_Million']=(group['total_deaths']/group['Population'])*1000000





fig = px.line(group, x="date", y="total_cases", color='East/West')

fig.update_layout( title_text="Germany - Total Cases by Region",height=300,width=800)

fig.show()



fig = px.line(group, x="date", y="total_deaths", color='East/West')

fig.update_layout( title_text="Germany - Total Deaths by Region",height=300,width=800)

fig.show()



fig = px.line(group, x="date", y="Fatality Rate", color='East/West')

fig.update_layout( title_text="Germany - Fatality Rate by Region",yaxis_tickformat = '%',height=300,width=800)

fig.show()



fig = px.line(group, x="date", y="Deaths_per_Million", color='East/West')

fig.update_layout( title_text="Germany - Deaths per Million by Region",height=300,width=800)

fig.show()
de_age=de_covid_data_ew.groupby(['East/West','age_group']).agg({'cases':'sum','deaths':'sum'}).reset_index()

de_age_pop=pd.merge(de_age,reg_pop,how='inner')

de_age_pop.age_group = de_age_pop.age_group.astype('str')



de_age_pop['Fatality_Rate']=de_age_pop['deaths']/de_age_pop['cases']

de_age_pop['Deaths_per_Million']=(de_age_pop['deaths']/de_age_pop['Population'])*1000000



fig=px.bar(de_age_pop, x="age_group", y="Fatality_Rate", color='East/West',barmode='group')

fig.update_layout(xaxis_type='category',yaxis_tickformat = '%',title_text="Germany - Fatality Rate Age Group",height=250,margin=dict(l=20, r=20, t=25, b=25))

fig.show()





fig=px.bar(de_age_pop, x="age_group", y="Deaths_per_Million", color='East/West',barmode='group')

fig.update_layout(xaxis_type='category',title_text="Germany - Deaths per Million by Age Group",height=250,margin=dict(l=20, r=20, t=25, b=0))

fig.show()
germany_province_data=pd.read_csv('/kaggle/input/hackathon/task_2-Gemany_per_state_stats_20June2020.csv',thousands=',')#Covid stats for Germany provinces

germany_province_data['Deaths'] = germany_province_data['Deaths'].str.replace(',', '').astype(float)

germany_province_data['cases_per_million']=germany_province_data['Cases']/germany_province_data['Population']

germany_province_data['deaths_per_million']=germany_province_data['Deaths']/germany_province_data['Population']

germany_province_data['Fatality_Rate']=germany_province_data['Deaths']/germany_province_data['Cases']



germany_province_data[['Population Density','Fatality_Rate','cases_per_million','deaths_per_million','Population','Deaths','Cases']].corr()



#germany_province_data

# covid_data_no_world_bcg_1000['mandatory'=='no','mandatory_flag']=0

# covid_data_no_world_bcg_1000['mandatory'=='yes','mandatory_flag']=1

# covid_data_no_world_bcg_1000['mandatory'=='Unknown','mandatory_flag']=2

# data=covid_data_no_world_bcg_1000

# data['Test_positivity_Rate']=data['total_cases']/data['total_tests']



covid_data_no_world_bcg_1000

dummies=pd.get_dummies(covid_data_no_world_bcg_1000['mandatory'])

covid_data_no_world_bcg_1000_d=pd.concat([covid_data_no_world_bcg_1000,dummies],axis=1)

covid_data_no_world_bcg_1000_d
covid_data_no_world_bcg_1000_d.columns