import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt #plotting, math, stats

%matplotlib inline

import seaborn as sns #plotting, regressions, stats
#Dataset from the World Health Organization

World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/full_data(14).csv")



plt.figure(figsize=(21,8)) # Figure size

plt.title('Cases across the world as of April 6, 2020') # Title

World.groupby("location")['total_cases'].max().plot(kind='bar', color='teal')
World.corr().style.background_gradient(cmap='magma')
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')



#I droped FIPS column. 

##not relevant for this analysis.

USA=df.drop(['fips','county'], axis = 1) 

USA
plt.figure(figsize=(19,17))

plt.title('Cases by state') # Title

sns.lineplot(x="date", y="cases", hue="state",data=USA, palette="Paired")

plt.xticks(USA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
##For ease of visualization

NY=USA.loc[USA['state']== 'New York']

LA=USA.loc[USA['state']== 'Louisiana']

WA=USA.loc[USA['state']== 'Washington']

IL=USA.loc[USA['state']== 'Illinois']

Mich=USA.loc[USA['state']== 'Michigan']

PUR=USA.loc[USA['state']== 'Puerto Rico']

# Concatenate dataframes 

States=pd.concat([NY,LA,WA,IL,PUR,Mich]) 



States=States.sort_values(by=['date'], ascending=True)

States

plt.figure(figsize=(15,9))

plt.title('COVID-19 cases comparison of WA, IL, NY, LA, PR, and Michigan') # Title

sns.lineplot(x="date", y="cases", hue="state",data=States)

plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
USAg=USA.groupby(['date']).max()

USAg
USAg=USAg.sort_values(by=['cases'], ascending=True)

USAg
Vuln = pd.read_csv("../input/uncover/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv")
Vuln= Vuln[['state', 'e_uninsur', 'epl_pov','epl_unemp','epl_age65','epl_age17','epl_disabl']]
# converting and overwriting values in column 

Vuln["state"]=Vuln["state"].str.lower()

Vuln["state"]=Vuln["state"].str.title()
Vuln.head()
Vuln.describe()
Vuln.corr().style.background_gradient(cmap='viridis')
census = pd.read_csv("../input/uncover/us_cdc/us_cdc/500-cities-census-tract-level-data-gis-friendly-format-2019-release.csv")
census=census[['stateabbr','placename', 'geolocation', 'bphigh_crudeprev',

               'stroke_crudeprev', 'obesity_crudeprev', 'diabetes_crudeprev','arthritis_crudeprev',

               'cancer_crudeprev', 'casthma_crudeprev', 'copd_crudeprev', 'csmoking_crudeprev', 

               'highchol_crudeprev', 'kidney_crudeprev']]

census
#COPD prevalence

plt.figure(figsize=(19,7)) # Figure size

census.groupby("stateabbr")['copd_crudeprev'].max().plot(kind='bar', color='olive')
census=census.replace(to_replace =("ND","OK", "UT", 'AK', 'SD','AL','AR'),

                 value =("North Dakota", "Oklahoma", 'Utah', "Alaska", "South Dakota", "Alabama", "Arkansas"))
census=census.replace(to_replace =("NC","OR", "NV", 'AZ', 'SC','CA','CO'),

                 value =("North Carolina", "Oregon", 'Nevada', "Arizona", "South Carolina", "California", "Colorado"))
census=census.replace(to_replace =("MN","WY", "WV", 'WI', 'WA','VT','VA'),

                 value =("Minnessota", "Wyoming", 'West Virginia', "Wisconsin", "Washington", "Vermont", "Virginia"))
census=census.replace(to_replace =("FL","NE", "MT", 'HI', 'LA','NM','GA','KS'),

                 value =("Florida", "Nebraska", 'Montana', "Hawaii", "Louisiana", "New Mexico", "Georgia", "Kansas"))
census=census.replace(to_replace =("NY","NJ", "OH", 'RI', 'PA','TX','ID','KY'),

                 value =("New York", "New Jersey", 'Ohio', "Rhode Island", "Pennsylvania", "Texas", "Idaho", "Kentucky"))
census=census.replace(to_replace =("CT","DC", "DE", 'IA', 'IL','IN','MD','MA'),

                 value =("Connecticut", "District of Columbia", 'Delaware', "Iowa", "Illinios", "Indiana", "Maryland", "Massachussetts"))
census=census.replace(to_replace =("ME","MI", "MO", 'MS', 'TN'),

                 value =("Maine", "Michigan", 'Missouri', "Mississippi", "Tennessee"))
#arthritis prevalence

plt.figure(figsize=(19,7)) # Figure size

census.groupby("stateabbr")['arthritis_crudeprev'].max().plot(kind='bar', color='peru')
census=census.drop(['placename', 'geolocation'], axis = 1) 

census = census.rename(columns={'stateabbr': 'state'})
census = census.rename(columns={'bphigh_crudeprev': 'high bp prev', 'stroke_crudeprev': 'stroke prev'})

census=census.rename(columns={'diabetes_crudeprev': 'diabetes prev', 'cancer_crudeprev': 'cancer prev', 'arthritis_crudeprev': 'arthritis prev'})
census=census.rename(columns={'casthma_crudeprev': 'asthma prev', 'copd_crudeprev': 'copd prev', 'csmoking_crudeprev': 'smoking prev'})
census=census.rename(columns={'highchol_crudeprev': 'highChol prev', 'kidney_crudeprev': 'kidney prev'})

census
census.describe()
census.corr().style.background_gradient(cmap='cividis')
chronic = pd.read_csv("../input/uncover/us_cdc/us_cdc/u-s-chronic-disease-indicators-cdi.csv")
# iterating the columns 

for col in chronic.columns: 

    print(col)
chronic=chronic[['locationdesc','topic','question','datavalue']]

#replace NaNs with zeros in the df

chronic=chronic.fillna(0)
chronic = chronic.rename(columns={'locationdesc': 'state','datavalue': 'rate of illness','topic': 'chronic illness','question': 'specific illness'})
chronic.head(3)
plt.figure(figsize=(22,6)) # Figure size

plt.title('US chronic illnesses') # Title

sns.countplot(chronic['chronic illness'])

plt.xticks(rotation=45)
chronic.describe()
chronic.corr().style.background_gradient(cmap='cool')
rank = pd.read_csv("../input/uncover/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv")

rank=rank[['state','num_deaths', 'percent_female','percent_excessive_drinking', 

           'num_uninsured','percent_vaccinated','percent_black','percent_american_indian_alaska_native',

           'percent_asian', 'percent_native_hawaiian_other_pacific_islander', 'percent_hispanic', 

           'percent_non_hispanic_white']]

rank.head()
plt.figure(figsize=(16,8)) # Figure size

plt.title('States pre-COVID19 morbidity ranks') # Title

rank.groupby("state")['num_deaths'].max().plot(kind='bar', color='darkred')
rank.describe()
rank.corr().style.background_gradient(cmap='inferno')
stats = pd.read_csv("../input/uncover/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv")

#replace NaNs with zeros in the df

stats=stats.fillna(0)

# iterating the columns 

for col in stats.columns: 

    print(col)
stats.drop(['hash', 'fips', 'datechecked'], axis=1, inplace=True)

stats.head()
plt.figure(figsize=(14,8)) # Figure size

plt.title('total tests') # Title

stats.groupby("state")['totaltestresults'].max().plot(kind='bar', color='steelblue')
stats=stats[['date', 'state','positive','negative','hospitalized', 'death']]

stats.head()
stats=stats.replace(to_replace ="WA",

                 value ="Washington")
stats=stats.replace(to_replace ="SC", 

                 value ="South Carolina")
stats=stats.replace(to_replace =("NJ","FL", 'AL', "TX", "OR"),

                 value =("New Jersey", "Florida", "Alabama", "Texas", "Oregon"))
stats=stats.replace(to_replace =("AR","AZ", "NY", "CA", "AK"),

                 value =("Arkansas", "Arizona", 'New York', "California", "Alaska"))
stats=stats.replace(to_replace =("MT","WI", "NC", 'OH',"RI", "VA"),

                 value =("Montana", "Wisconsin", 'North Carolina','Ohio', "Rhode Island", 'Virginia'))
stats=stats.replace(to_replace =("TN","GA", "IL", 'NH', "MA"),

                 value =("Tennessee", "Georgia", 'Illinios', "New Hampshire", "Massachussetts"))
stats=stats.replace(to_replace =("CO","CT", "DC", 'DE', "GU"),

                 value =("Colorado", "Connecticut", 'District of Columbia', "Delaware", "Guam"))
stats=stats.replace(to_replace =("HI","IA", "ID", 'IN', "KS", 'KY'),

                 value =("Hawaii", "Iowa", 'Idaho', "Indiana", "Kansas", "Kentucky"))
stats=stats.replace(to_replace =("LA","MD", "MN", 'MI', "MO", 'MS'),

                 value =("Louisiana", "Maryland", 'Minnessota', "Michigan", "Missouri", "Missippippi"))
stats=stats.replace(to_replace =("ME","NV", "WV", 'NM', 'PA', "VT"),

                 value =("Maine", "Nevada", 'West Virginia', "New Mexico", "Pennsylvania", "Vermont"))
stats=stats.replace(to_replace =("ND","OK", "UT", 'PR', 'SD'),

                 value =("North Dakota", "Oklahoma", 'Utah', "Puerto Rico", "South Dakota"))
stats=stats.replace(to_replace =("VI","WY", "NE"),

                 value =("Virgin Islands", "Wyoming", "Nebraska"))
stats.head(3)
stats.describe()
stats.corr().style.background_gradient(cmap='plasma')
# Merging the dataframes                       

a=pd.merge(USA, stats, how ='inner', on =('state', "date"))

a
dfs1=pd.concat([a,rank,chronic], sort=True) 

dfs1.head()
# Merging the dataframes                       

b=pd.concat([dfs1, Vuln], sort=False) 
# Merging the dataframes                       

c=pd.concat([b, census], sort=False) 

#replace NaNs with zeros in the df

c=c.fillna(0)

c.head()
# iterating the columns to list their names

for col in c.columns: 

    print(col)
# Grouped df by date and state and extract a number of stats from each group

d=c.groupby(

   ['date', 'state'], as_index = False

).agg(

    {

         'hospitalized':max,    # max values 

         'cases':max,

         'deaths': max,

         'num_uninsured':max, 

         'percent_vaccinated': max, 

         'num_uninsured': max,

         'percent_american_indian_alaska_native':max,        

         'percent_asian':max,

         'percent_black':max,        

        'percent_excessive_drinking':max,

        'percent_female':max,

        'percent_hispanic':max,

        'percent_native_hawaiian_other_pacific_islander':max,

        'percent_non_hispanic_white':max,

        'epl_pov':max,

        'epl_unemp': max,

        'epl_age65':max,

        'epl_age17':max,

        'epl_disabl':max,

        'high bp prev':max,

        'stroke prev':max,

        'obesity_crudeprev':max,

        'diabetes prev':max,

        'arthritis prev':max,

        'cancer prev':max,

        'asthma prev':max,

        'copd prev':max,

        'smoking prev':max,

        'highChol prev':max,

        'kidney prev':max

         

    }

)

d
sub1=d[d.date==0]

sub2=d[d.date!=0]
sub2=sub2[['state', 'cases', 'deaths', 'hospitalized']]

sub2.head()
# Merging the dataframes                       

risks=pd.merge(sub1, sub2, how ='inner', on ='state')

risks=risks.drop(['date'], axis = 1) 

sum_column = risks["hospitalized_x"] + risks["hospitalized_y"]

risks["hospitalized"] = sum_column

risks=risks.drop(['hospitalized_x','hospitalized_y'], axis = 1) 
sum_column2 = risks["cases_x"] + risks["cases_y"]

risks["cases"] = sum_column2

sum_column3 = risks["deaths_x"] + risks["deaths_y"]

risks["deaths"] = sum_column3
risks=risks.drop(['cases_x','cases_y', 'deaths_x','deaths_y'], axis = 1) 

risks
# Grouped df by date and state and extract a number of stats from each group

r=risks.groupby(

   ['state'], as_index = False).agg(    

    {

         'hospitalized':max,    # max values 

         'cases':max,

         'deaths': max,

         'num_uninsured':max, 

         'percent_vaccinated': max, 

         'num_uninsured': max,

         'percent_american_indian_alaska_native':max,        

         'percent_asian':max,

         'percent_black':max,        

        'percent_excessive_drinking':max,

        'percent_female':max,

        'percent_hispanic':max,

        'percent_native_hawaiian_other_pacific_islander':max,

        'percent_non_hispanic_white':max,

        'epl_pov':max,

        'epl_unemp': max,

        'epl_age65':max,

        'epl_age17':max,

        'epl_disabl':max,

        'high bp prev':max,

        'stroke prev':max,

        'obesity_crudeprev':max,

        'diabetes prev':max,

        'arthritis prev':max,

        'cancer prev':max,

        'asthma prev':max,

        'copd prev':max,

        'smoking prev':max,

        'highChol prev':max,

        'kidney prev':max

         

    }

)



r
r.describe()
r.corr().style.background_gradient(cmap='cubehelix')