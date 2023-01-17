# import pandas to work on dataframe

# matplotlib for visualization

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline

# Import these library to plot the world mapping

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
import warnings

warnings.filterwarnings("ignore")
Df = pd.read_csv("../input/ORCID_migrations_2016_12_16_by_person.csv")
Df.shape
Df.head()
Df.info()
# For this data the first ph.ds are received their ph.d in 1913

Df['phd_year'].min()
Df['phd_year'].max()
# Let us take look on the description of the two numerical features of the data

Df.describe()
Df.hist(bins=10,figsize=(15,8), alpha=0.5)

plt.show()
# I just look at the data where phd_year is not NaN. 

Df[Df['phd_year'].isnull()==False].head()
def cleanyear(phdYear,earliest):

    if (phdYear==phdYear) & (earliest!=earliest):

        return phdYear-4

    else:

        return earliest
Df['earliest_year'] = [ cleanyear(x,y) for (x,y) in  zip(Df['phd_year'], Df['earliest_year']) ]
Df[Df['phd_year'].isnull()==False].head()
def cleanyearv2(phdYear,earliest,hasPhd):

    if (phdYear!=phdYear) & (earliest==earliest) & (hasPhd==True) :

        return earliest+4

    else:

        return phdYear
Df['phd_year'] = [ cleanyearv2(x,y,z) for (x,y,z) in  zip(Df['phd_year'], Df['earliest_year'],Df['has_phd']) ]
Df.info()
# One can notice that after this cleaning operation, we get some phd_years greater than 2017.

Df[Df['phd_year']==2018].head(2)
def cleanyear2017(x):

    if x>2017:

        return 2017

    else:

        return x
Df['phd_year'] = [cleanyear2017(x) for x in Df['phd_year']]
Df[Df['phd_year'] > 2017]
Df[Df['has_phd']==False]['phd_year'].count()
Df['phd_year'].max()
# I look at the graduated cases

graduated = Df[Df['has_phd']==True]
graduated.head(10)
# Phds researchers proportion with respect to enrolled for phd.

proportion = Df[Df["has_phd"]==True]["orcid_id"].count()/Df["orcid_id"].count()
print("Percentage of graduated phds: {:.4}".format(proportion*100)+" %")

graduated.info()
plt.figure();

graduated.hist(bins=10,figsize=(15,8), alpha=0.5 )

plt.show()
MigrationOrNot = graduated.groupby('has_migrated').count()
MigrationOrNot
# I notice that percentage of phds who migrated during 1913 to 2017 is 25,6%

explode = (0, 0.1)

#print('Percentage of Ph.Ds who migrated during 1913 to 2017 is: {}'.format(8438900/(244902+84389))+' %')

MigrationOrNot['has_phd'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Phds')

plt.show()
PhdMigrated = graduated[graduated['has_migrated']== True]
PhdMigrated.head()
PhdMig = PhdMigrated.groupby('phd_year').count()
PhdMig.head()
PhdMig.iloc[20:30]
PhdMig.info()
PhdMig.iloc[60:]
# In this construction I drop the year 2017, because the data was collected partially and this fact could biase the analysis

X = list(PhdMig.index.values)[:64]
x_values = X

y_values = PhdMig['has_migrated'].iloc[:64]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs')

plt.show()
GraduatedYear = graduated.groupby('phd_year').count()
GraduatedYear.head(5)
GraduatedYearHasMigrated = GraduatedYear[GraduatedYear['has_migrated']==True]
GraduatedYearHasMigrated.hist(bins=10, figsize=(17,12), alpha=0.5 )

plt.show()
Df_it = Df[Df['phd_country']=='IT']
Df_it.head()
Df_it.info()
Df_it.groupby('has_migrated').count()
#I notice that just 20.1% of phd from Italia has migrated 

explode = (0.,0.1)

Df_it.groupby('has_migrated').count()['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Italian Phds')

plt.show()
Phd_it_migrated = Df_it[Df_it['has_migrated']==True ]
Phd_it_mig = Phd_it_migrated.groupby('phd_year').count()
Phd_it_mig.info()
Phd_it_mig.iloc[-6:-1]
#The pic of the number of phd migrated from Italy is year 2016

Phd_it_mig[Phd_it_mig['has_migrated']==max(Phd_it_mig['has_migrated']) ]
X = list(Phd_it_mig.index.values)[:-1]
x_values = X

y_values = Phd_it_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from Italia')

plt.show()
#France

Df_france = Df[Df['phd_country']=='FR']
Df_france.head()
Df_france.info()
Df_france.groupby('has_migrated').count()
#I notice that just 44.5% of phd from France has migrated 

explode = (0.,0.1)

Df_france.groupby('has_migrated').count()['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of French Phds')
Phd_france_migrated = Df_france[Df_france['has_migrated']==True ]
Phd_france_mig = Phd_france_migrated.groupby('phd_year').count()
Phd_france_mig.head()
Phd_france_mig.info()
Phd_france_mig[Phd_france_mig['has_migrated']==max(Phd_france_mig['has_migrated']) ]
X = list(Phd_france_mig.index.values)[:49]

x_values = X

y_values = Phd_france_mig['has_migrated'].iloc[:49]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from France')

plt.show()
#Great Bretain

Df_gb = Df[Df['phd_country']=='GB']
Df_gb.head()
Df_gb.info()
Df_gb.groupby('has_migrated').count()
#I notice that just 41.0% of phd from Canada has migrated 

explode = (0.,0.1)

Df_gb.groupby('has_migrated').count()['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Brithish Phds')
Phd_gb_migrated = Df_gb[Df_gb['has_migrated']==True ]
Phd_gb_mig = Phd_gb_migrated.groupby('phd_year').count()
Phd_gb_mig.head()
Phd_gb_mig[Phd_gb_mig['has_migrated']==max(Phd_gb_mig['has_migrated']) ]
X = list(Phd_gb_mig.index.values)[:-1]

x_values = X

y_values = Phd_gb_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from United Kingdom')

plt.show()
#USA

Df_usa = Df[Df['phd_country']=='US']
Df_usa.head()
Df_usa.info()
Df_usa.groupby('has_migrated').count()
#I notice that just 26.8% of phd from USA has migrated 

explode = (0.,0.1)

Df_usa.groupby('has_migrated').count()['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of USA Phds')

plt.show()
Phd_usa_migrated = Df_usa[Df_usa['has_migrated']==True ]
Phd_usa_mig = Phd_usa_migrated.groupby('phd_year').count()
Phd_usa_mig.head()
Phd_usa_mig[Phd_usa_mig['has_migrated']==max(Phd_usa_mig['has_migrated']) ]
X = list(Phd_usa_mig.index.values)[:-1]

x_values = X

y_values = Phd_usa_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from USA')

plt.show()
#Canada

Df_canada = Df[Df['phd_country']=='CA']
Df_canada.head()
Df_canada.info()
Df_canada.groupby('has_migrated').count()
#I notice that just 41.5% of phd from Canada has migrated 

explode = (0.,0.1)

Df_canada.groupby('has_migrated').count()['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Canadian Phds')

plt.show()
Phd_canada_migrated = Df_canada[Df_canada['has_migrated']==True ]
Phd_canada_mig = Phd_canada_migrated.groupby('phd_year').count()
Phd_canada_mig.head()
Phd_canada_mig[Phd_canada_mig['has_migrated']==max(Phd_canada_mig['has_migrated']) ]
X = list(Phd_canada_mig.index.values)[:-1]

x_values = X

y_values = Phd_canada_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from Canada')

plt.show()
Df_china = Df[Df['phd_country']=='CN']
Df1 = Df_china.groupby('has_migrated').count()
#I notice that just 17.5% of phd from China has migrated 

explode = (0.,0.1)

Df1['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of China Phds')

plt.show()
Phd_china_migrated = Df_china[Df_china['has_migrated']==True ]

Phd_china_mig = Phd_china_migrated.groupby('phd_year').count()
X = list(Phd_china_mig.index.values)[:-1]

x_values = X

y_values = Phd_china_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from China')

plt.show()
Df_india = Df[Df['phd_country']=='IN']
Df1 = Df_india.groupby('has_migrated').count()
#I notice that just 12.2% of phd from India has migrated 

explode = (0.,0.1)

Df1['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of India Phds')

plt.show()
Phd_india_migrated = Df_india[Df_india['has_migrated']==True ]

Phd_india_mig = Phd_india_migrated.groupby('phd_year').count()

X = list(Phd_india_mig.index.values)[:-1]

x_values = X

y_values = Phd_india_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from India')

plt.show()
Df_brazil = Df[Df['phd_country']=='BR']
Df1 = Df_brazil.groupby('has_migrated').count()
#I notice that just 9.0% of phd from Brazil has migrated 

explode = (0.,0.1)

Df1['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Brazil Phds')

plt.show()
Phd_brazil_migrated = Df_brazil[Df_brazil['has_migrated']==True ]

Phd_brazil_mig = Phd_brazil_migrated.groupby('phd_year').count()
X = list(Phd_brazil_mig.index.values)[:-1]

x_values = X

y_values = Phd_brazil_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from Brazil')

plt.show()
Df_australia = Df[Df['phd_country']=='AU']
Df1 = Df_australia.groupby('has_migrated').count()
#I notice that just 17.5% of phd from Australia has migrated 

explode = (0.,0.1)

Df1['orcid_id'].plot(figsize=(5,5), kind='pie', explode=explode, autopct='%1.1f%%')

plt.title('Migration of Australia Phds')

plt.show()
Phd_australia_migrated = Df_australia[Df_australia['has_migrated']==True ]

Phd_australia_mig = Phd_australia_migrated.groupby('phd_year').count()
X = list(Phd_australia_mig.index.values)[:-1]

x_values = X

y_values = Phd_australia_mig['has_migrated'].iloc[:-1]

plt.plot(x_values, y_values)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.ylabel('Phd Migrated')

plt.title('Migration of PhDs from Australia')

plt.show()
test = Df.groupby('phd_country').count()['orcid_id'].sort_values(ascending=False)

listCountry = test[:15].index

DfTest = Df.copy()
def otherCountry(x):

    if x in listCountry:

        return x

    elif x!= x:

        return 'Unknown'

    else:

        return 'Other'
DfTest['phd_country'] = [otherCountry(x) for x in Df['phd_country']]

DfTest.groupby('phd_country').count()['orcid_id'].plot(figsize=(8,8), kind='pie',  autopct='%1.1f%%')

plt.show()
DfKnownCountries = DfTest[DfTest['phd_country']!='Unknown'].groupby('phd_country').count()
DfKnownCountries.info()
DfKnownCountries
DfKnownCountries['orcid_id'].plot(figsize=(8,8), kind='pie')

plt.title('Phd researchers number with rescpect to contries')

plt.show()
Df_countries = DfKnownCountries['orcid_id'].sort_values(ascending=True)

Df_countries.plot(figsize=(10,10), kind='barh', color='blue', alpha = 0.3 )

plt.title('Number of Phd researchers with respect to their contries')



plt.show()
# I use the county codes to contruct the liste of country names

countries = ['Korea, Republic of', 'Japan', 'Germany', 'Sweden', 'Canada', 'France', 'Portugal', 'Brazil', 'Italy', 'Australia', 'China', 'India', 'Espain', 'United Kingdom', 'United States']
Nb_phds = list(Df_countries.iloc[:-1])
data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = Nb_phds,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Number of Phds')

            )

       ]



layout = dict(

    title = ' Number of Phds in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(41, 135, 202)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')