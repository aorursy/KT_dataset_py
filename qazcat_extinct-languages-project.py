import pandas as pd

import numpy as np



language = pd.read_csv('../input/data.csv')

language.head(1)

language.info()
#Set number of speakers null to 0

language['Number of speakers'].fillna(0,inplace=True)
#Database may already be sorted by number of speakers

number_of_speakers = language[['Name in English','Number of speakers']].sort_values('Number of speakers',ascending=False)

print('The Languages from the dataset with the most number of speakers are:')

print(number_of_speakers.head(10))
print("The number of languages from each category of endangerment are:")

print(language['Degree of endangerment'].value_counts())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style('white')

sns.countplot(language['Degree of endangerment'])

sns.despine()

plt.tight_layout()
#Preparing dicitonary for new dataframe that shows how many languages are in each country

degree_by_region = {}

country_codes_dict = {}



def cats_by_region(row):

    countries = row['Countries'].split(',')

    degree = row['Degree of endangerment']

    for raw_country in countries:

        country = raw_country.lstrip()

        if country in degree_by_region:            

            if degree in degree_by_region[country]:

                degree_by_region[country][degree] += 1

            else:

                degree_by_region[country][degree] = 1

        else:

            degree_by_region[country] ={degree:1}

    

language[['Countries','Degree of endangerment']].dropna().apply(cats_by_region, axis=1)



#Creating dicitonary of country codes for new dataset

def extract_codes(row):

    codes = row['Country codes alpha 3'].split(',')

    countries = row['Countries'].split(',')

    

    for i,raw_country in enumerate(countries):

        country = raw_country.lstrip()

        if not(country in country_codes_dict):

                country_codes_dict[country] = codes[i].lstrip()

            

language[['Countries','Country codes alpha 3']].dropna().apply(extract_codes, axis=1)

print('Done')
#Create new dataframe using data from dictionary

degrees_df = pd.DataFrame(degree_by_region).transpose()

degrees_df.fillna(0, inplace = True)



#Creating Totals

degrees_df['Total'] = degrees_df.apply(sum,axis=1)



#Changing index to allow easier joining onto the next dataset

degrees_df.reset_index(inplace=True)

degrees_df['Country Code'] = degrees_df.reset_index()['index'].apply(lambda s:country_codes_dict[s])

degrees_df.set_index('Country Code', inplace = True)



degrees_df.head(5)
#From Wikipedia

continents = {'Africa':'DZA,AGO,BWA,BDI,CMR,CPV,CAF,TCD,COM,MYT,COG,COD,BEN,GNQ,ETH,ERI,DJI,GAB,GMB,GHA,GIN,CIV,KEN,LSO,LBR,LBY,MDG,MWI,MLI,MRT,MUS,MAR,MOZ,NAM,NER,NGA,GNB,REU,RWA,SHN,STP,SEN,SYC,SLE,SOM,ZAF,ZWE,SSD,ESH,SDN,SWZ,TGO,TUN,UGA,EGY,TZA,BFA,ZMB'.split(','),

              'Asia':'AFG,AZE,BHR,BGD,ARM,BTN,IOT,BRN,MMR,KHM,LKA,CHN,TWN,CXR,CCK,CYP,GEO,PSE,HKG,IND,IDN,IRN,IRQ,ISR,JPN,KAZ,JOR,PRK,KOR,KWT,KGZ,LAO,LBN,MAC,MYS,MDV,MNG,OMN,NPL,PAK,PHL,TLS,QAT,RUS,SAU,SGP,VNM,SYR,TJK,THA,ARE,TUR,TKM,UZB,YEM'.split(','),

              'Europe':'ALB,AND,AZE,AUT,ARM,BEL,BIH,BGR,BLR,HRV,CYP,CZE,DNK,EST,FRO,FIN,ALA,FRA,GEO,DEU,GIB,GRC,VAT,HUN,ISL,IRL,ITA,KAZ,LVA,LIE,LTU,LUX,MLT,MCO,MDA,MNE,NLD,NOR,POL,PRT,ROU,RUS,SMR,SRB,SVK,SVN,ESP,SJM,SWE,CHE,TUR,UKR,MKD,GBR,GGY,JEY,IMN'.split(','),

              'North America':'ATG,BHS,BRB,BMU,BLZ,VGB,CAN,CYM,CRI,CUB,DMA,DOM,SLV,GRL,GRD,GLP,GTM,HTI,HND,JAM,MTQ,MEX,MSR,ANT,CUW,ABW,SXM,BES,NIC,UMI,PAN,PRI,BLM,KNA,AIA,LCA,MAF,SPM,VCT,TTO,TCA,USA,VIR'.split(','),

              'Oceania':'ASM,AUS,SLB,COK,FJI,PYF,KIR,GUM,NRU,NCL,VUT,NZL,NIU,NFK,MNP,UMI,FSM,MHL,PLW,PNG,PCN,TKL,TON,TUV,WLF,WSM'.split(','),

              'South America':'ARG,BOL,BRA,CHL,COL,ECU,FLK,GUF,GUY,PRY,PER,SUR,URY,VEN'.split(',')}



continent_dict = {}



for continent in continents:

    for code in continents[continent]:

        continent_dict[code] = continent



continent_df = pd.DataFrame.from_records([continent_dict]).transpose()

continent_df.columns = ['Continent']



continent_df.head()
#Combine dataframes to add continents

degrees_df = degrees_df.join(continent_df)

degrees_df.reset_index(inplace=True)

degrees_df.set_index('index', inplace =True, drop=False)

degrees_df.head(5)
degrees_df[degrees_df['Continent'].isnull()]
degrees_df.set_value('Angola','Country Code', 'AGO')

degrees_df.set_value('Democratic Republic of the Congo','Country Code', 'COD')



degrees_df.set_value('Angola','Continent', 'Africa')

degrees_df.set_value('Democratic Republic of the Congo','Continent', 'Africa')



degrees_df.head(5)
#Create choropleth map

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)



data = dict(type='choropleth',

locations = degrees_df['Country Code'], z = degrees_df['Total'],

text = degrees_df['index'], colorbar = {'title':'Total Languages'},

colorscale = 'Viridis', reversescale = True)



layout = dict(title='Total Langauages By Country',

geo = dict(showframe=False,projection={'type':'Mercator'}))



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
#Extinct

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)



data = dict(type='choropleth',

locations = degrees_df['Country Code'], z = degrees_df['Extinct'],

text = degrees_df['index'], colorbar = {'title':'Number of Extinct Languages'},

colorscale = 'Viridis', reversescale = True)



layout = dict(title='Total Extinct Languages By Country',

geo = dict(showframe=False,projection={'type':'Mercator'}))



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)