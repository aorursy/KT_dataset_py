#  Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  maps
import folium
from folium.plugins import MarkerCluster


%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)

#  Kaggle directories
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df  = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")  # suicides
gps = pd.read_csv("../input/world-capitals-gps/concap.csv")   # world GPS
# check df against gps
count = 0
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))
        count = count + 1
print('check complete:  {} missing'.format(count)) 

#  update names in df to match the gps file
df.replace({'Cabo Verde':'Cape Verde','Republic of Korea':'South Korea','Russian Federation':'Russia','Saint Vincent and Grenadines':'Saint Vincent and the Grenadines'},inplace=True)
# check df against gps
count = 0
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))
        count = count + 1
print('check complete:  {} missing'.format(count))        
df = df.join(gps.set_index('CountryName'), on='country')
df = df.drop(['HDI for year','country-year','CountryCode','CapitalName'], axis=1)
df.info()
#  Top 10 most populous countries in the world
top10 = ['China','India','United States','Indonesia','Brazil','Pakistan','Nigeria','Bangladesh','Russia','Mexico']
in_set = df.country[df.country.str.contains('|'.join(top10))].unique().tolist()

print('Out of the top 10 most populous countries: \n{}\n\nonly the following {} are present:\n{}'.format(top10,len(in_set),in_set))

#  dataset
print('\n\nDataset has', len(df['country'].unique()),'countries (out of 195) on' ,len(df['ContinentName'].unique()),'continents spanning' ,len(df['year'].unique()),'years.')
print(df.info())         #  dataset size and types
print('\nDATA SHAPES:  {}'.format(df.shape))
df.describe(include=['O'])   #  CATEGORICAL DATA
df.describe()   #  NUMERIC DATA
nulls = df.isnull().sum().sort_values(ascending = False)
prcet = round(nulls/len(df)*100,2)
df.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])

print('List of NULL rows\n{}'.format(df.null))
print('\nDUPLICATED rows:\t{}'.format(df.duplicated().sum()))

plt.title('NULLs heatmap')
sns.heatmap(df.isnull())
suicideRate = df['suicides/100k pop'].groupby(df['country']).mean().sort_values(ascending=False).reset_index()
suicideMean = suicideRate['suicides/100k pop'].mean()

plt.figure(figsize=(8,20))
plt.title('Suicide Rates per Country (mean={:.2f})'.format(suicideMean), fontsize=14)
plt.axvline(x=suicideMean,color='gray',ls='--')
sns.barplot(data=suicideRate, y='country',x='suicides/100k pop')

suicideRate.head(10)
YRS = sorted(df.year.unique()-1)  # not including 2016 data
POP = []    # population
GDC = []    # gdp_per_capita ($)
SUI = []    # suicides_no
SUR = []    # suicides/100k pop

for year in sorted(YRS):
    POP.append(df[df['year']==year]['population'].sum())
    GDC.append(df[df['year']==year]['gdp_per_capita ($)'].sum())
    SUI.append(df[df['year']==year]['suicides_no'].sum())
    SUR.append(df[df['year']==year]['suicides/100k pop'].sum())

#  plot population and gdp_per_capita ($), 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Population vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.axis('auto')
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,POP)
fig.add_subplot(122)
plt.title('GDP per Capita vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('GDP per Capita (in $)', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,GDC)
plt.show()
#  plot suicides_no and suicides/100k pop, 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Suicides vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUI)
fig.add_subplot(122)
plt.title('Suicides per 100k vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides/100k Population', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUR)
plt.show()
ageList = sorted(df.age.unique())
ageList.remove('5-14 years')
fig = plt.figure(figsize=(12,5))

for i in ageList:
    fig.add_subplot(121)
    plt.title('Suicide Rates per Age Group', fontsize=14)
    plt.xlabel('suicides/100k pop', fontsize=12)
    plt.xlim(0,50)
    plt.legend(ageList)
    df['suicides/100k pop'][df['age'] == i].plot(kind='kde')

    fig.add_subplot(122)
    plt.title('Suicide Rates vs GDP', fontsize=14)
    plt.xlabel('gdp_per_capita ($)', fontsize=12)
    plt.yticks([], [])
    plt.xlim(0,100000)
    #df['gdp_per_capita ($)'][df['age'] == i].plot(kind='kde')
    df['gdp_per_capita ($)'].plot(kind='kde')
fig = plt.figure(figsize=(10,6))
plt.title('Male/Female Suicides/100k per Continents', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data =df, x='sex',y='suicides/100k pop', hue='ContinentName',palette='Blues_r')
fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.title('Male/Female Suicides/100k vs Age', fontsize=14)
plt.xlabel('sex', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='age')
fig.add_subplot(122)
plt.title('Male/Female Suicides/100k vs Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='generation')
plt.show()
df_sort =  df.sort_values(by='age')  # sort by age
plt.figure(figsize=(10,8))
plt.title('Suicide Trend', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort,x='year',y='suicides/100k pop',hue='age',ci=None)
fig = plt.figure(figsize=(14,6))
fig.add_subplot(121)
plt.title('Suicide Trend - MALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'male'], x='year',y='suicides/100k pop',hue='age',ci=None)
fig.add_subplot(122)
plt.title('Suicide Trend - FEMALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'female'], x='year',y='suicides/100k pop',hue='age',ci=None)
plt.show()
df.columns
#  1.  drop columns not needed for correlation
df_corr = df.drop(['country','year','CapitalLatitude','CapitalLongitude'], axis=1)

#  2.  rearrange column names
df_corr = df_corr[['suicides/100k pop', 'sex', 'age', 'population',' gdp_for_year ($) ','gdp_per_capita ($)', 'generation','suicides_no','ContinentName']]

#  3.  encode
df_corr['sex'] = df_corr['sex'].map({'female':0,'male':1})
df_corr['age'] = df_corr['age'].map({
        '5-14 years':0,'15-24 years':1,'25-34 years':2,
        '35-54 years':3,'55-74 years':4,'75+ years':5})
df_corr['generation'] = df_corr['generation'].map({
        'Generation Z':0,'Millenials':1,'Generation X':2,
        'Boomers':3,'Silent':4,'G.I. Generation':5})
df_corr['ContinentName'] = df_corr['ContinentName'].map({
        'Africa':0,'Asia':1,'Australia':2,'Central America':3,
        'Europe':4,'North America':5,'South America':6})

#  remove commas and save as float64
df_corr[' gdp_for_year ($) '] = df_corr[' gdp_for_year ($) '].str.replace(',','').astype('float64')

#df_corr.describe(include=['O'])   #  CATEGORICAL DATA
df_corr.info()
from sklearn.preprocessing import MinMaxScaler
df_norm = MinMaxScaler().fit_transform(df_corr)
df_C = pd.DataFrame(df_norm, index=df_corr.index, columns=df_corr.columns)
dataCorr = df_C.corr()
plt.figure(figsize=(8,8))
plt.title('Suicide Correlation', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')

dataCorr['suicides/100k pop'].sort_values(ascending=False)
#  Correlation MALE - filter dataframe for male/female
dataMale   = df_C[(df_C['sex'] == 1)]                       # male
dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr
corrM = dataMaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation FEMALE - filter dataframe for male/female
dataFemale = df_C[(df_C['sex'] == 0)]                       # female
dataFemaleCorr = dataFemale.drop(["sex"], axis=1).corr()    # female corr
corrF = dataFemaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation heatmaps for FEMALE/MALE
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
plt.title('Suicide Correlation - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
fig.add_subplot(122)
plt.title('Suicide Correlation - FEMALE ', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
plt.show()
#  create dataframe with Country and mean of Suicide rates per 100k Population
df_choro = df[['suicides/100k pop','country']].groupby(['country']).mean().sort_values(by='suicides/100k pop').reset_index()

#  Update US name to match JSON file
df_choro.replace({'United States':'United States of America'},inplace=True)

#  https://www.kaggle.com/ktochylin/world-countries
world_geo = r'../input/world-countries/world-countries.json'
world_choropelth = folium.Map(location=[0, 0], tiles='cartodbpositron',zoom_start=1)

world_choropelth.choropleth(
    geo_data=world_geo,
    data=df_choro,
    columns=['country','suicides/100k pop'],
    key_on='feature.properties.name',
    fill_color='PuBu',  # YlGn
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Suicide Rates per 100k Population')

 
# display map
world_choropelth
#  create dataframe for mapping
mapdf = pd.DataFrame(columns =  ['country','suicides_no','lat','lon'])

mapdf.lat = mapdf.lat.astype(float).fillna(0.0)
mapdf.lon = mapdf.lat.astype(float).fillna(0.0)

mapdf['country']     = df['suicides_no'].groupby(df['country']).sum().index
mapdf['suicides_no'] = df['suicides_no'].groupby(df['country']).sum().values
for i in range(len(mapdf.country)):
    mapdf.lat[i] =  df.CapitalLatitude[(df['country'] == mapdf.country[i])].unique()
    mapdf.lon[i] = df.CapitalLongitude[(df['country'] == mapdf.country[i])].unique()


#  make map - popup displays country and suicide count
#  lat/lon must be "float"
world_map = folium.Map(location=[mapdf.lat.mean(),mapdf.lon.mean()],zoom_start=2)
marker_cluster = MarkerCluster().add_to(world_map)

for i in range(len(mapdf)-1):
    label = '{}:  {} suicides'.format(mapdf.country[i].upper(),mapdf.suicides_no[i])
    label = folium.Popup(label, parse_html=True)
    folium.Marker(location=[mapdf.lat[i],mapdf.lon[i]],
            popup = label,
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)


world_map.add_child(marker_cluster)
world_map         #  display map