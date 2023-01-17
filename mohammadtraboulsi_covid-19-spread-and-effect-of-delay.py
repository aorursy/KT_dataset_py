import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white", font_scale=1.2)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

containment.head()

rename_col  = {'Country': 'country',

               'Date Start': 'date', 'Keywords':'keywords'}

cols_to_keep = list(rename_col.values())

containment = containment.rename(columns=rename_col)

containment = containment.drop(containment.columns.difference(cols_to_keep),axis=1)

containment['date'] = pd.to_datetime(containment['date'])

containment.dropna(subset=['country','date','keywords'],inplace=True)

containment.loc[containment.country.str.contains('US:'),'country'] = 'United States' #replace all states with just US

containment.head()
# add data for lebanon

leb_cont = pd.read_csv('/kaggle/input/lebanon-containtmentcsv/lebanon_containment.csv')

leb_cont['date']= pd.to_datetime(leb_cont['date'])

containment = pd.concat([containment,leb_cont],ignore_index=True)
containment.query('country == "Lebanon"').head()
covid= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

cols_rename = {'Country/Region':'country','ObservationDate':'date','Confirmed':'confirmed',

               'Deaths':'deaths','Recovered':'recovered'}

covid.rename(columns=cols_rename,inplace=True)

cols_keep = ['date','country','confirmed','deaths','recovered']

covid.drop(covid.columns.difference(cols_keep),axis=1,inplace=True)

covid.head()
pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

col_rename = {'Country (or dependency)': 'country','Population (2020)':'population',

              'Density (P/Km²)':'density','Med. Age':'median_age','Urban Pop %':'urban_pop'}

pop.rename(columns=col_rename,inplace=True)

cols_keep = col_rename.values()

pop.drop(pop.columns.difference(cols_keep),axis=1,inplace=True)

pop.head()
# rename some countries for merging

containment_country_names= {'Czechia':'Czech Republic','Kosovo':'Serbia','Vatican City': 'Italy','Macau':'Macao',

                            'Guernsey': 'Others','Jersey': 'Others'}

containment= containment.replace({'country':containment_country_names})

containment = containment.query('country != "Others"') # removing 'Others' as country
# merging containment and covid data but first we need to rename some countries

# long list of names since I collected data from 2 different sources

# found this list by doing a left merge then checking the nan values

country_map= {'Mainland China': 'China','US': 'United States','UK':'United Kingdom',

              'Republic of Ireland':'Ireland','North Ireland':'Ireland', 'The Bahamas': 'Bahamas',

              'Bahamas, The': 'Bahamas', 'Burma': 'Myanmar','Cape Verde': 'Cabo Verde','Macau':'Macao',

              'Ivory Coast': 'Côte d\'Ivoire', 'occupied Palestinian territory': 'Palestine',

              'West Bank and Gaza':'Palestine','Republic of the Congo':'Congo',

              'Saint Vincent and the Grenadines':'St. Vincent & Grenadines','The Gambia':'Gambia',

              'Gambia, The':'Gambia','Saint Kitts and Nevis': 'Saint Pierre & Miquelon',

              'Reunion': 'France','Kosovo':'Serbia','Curacao':'Curaçao', 'East Timor':'Timor-Leste',

              'Vatican City': 'Italy','Faroe Islands':'Faeroe Islands','Guernsey': 'Others',

              'Diamond Princess': 'Others','Jersey': 'Others','North Macedonia':'Macedonia', 'MS Zaandam':'Others'

             }

covid.loc[covid.country.str.contains('Brazzaville'),'country'] = 'Congo' # dataset contains different city names for congo

covid.loc[covid.country.str.contains('Kinshasa'),'country'] = 'Congo'

covid.loc[covid.country.str.contains('\)'),'country'] = 'St. Martin' # just for proper formatting

covid.country = covid.country.str.strip() #removing whitespace so it doesnt cause problems with merge

covid= covid.replace({'country':country_map}) # now replace all country names

covid = covid.groupby(['country','date'])[['confirmed','deaths','recovered']].sum() # make sure we add the cases in all countries

covid = covid.reset_index()

covid['date'] = pd.to_datetime(covid['date'])

covid.head()
# fix some data for Lebanon

covid.loc[(covid.country == "Lebanon") & (covid.confirmed == 110),'confirmed'],  covid.loc[(covid.country == "Lebanon") & (covid.confirmed == 99),'confirmed'] = 99 ,110

# just make names easier to work with

pop = pop.replace({'country':{'Czech Republic (Czechia)': 'Czech Republic','State of Palestine': 'Palestine',

                             'Saint Martin':'St. Martin', 'North Macedonia': 'Macedonia'

                             }})

pop['urban_pop']= pop['urban_pop'].str.replace('%','')

pop['urban_pop'] = pd.to_numeric(pop['urban_pop'],errors='coerce') /100

pop['median_age'] = pd.to_numeric(pop['median_age'],errors='coerce').fillna(0).astype('int')

pop[pop.median_age == 0 ].head() # many missing values but thats fine
countries = pd.merge(covid,pop,how='left',on='country') # doing a left merge to check that no country name is different

countries[countries.population.isna()].country.value_counts() # one last check
df = pd.merge(containment,countries,how='outer',on=['country','date'])
# concat duplicate values

# idea from https://stackoverflow.com/a/53463151/4145941

index_cols = df.columns.tolist()

index_cols.remove("keywords") 

df = df.groupby(index_cols)["keywords"].apply(list)

df = df.reset_index()

df.keywords = [','.join(map(str, l)) for l in df['keywords']]
df[df.confirmed.isna()]
# finally here is our data

df.head()
df.shape
df.info()
df.describe()
# plotting on a log scale because of the huge difference in the data

plt.figure(figsize=(18,5))

plt.hist(df.confirmed,log=True, bins=np.arange(0,120000,4000));

yticks = [1,2,5,10,20,50,100,200,500,1000,2000,5000]

plt.yticks(yticks,yticks);
plt.figure(figsize=(18,5))

plt.hist(df.deaths,log=True, bins=np.arange(0,10000,500));

yticks = [1,2,5,10,20,50,100,200,500,1000,2000,5000]

plt.yticks(yticks,yticks);
plt.figure(figsize=(18,5))

plt.hist(df.recovered,log=True, bins=np.arange(0,77000,4000));

yticks = [1,2,5,10,20,50,100,200,500,1000,2000,5000]

plt.yticks(yticks,yticks);
# in order for this to work we need to install an additional library

# !pip install wordcloud

# idea from https://www.datacamp.com/community/tutorials/wordcloud-python



from wordcloud import WordCloud

text = ' '.join(df.keywords).replace('nan','')

wordcloud = WordCloud(min_font_size= 5,max_font_size=100).generate(text)

plt.figure(figsize=[18,8])

plt.title('Containment Measures')

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off");

plt.show()
# increase in total confirmed cases

# again, using log scale because of the exponential growth

plt.figure(figsize=(18,10))

np.log10(df.groupby('date')['confirmed'].sum()).plot()

yticks = [200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000]

plt.yticks(np.log10(yticks),yticks)

plt.title('Total Infections through time (log scale)');
# similarly lets take a look at the other variables with time

plt.figure(figsize=(18,10))

np.log10(df.groupby('date')['deaths'].sum()).plot()

yticks = [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000]

plt.yticks(np.log10(yticks),yticks);

plt.title('Total Confirmed Deaths with Time (log scale)');
plt.figure(figsize=(18,10))

np.log10(df.groupby('date')['recovered'].sum()).plot()

yticks = [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000]

plt.yticks(np.log10(yticks),yticks);

plt.title('Total Confirmed Recoveries with Time (log scale)');
# confirmed cases across countries

data = df.query('date == "2020-03-27"') # pick the most recent date

data = np.log10(data.groupby('country')['confirmed'].sum().sort_values(ascending=False)) # sorting countries and using log scale

yticks= [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000]

plt.figure(figsize=(25,7))

data.plot(kind='bar')

plt.yticks(np.log10(yticks),yticks);
# now lets take a look at the correlation between all variables

plt.figure(figsize=(14,5))

sns.heatmap(df.corr(),annot = True);
# lets first take a look at the relationship between confirmed cases and number of deaths

# lets also not forget to use a log scale for both values

# since some of these values are 0 we have to make sure to remove these values

df2 = df #.query('date == "2020-03-27"') # we can choose a single date to see the data more clearly

df2 = df2[(df2.confirmed != 0) & (df2.deaths != 0)]

x = np.log10(df2.confirmed)

y= np.log10(df2.deaths)

plt.figure(figsize=(18,10))

xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

yticks= xticks[:-2]

sns.regplot(x,y,scatter_kws={'alpha':1/5});

plt.xticks(np.log10(xticks),xticks,rotation=90)

plt.yticks(np.log10(yticks),yticks);
# Now lets look at the confirmed cases vs population

df2 = df.query("date == '2020-03-27'") # lets take a look at just 1 day to avoid duplicate values

df2 = df2[(df2.confirmed != 0) & (df2.population !=0)]

x = np.log10(df2.confirmed)

y = np.log10(df2.population)

plt.figure(figsize=(18,10))

xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

plt.xticks(np.log10(xticks),xticks,rotation=90)

yticks = [5000,10000,100000,1000000,int(3e6),int(1e7),int(3e7),int(1e8),int(3e8),int(1e9),int(3e9)]

plt.yticks(np.log10(yticks),yticks)

sns.regplot(x,y,scatter_kws={'alpha':1/2});
# Lets take a look to see if the density of a country is related

df2 = df.query("date == '2020-03-27'") # also last date so we can filter by country

df2 = df2[(df2.confirmed != 0) & (df2.density !=0)]

x = np.log10(df2.confirmed)

y = df2.density # no need to use log here

plt.figure(figsize=(18,10))

xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

plt.xticks(np.log10(xticks),xticks,rotation=90) # same xticks as above

#yticks = [5000,10000,100000,1000000,int(3e6),int(1e7),int(3e7),int(1e8),int(3e8),int(1e9),int(3e9)]

#plt.yticks(np.log10(yticks),yticks)

plt.ylim(0,500) # zooming in

sns.regplot(x,y,scatter_kws={'alpha':1/2});
# Now we take a look at median age vs deaths

df2 = df.query("date == '2020-03-27'")

df2 = df2[(df2.deaths != 0) & (df2.median_age !=0)]

x = np.log10(df2.deaths)

y= df2.median_age

plt.figure(figsize=(18,10))

xticks = [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,0.6]

plt.xticks(np.log10(xticks),xticks)

sns.regplot(x,y,scatter_kws={'alpha':0.4});
# one final thought: are people in the cities more suscptible to the virus

df2 = df.query("date == '2020-03-27'")

df2 = df2[(df2.confirmed != 0) & (df2.urban_pop !=0)]

x = np.log10(df2.confirmed)

y= df2.urban_pop

plt.figure(figsize=(18,10))

xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

plt.xticks(np.log10(xticks),xticks,rotation=90)

sns.regplot(x,y,scatter_kws={'alpha':0.4});
# add a new column called 'new cases'

# this will count the increase in new cases (per day) instead of the total cumulative value

df2 = df.groupby(['country','date'])['confirmed'].sum().diff()

df2 = df2.reset_index()

df2.loc[df2.confirmed<0,'confirmed'] =0 

df2= df2.fillna(0)

df['new_cases'] = df2['confirmed']
def plot_on_date(interest_date,interest_countries):

    '''

    interest_date: the date we are interested in (yyyy-mm-dd)

    interest_countries: a list of countries we are interested in.

    '''

    df2 = df.query(('date == @interest_date'))

    df2 = df2[(df2.confirmed != 0) & (df2.new_cases !=0)]

    x = np.log10(df2.confirmed)

    y = np.log10(df2.new_cases)

    xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

    plt.xticks(np.log10(xticks),xticks)

    yticks =  [1,3,10,30,100,300,1000,3000,10000,30000]

    plt.yticks(np.log10(yticks),yticks)

    ax = sns.regplot(x,y,scatter_kws={'alpha':0.5},fit_reg=False);

    plt.title(interest_date)

    for i in interest_countries:

        subset = df2.query('country == @i')

        x = np.log10(subset.confirmed) - 0.1

        y = np.log10(subset.new_cases) + 0.1

        ax.text(x,y,i)
interest_countries = ['United States', 'Italy','China','Lebanon','South Korea', 'United Kingdom','Spain','Egypt','Qatar','Netherlands','Turkey']

fig, (ax1,ax2) = plt.subplots(1,2)

fig.set_figheight(10)

fig.set_figwidth(20)



plt.subplot(121)

plot_on_date('2020-03-25',interest_countries)



plt.subplot(122)

plot_on_date(str(df.date.max().date()),interest_countries)

# get first case for each country

keywords = ['closure school','university closure','closure university','business suspension']

last_date = str(df.date.max().date())

df2= df.query('date == @last_date')

test_df = df[df.confirmed >  1].groupby('country',as_index=False).date.min().rename(columns={'date': 'date_first_case'})

test2_df = df[df.keywords.str.contains('|'.join(keywords))].groupby('country',as_index=False).date.min().rename(columns={'date': 'date_first_measure'})

data = pd.merge(test_df,test2_df,how='inner',on='country')

data['delay'] = (data['date_first_measure'] - data['date_first_case']).dt.days

df3 = pd.merge(data,df2,how='inner',on='country')
df3 = df3.query('new_cases > 0')

x = np.log10(df3.confirmed)

y = np.log10(df3.new_cases)

z = df3.delay

plt.figure(figsize=(15,8))

xticks = [1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]

plt.xticks(np.log10(xticks),xticks)

yticks =  [1,3,10,30,100,300,1000,3000,10000,30000]

plt.yticks(np.log10(yticks),yticks)

ax = sns.scatterplot(x,y,hue=z,palette='RdBu_r');

plt.title('Effect of delay on country performance ({})'.format(last_date));
# finally saving

df.to_csv('./final_dataset.csv',index=False)