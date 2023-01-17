# Importing necessary Python packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

pd.set_option('display.max_column',None)
investments = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding = "ISO-8859-1")
investments.head()



# for a quick glance at the material's first five rows.
investments.info()



# using the Pandas info function, we get a summary of the dataframe's contents, 

# eg. dtypes, columns and their names, entries and non-null values.

investments.describe()

# for some statistics on the dataframe's non-null entries (49k+) as well as its the mean, min, max values etc. - how the data is distributed. 
# Choosing the most suitable columns for our study

investments = investments[['name', ' market ',

       ' funding_total_usd ', 'status', 'country_code', 'region',

       'city', 'funding_rounds', 'founded_at','first_funding_at',

       'last_funding_at', 'seed', 'venture', 'equity_crowdfunding',

       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',

       'private_equity', 'post_ipo_equity', 'post_ipo_debt',

       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',

       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']]
# Further limiting our dataset to only nordic countries and setting the index from the default to 'country_code'

investments = investments.set_index('country_code')
# Using the Pandas loc function to access specific labelled rows and columns 

investments = investments.loc[['FIN', 'SWE', 'NOR', 'DNK', 'ISL'], :]
# For a random sample from the selected subset



investments.sample(10)
investments['round_F'].unique()

# We were interested in seeing whether there is much - or any - useful data in round_F - or, actually, after round_C
investments.head()
investments.reset_index()
#cleaning column names and reformatting the data

#Column names had some empty spaces

investments.columns = investments.columns.str.replace(' ','')
#removed the commas from the total funding

investments.funding_total_usd = investments.funding_total_usd.str.replace(',','')
#it was string, now it is a float.

investments.funding_total_usd = pd.to_numeric(investments.funding_total_usd, errors='coerce')
# changing the data type for column founded_at to datetime type

investments['founded_at'] = pd.to_datetime(investments['founded_at'], errors = 'coerce' )
# changing the data type for column first_funding_at to datetime type

investments.first_funding_at = pd.to_datetime(investments.first_funding_at, format='%Y/%m/%d', errors='coerce')
# changing the data type for column first_funding_at to datetime type

investments.first_funding_at = pd.to_datetime(investments.first_funding_at, format='%Y/%m/%d', errors='coerce')
# Now we can groupby country code.

investments.groupby('country_code').mean()
# deleting the columns round_G and round_H because all the countries have zero values

investments = investments.drop(["round_G", "round_H"], axis=1)
# rounding the mean values to 4 decimal places

investments.groupby('country_code').mean().round(4)
# rounding the values of the data frame to 4 decimal places

investments= investments.round(4)

investments
# We discuss whether to keep the region column instead of the city column, 

# as 'region' is clearer and less complex when we run data analysis



investments['region'].unique()
# So, our option would be to drop the city column as it DOES ADD lots of complex data when we run data analysis



investments['city'].unique()
# We decide to go with dropping the city column for clarity's sake.

investments = investments.drop(["city"], axis=1)
investments.head()
plt.subplots(figsize=(20,15))



sns.heatmap(investments.corr(), annot=True, linewidth=0.5);



# This heat map serves to give us a bigger picture of our investments data. 
# Ungroup dataset



investments = investments.reset_index(level='country_code')
# Startup etablishment for the past 20 years in Nordic countries (NC)



plt.rcParams['figure.figsize'] = 15,6

investments['name'].groupby(investments["founded_at"].dt.year).count().tail(20).plot(kind="bar")



ax = plt.axes()        

ax.yaxis.grid()

plt.ylabel('Count')

plt.title("Founded distribution ", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
# The total number of startups in each Nordic country



investments['country_code'].value_counts()
# The total number of startups in each Nordic country



plt.figure(figsize=(10,5))



sns.barplot(x=investments['country_code'].value_counts(), y=investments['country_code'].value_counts().index, palette='Reds_d')



ax = plt.axes()        

ax.xaxis.grid()

plt.xlabel('Number of startups')

plt.ylabel('Nordic Countries')

plt.show()
# Top 10 startup status based on market sector



operating = investments[investments.status == 'operating']

acquired = investments[investments.status == 'acquired']

closed = investments[investments.status == 'closed']
operating_count  = operating['market'].value_counts()

operating_count = operating_count[:10,]



print('Operating')

print(operating_count)
acquired_count  = acquired['market'].value_counts()

acquired_count = acquired_count[:10,]



print('Acquired')

print(acquired_count)
closed_count  = closed['market'].value_counts()

closed_count = closed_count[:10,]



print('Closed')

print(closed_count)
# Startup status for each Nordic country



startup = investments.groupby('country_code').status.value_counts()



startup
# Startup status for each Nordic country for bar plot

startup1 = investments.groupby('country_code').status.value_counts().reset_index(name='counts')

sns.catplot(x="country_code", y="counts", hue="status", kind="bar", data=startup1, height=8.27, aspect=11.7/8.27)
# Relationship between operating, acquired and closed startups in each Nordic country - normalised to a 100 percent bar chart 



import matplotlib.ticker as mtick



investments.groupby(['country_code','status']).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(bbox_to_anchor=(1,1), frameon = True, fancybox = True, framealpha = 0.95, shadow = True, 

           borderpad = 1)

plt.show()
# Status of Nordic startups in donut form



chart = startup.plot(kind="pie", figsize=(20,15), autopct="%1.0f%%", rotatelabels=True )

cen_cir = plt.Circle((0,0),0.70, fc='w')

plt.gcf().gca().add_artist(cen_cir)



chart.set_ylabel('')

plt.title("Status of nordic startups", loc='left')

plt.show()
# Most funded startups/companies in Nordic countries



most_funded = investments.nlargest(20, ['funding_total_usd'])

most_funded
Spotify_founded_year = investments['founded_at'][investments['name']=="Spotify"].dt.year.values[0]

Symphogen_founded_year  = investments['founded_at'][investments['name']=="Symphogen"].dt.year.values[0]

Klarna_founded_year = investments['founded_at'][investments['name']=="Klarna"].dt.year.values[0]

Supercell_founded_year  = investments['founded_at'][investments['name']=="Supercell"].dt.year.values[0]

Rovio_founded_year  = investments['founded_at'][investments['name']=="Rovio Entertainment"].dt.year.values[0]
# Comparison of founding dates selected, successful Nordic startups



plt.rcParams['figure.figsize'] = 15,6

investments['name'][investments["founded_at"].dt.year >= 1995].groupby(investments["founded_at"].dt.year).count().plot(kind="line")

plt.ylabel('Count')



plt.axvline(Spotify_founded_year,color='blue',linestyle ="--")

plt.text(Spotify_founded_year+0.15, 50,"Spotify \n (2006)")



plt.axvline(Symphogen_founded_year,color='black',linestyle ="--")

plt.text(Symphogen_founded_year+0.15, 20,"Symphogen \n(2000)")



plt.axvline(Klarna_founded_year,color='orange',linestyle ="--")

plt.text(Klarna_founded_year-1.00, 35,"Klarna \n(2005)")



plt.axvline(Supercell_founded_year,color='red',linestyle ="--")

plt.text(Supercell_founded_year-1.30, 70,"Supercell \n(2010)")



plt.axvline(Rovio_founded_year,color='grey',linestyle ="--")

plt.text(Rovio_founded_year-1.30, 30,"Rovio \nEntertainment \n(2003)")



plt.title("When were the well-known companies founded?", fontdict=None, position= [0.48,1.05])

plt.show()
# Total number of Nordic startups per market/industry with more than 1 million USD investment



most_high= investments[['market', 'name']][investments['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)

most_high.head(20)

top20 = most_high.head(20)

top20
Nordm = sns.catplot(x="market", y="name",  kind="bar", data=top20, height=5.27, aspect=11.7/5.27)

Nordm.set_xticklabels(rotation=45, horizontalalignment='right')
# Total number of startups per Finnish market with more than 1 million USD investment



fin_high= investments[investments['country_code'] == 'FIN']



finh = fin_high[['market', 'name']][fin_high['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)



Finl10=finh.head(10)

Finl10
Finm = sns.catplot(x="market", y="name",  kind="bar", data=Finl10, height=4.27, aspect=8.7/4.27,palette=sns.dark_palette("green"))

Finm.set_xticklabels(rotation=45, horizontalalignment='right')
# Total number of startups per Swedish market with more than 1 million USD investment



swe_high= investments[investments['country_code'] == 'SWE']



sweh = swe_high[['market', 'name']][swe_high['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)



SWL10=sweh.head(10)

SWL10

SWm = sns.catplot(x="market", y="name",  kind="bar", data=SWL10, height=4.27, aspect=8.7/4.27,palette=sns.dark_palette("red"))



SWm.set_xticklabels(rotation=45, horizontalalignment='right')
# Total number of startups per Norwegian market with more than 1 million USD investment



nor_high= investments[investments['country_code'] == 'NOR']



norh = nor_high[['market', 'name']][nor_high['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)



Nor10 =norh.head(10)

Nor10
NRm = sns.catplot(x="market", y="name",  kind="bar", data=Nor10, height=4.27, aspect=8.7/4.27,palette=sns.dark_palette("navy", reverse=True))



NRm.set_xticklabels(rotation=45, horizontalalignment='right')
# Total number of startups per Danish market with more than 1 million USD investment



dnk_high= investments[investments['country_code'] == 'DNK']



dnkh = dnk_high[['market', 'name']][dnk_high['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)



DN10=dnkh.head(10)

DN10
DNm = sns.catplot(x="market", y="name",  kind="bar", data=DN10, height=4.27, aspect=8.7/4.27,palette=sns.diverging_palette(255, 133, l=60, n=7, center="dark"))



DNm.set_xticklabels(rotation=45, horizontalalignment='right')
# Total number of startups per Icelandic market with more than 1 million USD investment



isl_high= investments[investments['country_code'] == 'ISL']



islh = isl_high[['market', 'name']][isl_high['funding_total_usd'] > 1000000].groupby(['market'], 

                                        as_index=False).count().sort_values('name', ascending=False)



IS10=islh.head(10)

IS10
ISm = sns.catplot(x="market", y="name",  kind="bar", data=IS10, height=4.27, aspect=6.7/8.27,palette=sns.color_palette("BrBG", 7))



ISm.set_xticklabels(rotation=45, horizontalalignment='right')
# 833 largest  debt_financing in the Nordic startups 





LDF = investments.nlargest(833,'debt_financing')

ax = LDF.country_code.value_counts().plot(kind='pie',autopct='%.2f%%',figsize=(12,12))

add_circle = plt.Circle((0,0),0.7,color='white')

fig=plt.gcf()

fig.gca().add_artist(add_circle)

ax.set_title(' debt_financing by country_code')

# grant recipients by country_code



LG = investments.nlargest(833,'grant')

ax = LG.country_code.value_counts().plot(kind='pie',autopct='%.2f%%',figsize=(12,12))

figG=plt.gcf()

figG.gca()

ax.set_title(' grant recipient startups by country_code')



# 10 largest funding_total_used and respective venture in Finland market

gbf=investments[(investments['country_code'] == 'FIN')]



gg= gbf.groupby('market').sum()

LF = gg.nlargest(10,'funding_total_usd')

fg=LF.plot(kind ='bar', y=['funding_total_usd','venture'], figsize=(20,10))



fg.set_title('Startups\' 10 largest funding_total_used and respective venture in Finland\'s market',fontsize=(20))
# The average fundding_total_used and respective venture by region in Finland

gbf=investments[(investments['country_code'] == 'FIN')]

gg= gbf.groupby('region').mean()

fg=gg.plot(kind ='bar', y=['funding_total_usd','venture'], figsize=(20,10))



fg.set_title('Startups\' average funding_total_usd and venture in  finland by region',fontsize=(20))

# The average funding from grant and debt_financing in Finland by region



gbf=investments[(investments['country_code'] == 'FIN')]

rg= gbf.groupby('region').mean()

fr=rg.plot(kind ='line', y=['grant','debt_financing'], figsize=(15,5))



fr.set_title('Startups\'s,grant and debt_financing in  finland by region',fontsize=(20))

# 200 largest grant for startups in Finland by region'

fin =investments[(investments['country_code'] == 'FIN')]

Lfd = fin.nlargest(200,'grant')

ax = Lfd.region.value_counts().plot(kind='pie',autopct='%.2f%%',figsize=(16,20))

add_circle = plt.Circle((0,0),0.7,color='white')

figd=plt.gcf()

figd.gca().add_artist(add_circle)

ax.set_title('200 largest grant for startups in Finland by region')
#checking differences in funding sources among the countries.

investments.groupby('country_code').sum()[['seed',

                                           'venture',

                                           'equity_crowdfunding', 

                                           'undisclosed', 

                                           'convertible_note',

                                           'debt_financing',

                                           'angel',

                                           'grant',

                                           'private_equity',

                                           'post_ipo_equity',

                                           'post_ipo_debt',

                                           'secondary_market', 

                                           'product_crowdfunding']].plot(kind = 'bar', figsize = (20,12), width = 1)

plt.title('Funding sources in Nordic countries', size = 'x-large')
investments['funding_in_seed'] = investments['seed'].map(lambda x :1  if x > 0 else 0)
plt.rcParams['figure.figsize'] =4,4

labels = ['No funding','Get funding']

sizes = investments['funding_in_seed'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

plt.axis('equal')

plt.tight_layout()

plt.title("Startups got funding in seed stage", fontdict=None, position= [0.48,1.1], size = 'x-large')



plt.show()
grouped_by_country = investments.groupby('country_code')

fin = grouped_by_country.get_group('FIN')

dnk = grouped_by_country.get_group('DNK')

isl = grouped_by_country.get_group('ISL')

nor = grouped_by_country.get_group('NOR')

swe = grouped_by_country.get_group('SWE')



fig, ax = plt.subplots(nrows=5, ncols=1, figsize = (5,30))



labels = ['No funding','Get funding']

sizes_fin = fin['funding_in_seed'].value_counts().tolist()

sizes_dnk = dnk['funding_in_seed'].value_counts().tolist()

sizes_isl = isl['funding_in_seed'].value_counts().tolist()

sizes_nor = nor['funding_in_seed'].value_counts().tolist()

sizes_swe = swe['funding_in_seed'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



ax[0].set_title("Finland")

ax[0].pie(sizes_fin, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[1].set_title("Denmark")

ax[1].pie(sizes_dnk, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[2].set_title("Iceland")

ax[2].pie(sizes_isl, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[3].set_title("Norway")

ax[3].pie(sizes_nor, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[4].set_title("Sweden")

ax[4].pie(sizes_swe, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

fig.suptitle('Startups got funding in seed stage in each Nordic country' , size = 'xx-large')

plt.show()

investments['funding_vc'] = investments['venture'].map(lambda v :1  if v > 0 else 0)
plt.rcParams['figure.figsize'] =3,3

labels = ['No funding','Get funding']

sizes = investments['funding_vc'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

plt.axis('equal')

plt.tight_layout()

plt.title("Startups got funding by VC", fontdict=None, position= [0.48,1.1], size = 'x-large')



plt.show()
grouped_by_country = investments.groupby('country_code')

fin = grouped_by_country.get_group('FIN')

dnk = grouped_by_country.get_group('DNK')

isl = grouped_by_country.get_group('ISL')

nor = grouped_by_country.get_group('NOR')

swe = grouped_by_country.get_group('SWE')



fig, ax = plt.subplots(nrows=5, ncols=1, figsize = (5,30))



labels = ['No funding','Get funding']

sizes_fin = fin['funding_vc'].value_counts().tolist()

sizes_dnk = dnk['funding_vc'].value_counts().tolist()

sizes_isl = isl['funding_vc'].value_counts().tolist()

sizes_nor = nor['funding_vc'].value_counts().tolist()

sizes_swe = swe['funding_vc'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



ax[0].set_title("Finland")

ax[0].pie(sizes_fin, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[1].set_title("Denmark")

ax[1].pie(sizes_dnk, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[2].set_title("Iceland")

ax[2].pie(sizes_isl, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[3].set_title("Norway")

ax[3].pie(sizes_nor, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[4].set_title("Sweden")

ax[4].pie(sizes_swe, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

fig.suptitle("Startups got funding by VC in each Nordic country" , size = 'xx-large')

plt.show()
investments['funding_angel'] = investments['angel'].map(lambda a :1  if a > 0 else 0)
plt.rcParams['figure.figsize'] =5,5

labels = ['No funding','Get funding']

sizes = investments['funding_angel'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

plt.axis('equal')

plt.tight_layout()

plt.title("Startups got funding by VC by angels", fontdict=None, position= [0.48,1.1], size = 'x-large')



plt.show()
grouped_by_country = investments.groupby('country_code')

fin = grouped_by_country.get_group('FIN')

dnk = grouped_by_country.get_group('DNK')

isl = grouped_by_country.get_group('ISL')

nor = grouped_by_country.get_group('NOR')

swe = grouped_by_country.get_group('SWE')



fig, ax = plt.subplots(nrows=5, ncols=1, figsize = (5,30))



labels = ['No funding','Get funding']

sizes_fin = fin['funding_angel'].value_counts().tolist()

sizes_dnk = dnk['funding_angel'].value_counts().tolist()

sizes_isl = isl['funding_angel'].value_counts().tolist()

sizes_nor = nor['funding_angel'].value_counts().tolist()

sizes_swe = swe['funding_angel'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['#ff9999','#99ff99'] 



ax[0].set_title("Finland")

ax[0].pie(sizes_fin, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[1].set_title("Denmark")

ax[1].pie(sizes_dnk, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[2].set_title("Iceland")

ax[2].pie(sizes_isl, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[3].set_title("Norway")

ax[3].pie(sizes_nor, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

ax[4].set_title("Sweden")

ax[4].pie(sizes_swe, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

fig.suptitle("Startups got funding by angels in each Nordic country" , size = 'xx-large')

plt.show()
# Total number of values and sum of each round (A - F) 



print('Total number of values in round_A: ', len(investments[investments['round_A'] != 0]))

print('Sum of round_A: $', investments['round_A'].sum())

print('')

print('Total number of values in round_B: ', len(investments[investments['round_B'] != 0]))

print('Sum of round_B: $', investments['round_B'].sum())

print('')

print('Total number of values in round_C: ', len(investments[investments['round_C'] != 0]))

print('Sum of round_C: $', investments['round_C'].sum())

print('')

print('Total number of values in round_D: ', len(investments[investments['round_D'] != 0]))

print('Sum of round_D: $', investments['round_D'].sum())

print('')

print('Total number of values in round_E: ', len(investments[investments['round_E'] != 0]))

print('Sum of round_E: $', investments['round_E'].sum())

print('')

print('Total number of values in round_F: ', len(investments[investments['round_F'] != 0]))

print('Sum of round_F: $', investments['round_F'].sum())
rounds = ['round_A','round_B','round_C','round_D','round_E','round_F']

amount = [investments['round_A'].sum(),

          investments['round_B'].sum(),

          investments['round_C'].sum(),

          investments['round_D'].sum(),

          investments['round_E'].sum(),

          investments['round_F'].sum()]
plt.rcParams['figure.figsize'] = 15,8

height = amount

bars =  rounds

y_pos = np.arange(len(bars))



plt.bar(y_pos, height , width=0.7, color= ['goldenrod','tomato','olivedrab','teal','chocolate','seagreen'] )

plt.ticklabel_format(style = 'plain')

plt.xticks(y_pos, bars)

ax = plt.axes()        

ax.yaxis.grid()

plt.title("Sum investment in each round", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()