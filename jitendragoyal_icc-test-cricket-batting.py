# This notbook is analysis of test cricket batting.

# I am doing this first time so please comment what you like or not in this notbook. 

# Suggestion are very important for me.

# if you find anything worng please specify or upvote if you learn from this.
# Importing libraries, we need further in this notebook.

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# read icc test cricket data in variable test_cricket

# creating test_cricket dataframe by reading icc test cricket csv file.

test_cricket = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='latin1')
# check the top of the dataframe

test_cricket.head()
# viewing the bottom eight line of the dataframe

test_cricket.tail(8)
# Player profile columns is not usable for us.

# Let drop it first

test_cricket.drop('Player Profile',axis=1,inplace=True)
# now describe the test cricket dataframe

test_cricket.describe(include='all', exclude=None)
# create new column and fill it by True if player is not out when he creates his High Score

test_cricket['HS_Not_Out'] = test_cricket['HS'].str.contains('*',regex=False)



# remove * from HS

test_cricket['HS'] = test_cricket['HS'].str.replace('*','')



# Check our new dataframe cojumn name HS_not_out

# test_cricket.head()
# convert HS_Not_Out column into integer means 1 for True and 0 for False

test_cricket['HS_Not_Out'] = test_cricket.HS_Not_Out.astype(int)

# test_cricket.head()
# now first check info of dataframe

test_cricket.info()
# DataFrame actual memory usage before converting dtype in bytes

test_cricket.memory_usage(deep=True).sum()
# this function changes dtype of column after removing - key word from the column



def change_dtype_replace_dash(col,datatype='int64'):

    test_cricket[col] = test_cricket[col].str.replace('-','0')

    test_cricket[col] = test_cricket[col].astype(datatype)
# only Avg has float value

change_dtype_replace_dash(col='Avg',datatype='float32')



# creating a list to change dtype as int64 and remove dash

col_list = ['Inn','HS','NO','Runs','100','50','0']

for col in col_list:

    change_dtype_replace_dash(col)

    

# view test_cricket data frame after changes in dtype

# test_cricket.head()



# dataframe info after changes and compare it from old

test_cricket.info()
# DataFrame actual memory usage after converting dtype

# result always in bytes

test_cricket.memory_usage(deep=True).sum()
# now describe the dataframe

test_cricket.describe()
test_cricket['HS_Not_Out'].value_counts()
# As we can see above player column has two things name and country in parentheses

# break this in two part

# create new data_frame by spliting player name and country in different columns

player_country = test_cricket['Player'].str.split("(",expand=True)

player_country.head()
# from above table we can see three column instead of two, WHY?

# Let's check it to find difference and improve it

# we first get unique value of column 2

player_country[2].unique()
# now check index where this 'PAK)' occurs

player_country[player_country[2] == 'PAK)']
# we can get same result by this line also (by checking datatype)

player_country[player_country[2].apply(lambda x: type(x) != type(None))]
# updating player name

player_country.iloc[2113,0] = 'Mohammad Nawaz (3)'



# updating country

player_country.iloc[2113,1] = 'PAK'
player_country.drop(2,axis=1,inplace=True)

player_country.head()
# column 0 contains one extra space after Last name Let's check it

player_country[0].loc[:4].str.len()
# remove extra space and replace/add it with our test_cricket player column

test_cricket['Player'] = player_country[0].str.strip()
# check extra space is removed or not

test_cricket['Player'].loc[:4].str.len()
# let's check all Country name

player_country[1].unique()
player_country[1].nunique()
# word ICC is not belong to any country

# so remove ), ICC/ and /ICC from player_country dataframe and add it in original dataframe

test_cricket['Country'] = player_country[1].str.replace('/ICC','').str.replace('ICC/','').str.replace(')','')

test_cricket.head()
test_cricket.Country.nunique()
countries = test_cricket.Country.unique().tolist()

countries
# breaking Span column in two column

career_span = test_cricket.Span.str.split('-',expand=True)

career_span.head()
# changeing column name

career_span.rename(columns={0:'PStart',1:'PStop'},inplace=True)

career_span.info()
career_span.memory_usage(deep=True).sum()
# changing data type of both columns

career_span = career_span.astype('int64')
career_span.info()
career_span.memory_usage(deep=True).sum()
# creating new column with data How many years a player played?

career_span['Span_Years'] = career_span.PStop - career_span.PStart

career_span.head()
# joining both the dataframe in one data frame

test_cricket = test_cricket.join(career_span)
test_cricket.head()
# this is the country wise counting of players for not out when they create their highest score

test_cricket.groupby('HS_Not_Out').Country.value_counts()
plt.figure(figsize=(12,8))

f = sns.barplot(x='HS_Not_Out',y='Country',data=test_cricket,estimator=sum,ci=None)

plt.show()
test_cricket[test_cricket.PStart == test_cricket.PStart.max()].Country.value_counts()
plt.figure(figsize=(20,8))

sns.boxplot(y='Avg',x='Country',data=test_cricket)

plt.show()
plt.figure(figsize=(10,8))

sns.jointplot(x='Inn',y='Runs',data=test_cricket,color='red',kind='reg')

plt.show()
# creating a dataframe - sum of 50s after grouping test_cricket by country

country_50 = test_cricket.groupby('Country')['50'].sum()



# mearging all country in other variable whoes 50s are less than 30

other = country_50[country_50 < 30].sum()



# removing countries, those 50s are less than 30

country_50 = country_50[country_50 >= 30]



# inserting other variable in dataframe

country_50['OTHER'] = other

country_50
plt.figure(figsize=(4,4))

plt.pie(x=country_50,labels=country_50.index,radius=3,autopct='%1.1f%%',colors=['tomato','slateblue','coral','yellowgreen','pink','skyblue','gray','brown','lightskyblue','violet','gold'])

plt.show()
plt.figure(figsize=(8,4))

sns.barplot(x='Player',y='NO',data=test_cricket.sort_values('NO',ascending=False).head(10),hue='Country',

            dodge=False,palette='Dark2')

plt.xticks(rotation=70)

plt.show()
player_count = test_cricket.Country.value_counts()

player_count
test_cricket[(test_cricket.Country == 'ENG/INDIA') | (test_cricket.Country == 'INDIA/PAK')]
# first we sort data by runs than grouped by country than use head for 10 players.

Top_10 = test_cricket.sort_values('Runs',ascending=False).groupby('Country').head(10)



# removing countries have less than 10 players

Top_10 = Top_10[Top_10.Country.map(Top_10.Country.value_counts() == 10)]



# giving the rank(in their country) to each player by Runs

Top_10['Rank'] = Top_10.groupby('Country').Runs.rank(ascending=False)



Top_10
# creating pivot table

Top_10_pivot = Top_10.pivot_table(index='Country',columns='Rank',values='Runs')

Top_10_pivot
plt.figure(figsize=(15,15))

sns.heatmap(Top_10_pivot,linewidths=0.1,annot=True,fmt='.1f',cmap='Spectral')

plt.show()
Top_10[Top_10.Country == 'IRE']
plt.figure(figsize=(12,8))

sns.violinplot(y='Runs',data=test_cricket[test_cricket.Country == 'INDIA'],inner="quartile",color='lightblue')

plt.ylabel('Runs')

plt.xlabel('INDIA')

plt.title('Runs distribution of INDIA')

plt.show()