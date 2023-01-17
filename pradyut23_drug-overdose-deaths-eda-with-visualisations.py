import numpy as np

import pandas as pd



df=pd.read_csv('/kaggle/input/drug-overdose-deaths/drug_deaths.csv',index_col='Unnamed: 0')
pd.set_option('max_columns',None)

pd.set_option('max_rows',None)

df.head()
columns=df.columns

columns=columns[-21:-4]

for i in columns:

    print(df[i].unique())

    

#Converting to int type

df[columns[3]]=df[columns[3]].astype('int64')



#Taking only 0 and 1 values for cause of death as Yes or No

columns=columns[2],columns[12],columns[16]

for i in columns:

    for j in range(5105):

        if df.loc[j,i]=='0' or df.loc[j,i]==0:

            df.loc[j,i]=0

        else:

            df.loc[j,i]=1

#Count of deaths due to each drug

columns=df.columns

drugs={}

for i in range(-21,-7,1):

    s=df[columns[i]].sum()

    print(columns[i],s)

    drugs[columns[i]]=s

for i in range(-6,-4):

    s=df[columns[i]].sum()

    print(columns[i],s)

    drugs[columns[i]]=s
df.groupby('Other').Other.count()
#Identifies and merges the same names with different spellings or errors in spellings

import fuzzywuzzy

from fuzzywuzzy import process



#Input:dataframe,column name, correct name, min similarity value, max similarity value and words to skip, in the order.

def replace_matches(df,column,string_to_match,min_ratio=50,max_ratio=100,leave=[]):

    strings=df[column].unique()

    matches=fuzzywuzzy.process.extract(string_to_match,strings,limit=10,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    close_matches=[match[0] for match in matches if match[1]>=min_ratio and match[1]<max_ratio and match[0] not in leave]

    rows=df[column].isin(close_matches)

    df.loc[rows,column]=string_to_match



df['Other']=df['Other'].str.lower()

df['Other']=df['Other'].str.strip()
other=df['Other'].unique()

#Getting word mattchings

matches=fuzzywuzzy.process.extract('buprenorphine',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

#Replacing the matches by correct word

replace_matches(df=df,column='Other',string_to_match='buprenorphine',min_ratio=61,leave=['morphine','morphiine','buprop','buprno'])

rows=df['Other'].isin(['buprenor, carfentanil','bupren, difluoro','bupre','bupren','pcp. bupren','bupren, hexadrone'])

df.loc[rows,'Other']='buprenorphine'



#Running for all the different words need to be corrected

matches=fuzzywuzzy.process.extract('hydromorphone',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='hydromorphone',min_ratio=67,leave=['morphine'])

rows=df['Other'].isin(['hydrom','h-morph','hydromorph, buprenor','hyd-morph','hydr-mor'])

df.loc[rows,'Other']='hydromorphone'



matches=fuzzywuzzy.process.extract('morphine',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='morphine',min_ratio=59,leave=['hydromorphone','buprenorphine','mitragine'])

rows=df['Other'].isin(['morphine, no rx in pmp','morphine no 6mam','morph/cod','morph pcp'])

df.loc[rows,'Other']='morphine'



matches=fuzzywuzzy.process.extract('opioid',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='opioid',min_ratio=46,leave=['zolpidem','buprenorphine','morphine'])

rows=df['Other'].isin(['u-47700 synthetic opioid','opiate screen','u-47700','u-47700, carfentanil','u47700'])

df.loc[rows,'Other']='opioid'



matches=fuzzywuzzy.process.extract('zolpidem',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='zolpidem',min_ratio=50,leave=['opioid','morphine'])



matches=fuzzywuzzy.process.extract('diphenhydramine',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='diphenhydramine',min_ratio=56)

rows=df['Other'].isin(['diphen, chlorphen'])

df.loc[rows,'Other']='diphenhydramine'



matches=fuzzywuzzy.process.extract('phenobarbital',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='phenobarbital',min_ratio=67)



matches=fuzzywuzzy.process.extract('difluorofentanyl',other,limit=20,scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches(df=df,column='Other',string_to_match='difluorofentanyl',min_ratio=58,leave=['carfentanil'])
#Replacing remaining wrong words

rows=df['Other'].isin(['buprop','buprno'])

df.loc[rows,'Other']='bupropion'



rows=df['Other'].isin(['mdma, n-ethyl-pentylone','mdma, pcp'])

df.loc[rows,'Other']='mdma'



rows=df['Other'].isin(['butalb'])

df.loc[rows,'Other']='butalbital'



rows=df['Other'].isin(['cod'])

df.loc[rows,'Other']='codeine'  



rows=df['Other'].isin(['ket'])

df.loc[rows,'Other']='ketamine'



rows=df['Other'].isin(['parox'])

df.loc[rows,'Other']='paroxetine'



rows=df['Other'].isin(['pcp, n-ethyl-pentylone','n-ethyl-pentylone','methoxypcp','pcp, morphine nos'])

df.loc[rows,'Other']='pcp'



rows=df['Other'].isin(['mitragine','mirtagynine'])

df.loc[rows,'Other']='mitragynine'



rows=df['Other'].isin(['others'])

df.loc[rows,'Other']='unidentified'

df['Other']=df['Other'].str.title()

df=df.replace('Other','Unidentified')

df=df.replace('Unknown','Unidentified')
#Clean 'Other' column

df.groupby('Other').Other.count().sort_values()
#Merging all the drugs from the Dataset and the 'Other' column as a datasset

j=0

other=df['Other'].unique()

for i in df.groupby('Other').Other.count().sort_values(ascending=False):

    drugs[other[j]]=i

    j+=1

drugsdf=pd.DataFrame(drugs.items())

drugsdf=drugsdf.sort_values(by=[1],axis=0,ascending=False)

drugsdf.reset_index(drop=True)
#No. of deaths caused by different drugs

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,5))

sns.barplot(x=drugsdf[0],y=drugsdf[1],data=drugsdf)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.xlabel('Drugs',size=17)

plt.ylabel('Deaths',size=17)

plt.title('Deaths Due To Different Drugs',size=20)
#Yearwise no. of deaths

plt.figure(figsize=(10,5))

year = pd.DataFrame(pd.to_datetime(df['Date']).dt.year.value_counts())

print(year)

sns.barplot(x=year.index.astype('int64'),y=year['Date'],data=year)

plt.title('Deaths in each Year',size=15)

plt.xlabel('Year',size=12)

plt.ylabel('Deaths',size=12)
#Deaths of residents of different cities

city=pd.DataFrame(df['ResidenceCity'].value_counts())

city
plt.figure(figsize=(15,5))

sns.barplot(x=city.index[:40],y=city['ResidenceCity'][:40],data=drugsdf)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.xlabel('Residence',size=17)

plt.ylabel('Deaths',size=17)

plt.title('Deaths of Residents of Top 40 cities',size=20)
#Deaths in different cities

death_city=pd.DataFrame(df['DeathCity'].value_counts())

death_city
plt.figure(figsize=(15,5))

sns.barplot(x=death_city.index[:40],y=death_city['DeathCity'][:40],data=death_city)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.xlabel('City',size=17)

plt.ylabel('Deaths',size=17)

plt.title('Top 40 Cities with Highest no. of Reported Deaths',size=20)
#Death distribution by sex ratio

male = df['Sex'].value_counts().values[0]

female = df['Sex'].value_counts().values[1]

plt.pie([male,female],labels=['Male','Female'],autopct= lambda x:'{:.2f}%  ({:,.0f})'.format(x,x*sum([male,female])/100),shadow=True, startangle=90,radius=1.8,textprops={'fontsize':16})
#Death distribution by race

print(df['Race'].unique())



plt.figure(figsize=(15,5))

sns.countplot(x='Race',data=df)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.xlabel('Race',size=12)

plt.ylabel('Deaths',size=12)

plt.title('Deaths Among Different Ethnicities',size=20)
#Death distribution by age

df['Age']=df['Age'].replace('Unidentified',0)

plt.figure(figsize=(15,5))

sns.distplot(df['Age'])

plt.xlabel('Age',size=12)

plt.title('Death Distribution by Age',size=20)
#Word Cloud

from wordcloud import WordCloud

death_city = df['DeathCity'].copy().dropna()

death_city_cloud = ' '.join(city for city in death_city)

plt.figure(figsize=(10,7))

wc=WordCloud(width=3000, height=2000).generate(death_city_cloud)

plt.imshow(wc,interpolation='bilinear')

plt.axis('off')