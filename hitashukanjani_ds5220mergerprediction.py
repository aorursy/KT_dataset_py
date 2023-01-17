# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns

import matplotlib.pyplot as plt

import json

import os

import re

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/crunchbasse-companies-details-dataset/Data2.xlsx')

df = df.replace(to_replace ="—", 

                 value = np.nan) 

df =  df.replace(to_replace ="—\r\n", 

                 value = np.nan, regex=True)



df.head()
df.isnull().sum()
feature_with_nan = [features for features in df.columns if df[features].isnull().sum()>1 and df[features].dtype == 'O']

len(feature_with_nan)



for features in feature_with_nan:

    print('{}: {}% missing value'.format(features, np.around(df[features].isnull().mean()*100,4)))
#numericals features

numerical_feature = [feature for feature in df.columns if df[feature].dtype != 'O']  

print('Number of numerical variables:', len(numerical_feature))  
#categorical feature

categorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']  

print('Number of categorical variables:', len(categorical_feature))  

categorical_feature
#number of categories in each feature

for feature in categorical_feature:

    print('The feature is {} and number of categories are {}'.format(feature, len(df[feature].unique())))

plt.figure(figsize=(10, 6))

status_count = df['Status'].value_counts()

bar_sector = sns.barplot(x = status_count.index, y = status_count.values ,palette="deep")

bar_sector.axes.set_title("Distribution of Company Status",fontsize=15)

bar_sector.set_xlabel("Status",fontsize=15)

bar_sector.set_ylabel("Count",fontsize=15)

bar_sector.tick_params(labelsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=0)

for i,j in zip(bar_sector.patches,status_count.values):

  height = i.get_height()

  bar_sector.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom',fontsize=12)



plt.show()
#drop unwanted columns



df = df.drop(['Name','Full Name','Number of Lead Investment', 'Investment Stage',

              'Number of Investment','Status','Purpose','Primary Organization'], axis=1)
df_sectors = ','.join(df['Type'])

sectors = df_sectors.split(',')

sectors = pd.DataFrame(df_sectors.split(','),columns=['Type'])

sector_cat_count = sectors['Type'].unique()

len(sector_cat_count)
#consider only upto 7 sectors for a company



types = []

for type in df['Type']:

  sep = ','

  rest = ""

  if len(type.split(sep)) > 7:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] + ',' + type.split(sep)[3] + ',' + type.split(sep)[4] + ',' + type.split(sep)[5] + ',' + type.split(sep)[6] + ',' + type.split(sep)[7]

  elif len(type.split(sep)) > 6:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] + ',' + type.split(sep)[3] + ',' + type.split(sep)[4] + ',' + type.split(sep)[5] + ',' + type.split(sep)[6]

  elif len(type.split(sep)) > 5:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] + ',' + type.split(sep)[3] + ',' + type.split(sep)[4] + ',' + type.split(sep)[5]

  elif len(type.split(sep)) > 4:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] + ',' + type.split(sep)[3] + ',' + type.split(sep)[4]

  elif len(type.split(sep)) > 3:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] + ',' + type.split(sep)[3]

  elif len(type.split(sep)) > 2:

    rest = type.split(sep)[1] + ',' + type.split(sep)[2] 

  elif len(type.split(sep)) > 1:

    rest = type.split(sep)[1]

  else:

    rest = type.split(sep)[0]

  types.append(rest)



df['Type'] = types



#find top 25 sectors overall

df_sectors = ','.join(df['Type'])

sectors = df_sectors.split(',')

sectors = pd.DataFrame(df_sectors.split(','),columns=['Type'])

sector_cat_count = sectors['Type'].value_counts()

sector_cat_count = sector_cat_count.sort_values(ascending=False)

top_sectors = sector_cat_count[0:25]



#replace entire dataframe with top 25 sectors

type_df = []

for type in df['Type']:

  types = type.split(',')

  intersect = [x for x in types if x in top_sectors.index]

  strg = ','.join([str(elem) for elem in intersect])

  if not strg:

    strg = "Other"

  type_df.append(strg)

  



df['Type'] = type_df
# #convert vales of different currency to USD 

# !pip install currencyconverter

# from currency_converter import CurrencyConverter

# def get_symbol(price):

#         import re

#         pattern =  r'(\D*)\d*\.?\d*(\D*)'

#         g = re.match(pattern,price).groups()

#         return g[0] 

 



# c = CurrencyConverter()

# converted_amount = []

# for price in df['Funding Amount']:

#     if isinstance(price, float):

#         converted_amount.append(int(0))

#         continue

#     symbol = get_symbol(price)

#     prc = price.replace(symbol, "").replace(",", "")

#     prc = int(prc)

#     if symbol == '£':

#         prc = c.convert(prc, 'GBP','USD')

#     elif symbol == '€':

#         prc = c.convert(prc, 'EUR','USD')

#     elif symbol == 'IDR':

#         prc = c.convert(prc, 'IDR','USD')

#     elif symbol == 'NOK':

#         prc = c.convert(prc, 'NOK','USD')

#     elif symbol == 'PLN':

#         prc = c.convert(prc, 'PLN','USD')

#     elif symbol == 'CN¥':

#         prc = c.convert(prc, 'CNY','USD')

#     elif symbol == 'MYR':

#         prc = c.convert(prc, 'MYR','USD')

#     elif symbol == 'PHP':

#         prc = c.convert(prc, 'PHP','USD')

#     elif symbol == 'CA$':

#         prc = c.convert(prc, 'CAD','USD')

#     elif symbol == 'RUB':

#         prc = c.convert(prc, 'RUB','USD')

#     elif symbol == 'SEK':

#         prc = c.convert(prc, 'SEK','USD')

#     elif symbol == 'CHF':

#         prc = c.convert(prc, 'CHF','USD')

#     elif symbol == 'ISK':

#         prc = c.convert(prc, 'ISK','USD')

#     elif symbol == 'TRY':

#         prc = c.convert(prc, 'TRY','USD')

#     elif symbol == 'SGD':

#         prc = c.convert(prc, 'SGD','USD')

#     elif symbol == 'ZAR':

#         prc = c.convert(prc, 'ZAR','USD')

#     elif symbol == 'DKK':

#         prc = c.convert(prc, 'DKK','USD')

#     elif symbol == 'A$':

#         prc = c.convert(prc, 'AUD','USD')

#     elif symbol == '₹':

#         prc = c.convert(prc, 'INR','USD')

#     elif symbol == 'R$':

#         prc = c.convert(prc, 'BRL','USD')

#     elif symbol == '¥':

#         prc = c.convert(prc, 'JPY','USD')

#     elif symbol == '₩':

#         prc = c.convert(prc, 'KRW','USD')

#     elif symbol == 'MX$':

#         prc = c.convert(prc, 'MXN','USD')

#     elif symbol == 'NZ$':

#         prc = c.convert(prc, 'NZD','USD')

#     elif symbol == '₪':

#         prc = c.convert(prc, 'ILS','USD')

#     elif symbol != '$':

#         df = df[df['Funding Amount'] != price]

#         continue

#     else:

#         converted_amount.append(prc)

#         continue

#     prc = round(prc)

#     converted_amount.append(prc)
# #Categorize Funding Amount

# cc_amount = pd.cut(converted_amount,bins=[-1,1000000,10000000,100000000,30079814466],

#        labels=['Less than $1M','$1M to $10M','$10M to 100M','100M+'])

# df['Funding Amount'] = cc_amount
df['Acquisition Status'].value_counts()
# Merge Acquisition Status



df['Acquisition Status'] = df['Acquisition Status'].astype(str)

df['Acquisition Status'] = df['Acquisition Status'].map(lambda x: re.sub(r'\W+', '', x))

df['Acquisition Status'] = df['Acquisition Status'].replace(to_replace ="MadeAcquisitionsWasAcquired", value = "Merger")

df['Acquisition Status'] = df['Acquisition Status'].replace(to_replace ="WasAcquired", value = "Merger")

df['Acquisition Status'] = df['Acquisition Status'].replace(to_replace ="MadeAcquisitions", value = "Acquisition")

df['Acquisition Status'] = df['Acquisition Status'].replace(to_replace ='nan',value = 'No Participation')
# Categorize Job Titles into Most common 6 Titles



df['Job Title'].fillna("Other", inplace = True) 

df['Job Title'] =  df['Job Title'].str.upper()

df['Job Title'] =  df['Job Title'].apply(lambda x: x.strip())




# Categorize common Bizzare words into particular job title

df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF EXECUTIVE OFFICER','CHIEF EXECUTIVE','C.E.O','CEO (CEO)','CEO (EMEA)',

                                                      'CEO INTERNATIONAL','CEO CO-FOUNDER','CEO (CEO)','FOUNDER (CEO)','CEO CROSSBORDER',

                                                     'CEO - HYPERGIANT INDUSTRIES','CEO - HYPERGIANT INDUSTRIES','FOUNDING CEO','CEO OF BLIZZARD ENTERTAINMENT',

                                                     'CEO (FMR. CTO)','CEO OF 3D NANO BATTERIES LLC','MD - CEO','CEO OF 3D NANO BATTERIES LLC','CO-CEO',

                                                     'CEO OF STITCH','CEO OFFCER','CO - CEO','FOUNDER - CEO','CEO TECHHUB','FOUNDER I CEO','CEO IN RESIDENCE'

                                                     ]):'CEO'},regex=True)





df['Job Title'] = df['Job Title'].replace({'|'.join(['CO FOUNDER','COFOUNDER','CO - FOUNDER','CO- FOUNDER','CO-FOUNDER','CO -FOUNDER',

                                                               'CO-FOUNDER - MYPROTEIN GROUP','CO-FOUNDER','MEMBER OF FOUNDING TEAM','TITLECO-FOUNDER',

                                                               'CO-FOUNDER - STARTUPBOOTCAMP AFRICA','FOUNDER']):'CO-FOUNDER'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF TECHNOLOGY OFFICER','CHIEF TECHNICAL OFFICER','CTO SOFTWARE','CHIEF TECHNOLOGY','CTO'

                                                     ]):'CTO'},regex=True)

df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF OPERATING OFFICER']):'COO'},regex=True)

df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF PRODUCT OFFICER','CPO ITALY']):'CPO'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['VICE PRESIDENT','VICE PRESIDENT OF ENGINEERING','VP OF RESEARCH','VP OF STRATEGIC BUSINESS DEVELOPMENT',

                                                      'VP IMAGING SYSTEMS','VP BUSINESS DEVELOPMENT','CORPORATE VICE PRESIDENT','VP OF ENGINEERING','VP OF PRODUCT',

                                                      'VP OF LEARNING AT WORK','VP BUSINESS DEVELOPMENT','VP PRODUCT','VP ENGINEERING','VP MARKETING','VP PRODUCT MANAGEMENT',

                                                      'VP MACHINE LEARNING','EXECUTIVE VICE-PRESIDENT','VICE-PRESIDENT','VP OPERATIONS','VP R','VP MANAGEMENT','VP - COMMERCIAL',

                                                      'VP OF GROWTH','VP OF TECHNOLOGY','VP OF OPS','VP CONTENT','VP MANAGEMENT','VP OF PRODUCT','VP OF TECHNOLOGY',

                                                      'VP OF INVESTMENTS','VP TECHNOLOGY','VP OF PROFESSIONAL SERVICES','VP DATA','VP OF INNOVATION','VP OF GROWTH'

                                                      'VP OF STRATEGIC BUSINESS DEVELOPMENT','VP OF ALGORITHMS','VP - ENTERPRISE DEVELOPMENT','VP OF SPECIAL PROJECTS (POSTMATES X)',

                                                      'V.P. MATERIALS DEVELOPMENT','VP OF STRATEGIC INITIATIVES','VP STRATEGY','CORPORATE VP',

                                                     'VP OF INVESTOR RELATIONS']):'VP'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF MARKETING OFFICER']):'CMO'},regex=True)

df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF FINANCIAL OFFICER']):'CFO'},regex=True)

df['Job Title'] = df['Job Title'].replace({'|'.join(['CHIEF COMMERCIAL OFFICER']):'CCO'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['MANAGING DIRECTOR','DIRECTOR','MANAGING DIRECTOR SOUTHEAST ASIA','MANAGING DIRECTOR OPERATIONS',

                                                     'GLOBAL MD','MD MIDDLE EAST INTERNET GROUP','BUSINESS DEVELOPMENT MD','MEDICAL MD',

                                                     'NON-EXECUTIVE MD','CREATIVE MD OF ADVERTISING','EXECUTIVE MD OF THE HYPERLEDGER PROJECT',

                                                     'MD OF ENGINEERING','MD OF MARKETING','FINANCE MD','CREATIVE MD','MD OF FINANCE','GENERAL MD',

                                                     'MD HONG KONG','MD OF R','MD OF THE STANFORD AI LAB','MD OF GLOBAL ACQUIRING','REGIONAL MD'

                                                     'GROUP MD','PROGRAM MD','MD OF PURCHASING','MD OF GLOBAL FINANCIAL SOLUTIONS','MD OF SPECIAL PROJECTS',

                                                     'TECHNICAL MD','MD OF BD','SR. MD','PRESIDENT MD','SENIOR MD OF INVESTMENTS','MD OF TECHNOLOGY',

                                                     'MD OPERATIONS','MD OF PRODUCT','REGIONAL MD','FOUNDING MD']):'MD'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['BOARD MEMBER','CHAIRMAN OF THE BOARD','BOARD OF DIRECTORS','MEMBER OF THE BOARD OF DIRECTORS',

                                                               'CHAIRMAN OF THE BOARD OF DIRECTORS','BOARD DIRECTOR','MEMBER OF THE BOARD','BOARD MEMBERS',

                                                               'BOD OF DIRECTORS','BOARD CHAIRMAN','MEMBER OF BOARD','BOARD OF DIRECTER','BOD OF MDS',

                                                                'BOARD OF MDS','BOARD OF MD','ADVISORY BOD','CHAIRMAN OF BOD','BOARD MD','BOARD OF OVERSEERS'

                                                     ]):'BOD'},regex=True)



df['Job Title'] = df['Job Title'].replace({'|'.join(['SENIOR VP','SENIOR VP OF MANUFACTURING','SVP OF SALES','VP OF ENTREPRENEURSHIP','SPV','SVP DIGITAL PRODUCT',

                                                     'SPV OF PRODUCT','SVP OF R','SPV OF MARKETING','SR. VP','SVP OF MANUFACTURING','SVP INNOVATION']):'SVP'},regex=True)



# Top 6 titles

df_title = ' & '.join(df['Job Title'])

df_title = df_title.replace('AND','&').replace(',','&').replace('/','&').replace('|','&').replace('+','&')

titles = pd.DataFrame(df_title.split('&'),columns=['Job Title'])

titles['Job Title'] = titles['Job Title'].apply(lambda x: x.strip())



title_count = titles['Job Title'].value_counts()

title_count = title_count.sort_values(ascending=False)

top_titles = title_count[0:6]



#replace entire dataframe with top 6 Job titles

title_df = []

for type in df['Job Title']:

  type = type.replace('AND','&').replace(',','&').replace('/','&').replace('|','&').replace('+','&')

  types = type.split('&')



  intersect = [x.strip() for x in types if x.strip() in top_titles.index]

  strg = '&'.join([str(elem) for elem in intersect])

  if not strg:

    strg = "OTHER"

  title_df.append(strg)

  

df['Job Title'] = title_df
#Continent wise data cleaning



df_h_list = []

countries = []

states = []

cities = []

for i in df['Headquaters']:

  df_h = i.split(',')

  df_h_list.append(df_h)

for i in df_h_list:

  if(len(i)==3):

    i[2] = i[2].strip()

  

    countries.append(i[2])

    states.append(i[1])

    cities.append(i[0])



countries_df = pd.DataFrame(countries,columns = ['Countries'])
import pandas as pd

df1 = pd.read_csv('/kaggle/input/country-continent/country_cont.csv')

asia1 = []

africa1 = []

south_america1 = []

north_america1 = []

europe1 = []

oceania1 = []

asia = df1['Continent_Name']=='Asia'

for i in df1[asia]['Country_Name']:

  i = i.split(',')

  i[0] = i[0].strip()

  asia1.append(i[0])





africa = df1['Continent_Name']=='Africa'

for i in df1[africa]['Country_Name']:

  i = i.split(',')

  i[0] = i[0].strip()

  africa1.append(i[0])





na = df1['Continent_Name']=='North America'

for i in df1[na]['Country_Name']:

  i = i.split(',')

  i[0] = i[0].strip()

  # if(i[0]=='United States of America'):

  #   i[0] = i[0].replace('United States of America','United States') 

  north_america1.append(i[0])





sa = df1['Continent_Name']=='South America'

for i in df1[sa]['Country_Name']:

  

  i = i.split(',')

  i[0] = i[0].strip()

  south_america1.append(i[0])





europe = df1['Continent_Name']=='Europe'

for i in df1[europe]['Country_Name']:

  

  i = i.split(',')

  i[0] = i[0].strip()

  europe1.append(i[0])



oceania = df1['Continent_Name']=='Oceania'

for i in df1[oceania]['Country_Name']:

  i = i.split(',')

  i[0] = i[0].strip()

  oceania1.append(i[0])





for i in countries_df['Countries']:

  if(i == 'South Korea'):

    countries_df['Countries'].replace(i,'Asia',inplace = True)



  if(i in asia1):

    countries_df['Countries'].replace(i,'Asia',inplace = True)



for i in countries_df['Countries']:

  if(i == 'United States'):

    countries_df['Countries'].replace(i,'North America',inplace = True)

  if(i in north_america1):

    countries_df['Countries'].replace(i,'North America',inplace = True)

    



    

  # if(i in north_america1):

  #   print(i)

  #   countries_df['Countries'].replace(i,'North America',inplace = True)



for i in countries_df['Countries']:

  if(i == "Côte d'Ivoire"):

    countries_df['Countries'].replace(i,'Africa',inplace = True)



  if(i in africa1):

    countries_df['Countries'].replace(i,'Africa',inplace = True)



for i in countries_df['Countries']:

  if(i in oceania1):

    countries_df['Countries'].replace(i,'Australia',inplace = True)



for i in countries_df['Countries']:

  if(i in south_america1):

    countries_df['Countries'].replace(i,'South America',inplace = True)



for i in countries_df['Countries']:

  if(i=='United Kingdom'):

    countries_df['Countries'].replace(i,'Europe',inplace = True)

  if(i=='The Netherlands'):

    countries_df['Countries'].replace(i,'Europe',inplace = True)



  if(i in europe1):

    countries_df['Countries'].replace(i,'Europe',inplace = True)



df['Headquaters'] = countries_df['Countries']

countries_df['Countries'].unique()

#Categorize number of Employees



df['Number of employees'].replace(np.nan,'11-50',inplace = True)

df['Number of employees'].replace('1-10','<10',inplace = True)

df['Number of employees'].replace('11-50','11-50',inplace = True)

df['Number of employees'].replace('51-100','50-100',inplace = True)



df['Number of employees'].replace('101-250','100-500',inplace = True)

df['Number of employees'].replace('251-500','100-500',inplace = True)



df['Number of employees'].replace('501-1000','500-5000',inplace = True)

df['Number of employees'].replace('1001-5000','500-5000',inplace = True)



df['Number of employees'].replace('5001-10000','>5000',inplace = True)

df['Number of employees'].replace('10001+','>5000',inplace = True)

#Estimated Revenue Categorization





df['Estimated Revenue'].replace('Less than $1M','Less than $1M',inplace = True)



df['Estimated Revenue'].replace('$1M to $10M','$1M - $10M',inplace = True)



df['Estimated Revenue'].replace('$10M to $50M','$10M - $100M',inplace = True)

df['Estimated Revenue'].replace('$50M to $100M','$10M - $100M',inplace = True)



df['Estimated Revenue'].replace('$100M to $500M','$100M-$1B+',inplace = True)

df['Estimated Revenue'].replace('$500M to $1B','$100M-$1B+',inplace = True)

df['Estimated Revenue'].replace('$1B to $10B','$100M-$1B+',inplace = True)

df['Estimated Revenue'].replace('$10B+','$100M-$1B+',inplace = True)



df['Estimated Revenue'].replace(np.nan,'$1M to $10M',inplace = True) #mode



df['Estimated Revenue'].unique()
# Categorize Founders 



df['Founders']= df['Founders'].astype(float) 

print(df['Founders'].median())



df['Founders'] = df['Founders'].fillna('2')

df = df.loc[df['Founders'].astype(float) <= 7]



df['Founders']= df['Founders'].astype(float) 



df['Founders']=pd.cut(df['Founders'], bins=[0,1,2,7], labels = ["1","2","3-7"])
df['Acquisitions'].value_counts()

df['Acquisitions'].isnull().sum()

# # Categorize Number of Acquisitions

# df['Acquisitions']= df['Acquisitions'].astype(float) 

# #print(data['Acquisitions'].median())



# df['Acquisitions'] = df['Acquisitions'].fillna('2')

# df = df.loc[df['Acquisitions'].astype(float) <= 10]

# df['Acquisitions']= df['Acquisitions'].astype(float) 

# df['Acquisitions']=pd.cut(df['Acquisitions'], bins=[0,1,5,10], labels = ["1","2-5","6-10"])
#binning funding rounds and removed outliers

df['Funding Rounds']= df['Funding Rounds'].astype(float) 

print(df['Funding Rounds'].median())

df['Funding Rounds'] = df['Funding Rounds'].fillna('3')

df = df.loc[df['Funding Rounds'].astype(float) <= 13]

df['Funding Rounds']= df['Funding Rounds'].astype(float)

df['Funding Rounds']=pd.cut(df['Funding Rounds'], bins=[0,1,4,13], labels = ["1","2-4","5-13"])
import plotly.express as px

fig = px.box(df, y='Active Products', hover_data= ["Active Products"], width=400, height=400)

fig.update_layout(

    yaxis = dict(range=[0, 500]),

    yaxis_title="Active Products Distribution",

    font=dict(

         size=12

    )

)

fig.show()
def normalize_outliers(data):

    threshold = 1.5

    for col in data.columns:

      normalized_data = []

      mean = np.mean(data[col])

      std = np.std(data[col])

      for y in data[col]:

          z_score= (y - mean)/std

          if np.abs(z_score) > threshold:

            normalized_data.append(mean)

          else:

            normalized_data.append(y)

      data[col] = normalized_data

    return data


z = pd.DataFrame(df['Active Products'].astype(float))

z['Active Products'] = z.fillna(z['Active Products'].astype(float).mean())

print(df['Active Products'].astype(float).mean())

print(df['Active Products'].astype(float).median())

print(df['Active Products'].astype(float).std())

z = normalize_outliers(z)

import plotly.express as px

fig = px.box(z, y='Active Products', hover_data= ["Active Products"], width=400, height=400)

fig.update_layout(

    yaxis = dict(range=[0, 150]),

    yaxis_title="Active Product Distribution",

    font=dict(

         size=12

    )

)

fig.show()
#binning Active products

df['Active Products'] = z

df['Active Products']= df['Active Products'].astype(float) 

df['Active Products']=pd.cut(df['Active Products'], bins=[-1,10.0,22.0,30.0,60.0], labels = ["1-10","11-22","23-30","31-60"])

#Nunmber of Lead Investors

df['Nunmber of Lead Investors']= df['Nunmber of Lead Investors'].astype(float) 

print(df['Nunmber of Lead Investors'].median())

df['Nunmber of Lead Investors'] = df['Nunmber of Lead Investors'].fillna('2')



df['Nunmber of Lead Investors'] = df['Nunmber of Lead Investors'].astype(float)

df['Nunmber of Lead Investors']=pd.cut(df['Nunmber of Lead Investors'], bins=[0,1,3,15], labels = ["1","2-3","4-15"])
#Bining Number of Investors



df['Nmber of Investors']= df['Nmber of Investors'].astype(float) 

print(df['Nmber of Investors'].median())



df['Nmber of Investors'] = df['Nmber of Investors'].fillna('5')



df['Nmber of Investors'] = df['Nmber of Investors'].astype(float)

df['Nmber of Investors']=pd.cut(df['Nmber of Investors'], bins=[0,1,4,9,104], labels = ["1","2-3","4-20", "21-104"])
#Binning Founded organizations

df['Founded Organization']= df['Founded Organization'].astype(float) 

print(df['Founded Organization'].median())

df['Founded Organization'] = df['Founded Organization'].fillna('2')



df['Founded Organization']  = df['Founded Organization'] .astype(float)

df['Founded Organization'] =pd.cut(df['Founded Organization'] , bins=[0,1,3,23], labels = ["1","2-3","4-23"])
df['Founded Organization'].isna().sum()
#binning Portfolio companies



df['Portfolio Companies']= df['Portfolio Companies'].astype(float) 

print(df['Portfolio Companies'].median())

df['Portfolio Companies'] = df['Portfolio Companies'].fillna('3')



df['Portfolio Companies']  = df['Portfolio Companies'].astype(float)

df['Portfolio Companies'] =pd.cut(df['Portfolio Companies'], bins=[0,2,3,208], labels = ["1-2","3","4-208"])
# funding status



val = df['Funding Status'].mode()

print(val)

df['Funding Status'] = df['Funding Status'].fillna('Early Stage Venture')

df.head()
#categorical feature

categorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']  

print('Number of categorical variables:', len(categorical_feature))  



#number of categories in each feature

for feature in categorical_feature:

    print('The feature is {} and number of categories are {}'.format(feature, len(df[feature].unique())))

countries_cat_count = countries_df['Countries'].value_counts()

countries_cat_count = countries_cat_count.sort_values(ascending = False)

print('There are a total of ',len(countries_cat_count),'different continents in our dataset')



countries_cat_count = countries_cat_count.iloc[:30]

print(countries_cat_count)



plt.figure(figsize=(14,8))

bar_countries = sns.barplot(x =countries_cat_count.index, y = countries_cat_count.values,palette="YlOrRd")

sns.set_style("white")

plt.title('Distribution of different continents')

plt.xlabel('Continent')

plt.ylabel('Counts of each continent')

locs, labels = plt.xticks()

plt.setp(labels, rotation=0)

for i,j in zip(bar_countries.patches,countries_cat_count.values):

  height = i.get_height()

  bar_countries.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.savefig('Headquaters.png',bbox_inches='tight')

df_sectors = ','.join(df['Type'])

sectors = pd.DataFrame(df_sectors.split(','),columns=['Type'])

sector_cat_count = sectors['Type'].value_counts()





print('There are',len(sector_cat_count),'different sectors of Businesses in our Dataset')



plt.figure(figsize=(14,8))

bar_sector = sns.barplot(x =sector_cat_count.index, y = sector_cat_count.values,palette="deep")

sns.set_style("white")

plt.title('Distribution of different sectors of business')

plt.xlabel('categories of sectors')

plt.ylabel('Counts of sector appearing in companies')

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

for j,i in enumerate(bar_sector.patches):

  height = i.get_height()

  bar_sector.text(i.get_x() + i.get_width()/2, height + 5, sector_cat_count[j], ha='center', va='bottom')

plt.savefig('Sectors.png',bbox_inches='tight')
df_title = '&'.join(df['Job Title'])

titles = pd.DataFrame(df_title.split('&'),columns=['Job Title'])

title_count = titles['Job Title'].value_counts()

title_count = title_count.sort_values(ascending=False)



print('There are',len(title_count),'different Job titles  in our Dataset')



plt.figure(figsize=(14,8))

bar_sector = sns.barplot(x =title_count.index, y = title_count.values,palette="rocket")

sns.set_style("white")

plt.title('Top Job Titles of Investors')

plt.xlabel('Job Titles')

plt.ylabel('Count')

locs, labels = plt.xticks()

plt.setp(labels, rotation=0)

for j,i in enumerate(bar_sector.patches):

  height = i.get_height()

  bar_sector.text(i.get_x() + i.get_width()/2, height + 5, title_count[j], ha='center', va='bottom')

plt.savefig('Job Titles.png',bbox_inches='tight')


founded_y = []

founded1 =[]

leny = []

for i in df['Founded Year']:

  if isinstance(i, float):

    i= ' '

  i = i.split(' ')

  founded_y.append(i)



for i in founded_y:

  if(len(i)==3):

    founded1.append(i[2])

  elif(len(i)==2):

    founded1.append(i[1])

  elif(len(i)==1):

    founded1.append(i[0])





year_counts = np.unique(founded1,return_counts=True)





year_counts1 = year_counts[0][:-1]

year_counts1_c = year_counts[1][:-1]



year_counts1 = year_counts1[::-1]

year_counts1_c = [int(i) for i in year_counts1_c]

year_counts1_c = year_counts1_c[::-1]



year_counts1 = year_counts1[0:30]

year_counts1_c = year_counts1_c[0:30]





plt.figure(figsize=(14,8))



bar_year = sns.barplot(x =year_counts1, y = year_counts1_c,palette="PuBu")

plt.title('Distribution of different  founding years')

plt.xlabel('Years')

plt.ylabel('Counts of each year')

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)





for i,j in zip(bar_year.patches,year_counts1_c[::-1]):

  height = i.get_height()

  bar_year.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.savefig("Founding_Years",bbox_inches='tight')

df['Founded Year'] = founded1

#count of all companies with the given 3 acquisition statuses



plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Acquisition Status'].value_counts()



bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="GnBu_d")

sns.set_style("white")

plt.xlabel("M&A Status", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Distribution of M&A Status", y=1.02);



locs, labels = plt.xticks()

plt.setp(labels, rotation=0)

for i,j in zip(bar_plot.patches,acqstat.values):

  height = i.get_height()

  bar_plot.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.savefig('acq_status.png',bbox_inches='tight')
#count of all companies with the given funding statuses



plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Funding Status'].value_counts()



bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Funding Status", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with these funding status", y=1.02);



locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

for i,j in zip(bar_plot.patches,acqstat.values):

  height = i.get_height()

  bar_plot.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.show()

import plotly.express as px

fig = px.box(df, x="IPO status", y="Funding Amount", hover_data= ["IPO status"])

fig.show()
# plt.figure(figsize=(14,8))

# sns.set(font_scale=1.4)

# acqstat = df['Funding Amount'].value_counts()

# bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="OrRd")

# plt.title('Distribution of Funding Amount')

# plt.xlabel('Funding Amount')

# plt.ylabel('Company Count')

# for i, p in enumerate(bar_plot.patches):

#     height = p.get_height()

#     bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

# plt.savefig("Funding_Amount",bbox_inches='tight')





founded_y = []

founded1 =[]

leny = []

df['Exit Date'].fillna(" ", inplace = True)

for i in df['Exit Date']:

  i = i.split(' ')

  founded_y.append(i)



for i in founded_y:

  if(len(i)==3):

    founded1.append(i[2])

  elif(len(i)==2):

    founded1.append(i[1])

  elif(len(i)==1):

    founded1.append(i[0])





year_counts = np.unique(founded1,return_counts=True)





year_counts1 = year_counts[0][:-1]

year_counts1_c = year_counts[1][:-1]



year_counts1 = year_counts1[::-1]

year_counts1_c = [int(i) for i in year_counts1_c]

year_counts1_c = year_counts1_c[::-1]



year_counts1 = year_counts1[0:30]

year_counts1_c = year_counts1_c[0:30]





plt.figure(figsize=(14,8))

bar_year = sns.barplot(x =year_counts1, y = year_counts1_c,palette="YlOrRd")

sns.set_style("white")

plt.title('Distribution of different  exit date')

plt.xlabel('Years')

plt.ylabel('Counts of each year')

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)





for i,j in zip(bar_year.patches,year_counts1_c[::-1]):

  height = i.get_height()

  bar_year.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.show()

df['Exit Date'] = founded1
countries_cat_count = df['Number of employees'].value_counts()

countries_cat_count = countries_cat_count.sort_values(ascending = False)

print('There are a total of ',len(countries_cat_count),'different employee categories in our dataset')



countries_cat_count = countries_cat_count.iloc[:30]

print(countries_cat_count)



plt.figure(figsize=(14,8))

bar_countries = sns.barplot(x =countries_cat_count.index, y = countries_cat_count.values,palette="YlOrRd")

sns.set_style("white")

plt.title('Distribution of different employees')

plt.xlabel('Employees')

plt.ylabel('Counts of each Employee category')

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

for i,j in zip(bar_countries.patches,countries_cat_count.values):

  height = i.get_height()

  bar_countries.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.show()

countries_cat_count = df['Estimated Revenue'].value_counts()

countries_cat_count = countries_cat_count.sort_values(ascending = False)

print('There are a total of ',len(countries_cat_count),'different revenue categories in our dataset')



countries_cat_count = countries_cat_count.iloc[:30]

print(countries_cat_count)



plt.figure(figsize=(14,8))

bar_countries = sns.barplot(x =countries_cat_count.index, y = countries_cat_count.values,palette="YlOrRd")

sns.set_style("white")

plt.title('Distribution of different revenue')

plt.xlabel('revenue')

plt.ylabel('Counts of each revenue category')

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

for i,j in zip(bar_countries.patches,countries_cat_count.values):

  height = i.get_height()

  bar_countries.text(i.get_x() + i.get_width()/2, height + 5, j, ha='center', va='bottom')

plt.show()
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Founders'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Founders", labelpad=14)

plt.ylabel("Number of companies", labelpad=14)

plt.title("Count of companies with number of Founders", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.savefig('Founders',bbox_inches='tight')
# plt.figure(figsize=(14,8))

# sns.set(font_scale=1.4)

# acqstat = df['Acquisitions'].value_counts()

# acqstat

# bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

# sns.set_style("white")

# plt.xlabel("Acquisitions", labelpad=14)

# plt.ylabel("Count of companies", labelpad=14)

# plt.title("Count of companies with number of Acquisitions", y=1.02);

# for i, p in enumerate(bar_plot.patches):

#     height = p.get_height()

#     bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

# plt.show()
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Funding Rounds'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Funding Rounds", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with number of Funding Rounds", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.show()
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Active Products'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Active Products", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with number of Active Products", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.savefig("active_products", bbox_inches='tight')
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Nunmber of Lead Investors'] .value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Number of Lead Investors", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with Number of Lead Investors", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.show()

plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Nmber of Investors'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Number of Lead Investors", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with Number of Lead Investors", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.show()
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Founded Organization'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Founded Organization", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with number of Founded Organization", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.show()
plt.figure(figsize=(14,8))

sns.set(font_scale=1.4)

acqstat = df['Portfolio Companies'].value_counts()

acqstat

bar_plot = sns.barplot(x =acqstat.index, y = acqstat.values,palette="RdPu")

sns.set_style("white")

plt.xlabel("Portfolio Companies", labelpad=14)

plt.ylabel("Count of companies", labelpad=14)

plt.title("Count of companies with number of Portfolio Companies", y=1.02);

for i, p in enumerate(bar_plot.patches):

    height = p.get_height()

    bar_plot.text(p.get_x()+p.get_width()/2., height + 0.1, acqstat[i],ha="center")

plt.show()
df.to_excel("data_Model.xlsx")
df.isnull().sum()
df = pd.read_excel('/kaggle/input/datamodel/data_mod.xlsx')

data_model_df = df

df1 = data_model_df['Type'].str.get_dummies(sep=',')

df11 = pd.DataFrame(df1)

df2 = data_model_df['Job Title'].str.get_dummies(sep='&')

df22 = pd.DataFrame(df2)
data_model_df = data_model_df.join(df11)

data_model_df = data_model_df.join(df22)



# df.join(df2)
data_model_df = pd.get_dummies(data_model_df, columns=['Headquaters','Estimated Revenue','Founders',

                                   'Number of employees','Funding Rounds','Funding Status','Active Products','Funding Amount',

                                   'Nunmber of Lead Investors','Nmber of Investors','IPO status','Founded Organization',

                                    'Portfolio Companies','Founded Year','Exit Date'])

data_model_df.head()

data_model_df.head()
test_data = data_model_df

sample_df = data_model_df[data_model_df['Acquisition Status']=='WasAcquired'].sample(n=1100, random_state=1)

sample_df = sample_df.append(data_model_df[data_model_df['Acquisition Status']=='MadeAcquisitions']).sample(n=1000, random_state=1)

sample_df = sample_df.append(data_model_df[data_model_df['Acquisition Status']=='Did Not Participate'].sample(n=1500, random_state=1))

#np.unique(sample_df['Acquisition Status'],return_counts = True)

data_model_df = sample_df

# test_data = data_model_df

# sample_df = data_model_df[data_model_df['Acquisition Status']=='Did Not Participate'].sample(n=1750, random_state=1,replace=True)

# sample_df = sample_df.append(data_model_df[data_model_df['Acquisition Status']=='WasAcquired'].sample(n=1850, random_state=1,replace=True))

# sample_df = sample_df.append(data_model_df[data_model_df['Acquisition Status']=='MadeAcquisitions'].sample(n=1650, random_state=1,replace=True))

# #np.unique(sample_df['Acquisition Status'],return_counts = True)

# data_model_df = sample_df
data_model_df.shape
test_data.shape
# test_data = test_data.drop(sample_df.index)

# data_model_df = test_data 
Acquired_Status = data_model_df['Acquisition Status']

Acquired_price = data_model_df['Acquired Price']

Acquired_By = data_model_df['Acquired By']


data_model_df = data_model_df.drop(['Acquired Price'],axis = 1)

data_model_df = data_model_df.drop(['Acquired By'],axis = 1)

data_model_df = data_model_df.drop(['Acquisition Status'],axis = 1)

data_model_df = data_model_df.drop(['Acquisitions'],axis = 1)

data_model_df = data_model_df.drop(['Type'],axis = 1)

data_model_df = data_model_df.drop(['Job Title'],axis = 1)

data_model_df= data_model_df.drop(['Unnamed: 0'],axis=1)

data_model_df= data_model_df.drop(['Unnamed: 0.1'],axis=1)
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB as NB

from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import GradientBoostingClassifier as GB

from sklearn.metrics import average_precision_score

from sklearn.ensemble import AdaBoostClassifier as ab

from sklearn.tree import DecisionTreeClassifier as dt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import validation_curve

from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from xgboost import XGBClassifier



from sklearn.linear_model import SGDClassifier as sg

import matplotlib.pyplot as plt

import numpy as np
X_train, X_test, y_train, y_test = train_test_split(data_model_df, Acquired_Status, test_size=0.33, random_state=42)
y_train.value_counts()
y_test.value_counts()
model_name=['naive_bayes','RandomForestClassifier','KNeighborsClassifier','GradientBoostingClassifier','AdaBoostClassifier','DecisionTreeClassifier','LogisticRegression']

models_list= [NB(),RF(),KNN(),GB(),ab(),dt(),LR(max_iter=600)]

for i, j in zip(model_name, models_list):

    scores = cross_val_score(j, X_train, y_train, cv=5)

    

    print(i+"--"+ "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
train_sizes, train_scores, test_scores = learning_curve(LR(max_iter = 1000), 

                                                        X_test, 

                                                        y_test,

                                                        cv = 5,

                                                       

                                                        scoring='accuracy',

                                                        

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)





test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)





plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")





plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



plt.title("Learning Curve for Logistic Regression")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.savefig('logistic_reg.png', bbox_inches='tight')

train_sizes, train_scores, test_scores = learning_curve(NB(), 

                                                        X_test, 

                                                        y_test,

                                                        cv = 5,

                                                       

                                                        scoring='accuracy',

                                                        

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)





test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)





plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")





plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



plt.title("Learning Curve for Naive Bayes")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.savefig("NB.png",bbox_inches='tight')
train_sizes, train_scores, test_scores = learning_curve(dt(criterion = 'entropy' ,max_depth = 10), 

                                                        X_test, 

                                                        y_test,

                                                        cv = 5,

                                                       

                                                        scoring='accuracy',

                                                        

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)





test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)





plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")





plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



plt.title("Learning Curve for Decision Tree")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.savefig("DT.png",bbox_inches='tight')

train_sizes, train_scores, test_scores = learning_curve(ab(), 

                                                        X_test, 

                                                        y_test,

                                                        cv = 5,

                                                       

                                                        scoring='accuracy',

                                                        

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)





test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)





plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")





plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



plt.title("Learning Curve for Ada Boost")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.savefig('AdaBoost.png', bbox_inches='tight')
X = data_model_df

y = Acquired_Status
from itertools import cycle

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from sklearn.metrics import roc_auc_score







y_1 = label_binarize(Acquired_Status, classes=['Did Not Participate', 'MadeAcquisitions', 'WasAcquired'])

n_classes = y_1.shape[1]

print(n_classes)



random_state = np.random.RandomState(0)

n_samples, n_features = X.shape

# print(df.shape)

X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(X, y_1, test_size=0.33, random_state=42)



classifier = OneVsRestClassifier(dt(criterion = 'entropy' ,max_depth = 15))

y_score = classifier.fit(X_train_roc,y_train_roc).predict(X_test_roc)



fpr = dict()

tpr = dict()

roc_auc = dict()

yu = np.unique(Acquired_Status)

for i,v in enumerate(yu):

    fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.figure(figsize=(14,8))

lw = 2

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



#plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate',fontsize = 25)

plt.ylabel('True Positive Rate',fontsize = 25)

plt.title('ROC CURVE DECISION TREE',fontsize = 30)

plt.legend(loc="lower right",prop={'size': 23})

plt.savefig("DT_ROC.png",bbox_inches='tight')
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy' , random_state=42,max_depth = 25)

#scores = cross_val_score(X_train, y_train, cv=5)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

score = clf.score(X_test,y_test)



# plt.figure(figsize=(20,18))

# tree.plot_tree(clf)
# y_pred = clf.predict(data_model_df)

# y_test = Acquired_Status

# score = clf.score(data_model_df,y_test)
import seaborn as sns

def print_ConfusionMatrix(actual, pred,score,algo):

  cm = confusion_matrix(actual, pred)

  sns.set(font_scale=2)

  plt.figure(figsize=(9,9))

  sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlOrRd');

  plt.ylabel('Actual label');

  plt.xlabel('Predicted label');

  all_sample_title = 'Accuracy Score: {0}'.format(round(score,4))

  plt.title(all_sample_title)

  plt.savefig(algo, bbox_inches='tight')

 

print_ConfusionMatrix(y_test, y_pred,score,'decision_tree.png')
#plt.figure(figsize=(20,18))

#tree.plot_tree(clf)

clf = LR(random_state=0,max_iter = 1000).fit(X_train, y_train)

y_pred = clf.predict(X_test)

score = clf.score(X_test, y_test)

print_ConfusionMatrix(y_test, y_pred,score,"LR_CM.png")
#Random Forest confusion matrix

clf = RF(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)

score = clf.score(X_test, y_test)

print_ConfusionMatrix(y_test, y_pred,score,"RF.png")
# y_pred = clf.predict(data_model_df)

# y_test = Acquired_Status

# score = clf.score(data_model_df,y_test)

# print_ConfusionMatrix(y_test, y_pred,score,"RF.png")
from sklearn.preprocessing import label_binarize



y = Acquired_Status

y = label_binarize(Acquired_Status, classes=['Did Not Participate', 'MadeAcquisitions', 'WasAcquired'])

n_classes = y.shape[1]



random_state = np.random.RandomState(0)

n_samples, n_features = X.shape

# print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





classifier = OneVsRestClassifier(dt(criterion = 'entropy' ,max_depth = 15))

y_score = classifier.fit(X_train,y_train).predict(X_test)



from sklearn.multiclass import OneVsRestClassifier

# Run classifier

classifier = OneVsRestClassifier(dt(criterion = 'entropy' ,max_depth = 15))

y_score = classifier.fit(X_train,y_train).predict(X_test)



from sklearn.metrics import precision_recall_curve





# For each class

precision = dict()

recall = dict()

average_precision = dict()

for i in range(n_classes):

    precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], y_score[:,i])

    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])



# A "micro-average": quantifying score on all classes jointly

precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),

    y_score.ravel())

average_precision["micro"] = average_precision_score(y_test, y_score,

                                                     average="micro")

print('Average precision score, micro-averaged over all classes: {0:0.2f}'

      .format(average_precision["micro"]))

plt.figure()

plt.figure(figsize=(14,8))

plt.step(recall['micro'], precision['micro'], where='post')



plt.xlabel('Recall',fontsize = 25)

plt.ylabel('Precision',fontsize = 25)

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title(

    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'

    .format(average_precision["micro"]),fontsize=30)
from itertools import cycle

# setup plot details

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])



plt.figure(figsize=(14,8))

f_scores = np.linspace(0.2, 0.8, num=4)

lines = []

labels = []

for f_score in f_scores:

    x = np.linspace(0.01, 1)

    y = f_score * x / (2 * x - f_score)

    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)

    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)

labels.append('iso-f1 curves')

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)

lines.append(l)

labels.append('micro-average Precision-recall (area = {0:0.2f})'

              ''.format(average_precision["micro"]))



for i, color in zip(range(n_classes), colors):

    l, = plt.plot(recall[i], precision[i], color=color, lw=2)

    lines.append(l)

    labels.append('Precision-recall for class {0} (area = {1:0.2f})'

                  ''.format(i, average_precision[i]))



fig = plt.gcf()

fig.subplots_adjust(bottom=0.1)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Recall',fontsize=25)

plt.ylabel('Precision',fontsize = 25)

plt.title('Precision-Recall Class',fontsize=30)

plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# # Interaction terms

# poly = PolynomialFeatures(degree=2)

# X_train = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names(X.columns))

# X_test = pd.DataFrame(poly.transform(X_test), columns=poly.get_feature_names(X.columns))

# # Standardize data

# scaler = StandardScaler()

# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
# # Import your necessary dependencies

# from sklearn.feature_selection import RFE

# from sklearn.linear_model import LogisticRegression

# # Feature extraction

# model = LogisticRegression(max_iter = 1000)

# rfe = RFE(model, 14)

# fit = rfe.fit(X_train, y_train)
# from sklearn.feature_selection import RFECV

 

# # RFE

# from sklearn.svm import SVC

# clf = LR(random_state=0,max_iter = 1000)



# rfe = RFECV(estimator=clf, cv=4, scoring='accuracy')

# rfe = rfe.fit(X_train, y_train)

 

# # Select variables and calulate test accuracy

# cols = X_train.columns[rfe.support_]

# acc = accuracy_score(y_test, rfe.estimator_.predict(X_test[cols]))

# print('Number of features selected: {}'.format(rfe.n_features_))

# print('Test Accuracy {}'.format(acc))

 

# # Plot number of features vs CV scores

# plt.figure()

# plt.xlabel('k')

# plt.ylabel('CV accuracy')

# plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)

# plt.show()
# print(X.columns[fit.n_features_])

# idx = np.where(fit.support_ == 1)

# print(X.columns[idx])



# import matplotlib.pyplot as plt

# from sklearn.svm import SVC

# from sklearn.model_selection import StratifiedKFold

# from sklearn.feature_selection import RFECV

# from sklearn.datasets import make_classification







# # Create the RFE object and compute a cross-validated score.

# svc = SVC(kernel="linear")

# # The "accuracy" scoring is proportional to the number of correct

# # classifications

# rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),

#               scoring='accuracy')

# rfecv.fit(X, y)



# print("Optimal number of features : %d" % rfecv.n_features_)



# # Plot number of features VS. cross-validation scores

# plt.figure()

# plt.xlabel("Number of features selected")

# plt.ylabel("Cross validation score (nb of correct classifications)")

# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

# plt.show()
# select = rfecv.get_support()

# idx = np.where(select)

# print(X.columns[idx])
# data_model_best_subset_df = data_model_df[X.columns[idx]]

# data_model_best_subset_df.head()

# X_train, X_test, y_train, y_test = train_test_split(data_model_best_subset_df, Acquired_Status, test_size=0.33, random_state=42)
# model_name=['naive_bayes','RandomForestClassifier','KNeighborsClassifier','GradientBoostingClassifier','AdaBoostClassifier','DecisionTreeClassifier','LogisticRegression']

# models_list= [NB(),RF(),KNN(),GB(),ab(),dt(),LR(max_iter=600)]

# for i, j in zip(model_name, models_list):

#     scores = cross_val_score(j, X_train, y_train, cv=5)

    

#     print(i+"--"+ "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
dfold = pd.read_excel('/kaggle/input/modelknn/Data_knn.xlsx')

df = pd.read_excel('/kaggle/input/datamodel/data_mod.xlsx')



sample_df = df[df['Acquisition Status']=='WasAcquired']

sample_df = sample_df.append(df[df['Acquisition Status']=='MadeAcquisitions'])

sample_df = sample_df.append(df[df['Acquisition Status']=='Did Not Participate'].sample(n=1500, random_state=1))

#np.unique(sample_df['Acquisition Status'],return_counts = True)

df = sample_df



extra = df

extra.shape
df= df.drop(['Unnamed: 0'],axis=1)

df = df.drop(['Unnamed: 0.1'],axis = 1)

df = df.drop(['Job Title'],axis = 1)

df = df.drop(['Acquired Price'],axis = 1)

df = pd.get_dummies(df, columns=['Headquaters','Estimated Revenue','Founders',

                                   'Number of employees','Funding Rounds','Funding Status','Active Products','Funding Amount',

                                   'Nunmber of Lead Investors','Nmber of Investors','IPO status','Founded Organization','Acquired By','Acquisition Status','Acquisitions',

                                    'Portfolio Companies','Founded Year','Exit Date'])
df1 = df['Type'].str.get_dummies(sep=',')

df11 = pd.DataFrame(df1)
df = df.drop(['Type'],axis = 1)
df = df.join(df11)

extra.dtypes

extra= extra.drop(['Unnamed: 0'],axis=1)

extra = extra.drop(['Unnamed: 0.1'],axis = 1)

extra = extra.drop(['Founded Year'],axis = 1)

extra = extra.drop(['Exit Date'],axis = 1)

extra = extra.drop(['Job Title'],axis = 1)

extra = extra.drop(['Acquired Price'],axis = 1)

extra = extra.drop(['Acquired By'],axis = 1)

extra = extra.drop(['Acquisitions'],axis = 1)

extra.isnull().sum()
extra['Headquaters'].mode()
extra['Headquaters'] = extra['Headquaters'].fillna('North America')
#kmodes

from kmodes.kmodes import KModes



km = KModes(n_clusters=8, init='Huang', n_init=8, verbose=1)

clusters = km.fit_predict(extra)





# Print the cluster centroids

print(km.cluster_centroids_)





#ploting

cost = pd.DataFrame(km.epoch_costs_, columns=['cost'])

cc= pd.DataFrame(range(len(km.epoch_costs_)), columns=['index'])

cost1 = cc.join(cost)

print(cost1)
cost.plot.line()
from scipy.sparse import csr_matrix



movie_features_df_matrix = csr_matrix(df.values)

# movie_features_df_matrix.shape

from sklearn.neighbors import NearestNeighbors





model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute',radius = 1.5)

model_knn.fit(movie_features_df_matrix)
sectors=dfold.iloc[:,1:3]

query_index = np.random.choice(df.shape[0])

print(sectors.loc[query_index]['Name'])

print(sectors.loc[query_index]['Type'])

distances, indices = model_knn.kneighbors(df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 4)

distances = distances[0][1:6]

indices = indices[0][1:6]

print(distances,indices)
print('Prediction for : ',sectors.loc[query_index]['Name'])

print('Sector :',sectors.loc[query_index]['Type'])

print('---------------------------------------------')

print('---------------------------------------------')

for i in indices:

  print('Company it could participate with is : ')

  print('{0} :'.format(sectors.iloc[i]['Name']))

  print('Sector :{0}'.format(sectors.iloc[i]['Type']))

  print('---------------------------------------------')