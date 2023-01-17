import pandas as pd

import numpy as np

compdf = pd.read_csv("companies.csv")
compdf= pd.read_csv("companies.csv", low_memory=False)
compdf.head()
compdf.columns
compdf = compdf.drop(['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',

       'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17'], axis=1)
compdf.head()
compdf.count()
compdf['permalink'].nunique()
import numpy as np

import pandas as pd



r2df= pd.read_csv("rounds2.csv")
import numpy as np

import pandas as pd



r2df= pd.read_csv("rounds2.csv", sep= ' ', delimiter = ",", engine='python')
r2df.head()
r2df['company_permalink'].str.lower()
r2df['company_permalink'].nunique()
r2df['company_permalink'].str.lowercase.nunique()
r2df['company_permalink'].str.lower.nunique()
r2df['company_permalink'].str.lower().nunique()
import numpy as np

import pandas as pd

rou2df = pd.read_csv('rounds2.csv',encoding='ISO-8859-1')
rou2df.head()
rou2df['company_permalink'].lower.str()
rou2df['company_permalink'].str.lower()
rou2df['company_permalink'].str.lower().nunique()
rou2df['company_permalink'].str.lower().unique()
print(len(rou2df['company_permalink'].str.lower().unique()))
rou2df.company_permalink = rou2df.company_permalink.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

rou2df.head()
print(len(rou2df['company_permalink'].str.lower().unique()))
rou2df['company_permalink'] = rou2df['company_permalink'].str.lower()

print(len(rou2df['company_permalink'].unique()))
df1 = pd.DataFrame(compdf.permalink.unique())

df2 = pd.DataFrame(rou2df.company_permalink.unique())
df2.equals(df1)
df1.equals(df2)
compdf['permalink'] = compdf['permalink'].str.lower()

print(len(compdf['permalink'].unique()))
df1 = pd.DataFrame(compdf.permalink.unique())

df2 = pd.DataFrame(rou2df.company_permalink.unique())
df2.equals(df1)
df1.equals(df2)
import pandas as pd

import numpy as np

companiesdf = pd.read_csv("companies.csv", encoding='ISO-8859-1',sep='\t')
companiesdf.head()
companiesdf.permalink = companiesdf.permalink.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

companiesdf.name = companiesdf.name.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

companiesdf.head()
#let us use only the txt file instead of using csv file that we have converted.

import pandas as pd

import numpy as np

companiesdf = pd.read_csv("companies.txt", encoding='ISO-8859-1',sep='\t')



companiesdf.permalink = companiesdf.permalink.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

companiesdf.name = companiesdf.name.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

companiesdf.head()
# How many unique companies are present in companies?

companiesdf['permalink'] = companiesdf['permalink'].str.lower()

print(len(companiesdf['permalink'].unique()))



cdf1 = pd.DataFrame(companiesdf.permalink.unique())

rdf2 = pd.DataFrame(rou2df.company_permalink.unique())
cdf1.equals(rdf2)
set(companiesdf['permalink'].unique()).difference(set(rou2df['company_permalink'].unique()))
master_frame = pd.merge(rou2df, companiesdf, how = 'left', left_on = 'company_permalink', right_on = 'permalink')
master_frame.columns()
master_frame.columns
master_frame.head()
len(master_frame)
master_frame.isnull().sum(axis=0)
print(round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2))
master_frame = master_frame.drop(['funding_round_code', 'funding_round_permalink', 'funded_at','permalink', 'homepage_url',

                                 'state_code', 'region', 'city', 'founded_at','status'], axis = 1)
#Now again let us find the percentage of Null values in the master_frame.

print(round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2))
#Dropping rows based on null columns

master_frame = master_frame[~(master_frame['raised_amount_usd'].isnull() | master_frame['country_code'].isnull() |

                             master_frame['category_list'].isnull())]
#let us find the percentage of retained rows

print(100*(len(master_frame.index)/114949))
len(master_frame.index)

master_frame.shape
master_frame.funding_round_type.value_counts()
master_frame['funding_round_type'].counts()
master_frame['funding_round_type'].count()
master_frame = master_frame.drop(['debt_financing', 'grant', 'undisclosed', 'convertible_note', 'equity_crowdfunding', 'post_ipo_equity', 'product_crowdfunding', 'post_ipo_debt', 'non_equity_assistance', 'secondary_market' ])
master_frame = master_frame[funding_round_type].drop(['debt_financing', 'grant', 'undisclosed', 'convertible_note', 'equity_crowdfunding', 'post_ipo_equity', 'product_crowdfunding', 'post_ipo_debt', 'non_equity_assistance', 'secondary_market' ])
master_frame = master_frame[(master_frame['funding_round_type'] == 'venture') 

                            | (master_frame['funding_round_type'] == 'seed')

                            | (master_frame['funding_round_type'] == 'angel')

                            | (master_frame['funding_round_type'] == 'private_equity')]
master_frame.funding_round_type.value_counts()
master_frame['raised_amount_usd'] = master_frame['raised_amount_usd']/1000000

master_frame.head()
round(master_frame.groupby('funding_round_type').raised_amount_usd.mean(), 2)
master_frame = master_frame[master_frame['funding_round_type'] == 'venture'] 
master_frame.columns
master_frame.head()
top9 = master_frame.pivot_table(values = 'raised_amount_usd', index = 'country_code', aggfunc = 'sum')
top9.head()
masterframe_df = pd.merge(rou2df, companiesdf, how = 'left', left_on = 'company_permalink', right_on = 'permalink')
top_9 = masterframe_df.pivot_table(values = 'raised_amount_usd', index = 'country_code', aggfunc = 'sum')
top_9.head()
masterframe_df.head()
top9 = top9.sort_values(by = 'raised_amount_usd', ascending = False)

top9 = top9.iloc[:9, ]

top9
top_9 = top_9.sort_values(by = 'raised_amount_usd', ascending = False)

top_9 = top_9.iloc[:9, ]

top_9
master_frame = master_frame[(master_frame['country_code'] == 'USA')

                            | (master_frame['country_code'] == 'GBR')

                            | (master_frame['country_code'] == 'IND')]
master_frame.head()
master_frame['category_list'] = master_frame['category_list'].apply(lambda x: x.split('|')[0])
import numpy as np

import pandas as pd

mappdf = pd.read_csv('mapping.csv')
mappdf.head()
mappdf.category_list = mappdf.category_list.replace({'0':'na', '2.na' :'2.0'}, regex=True)

mappdf.head()
mappdf = pd.melt(mappdf, id_vars =['category_list'], value_vars =['Manufacturing','Automotive & Sports',

                                                              'Cleantech / Semiconductors','Entertainment',

                                                             'Health','News, Search and Messaging','Others',

                                                             'Social, Finance, Analytics, Advertising']) 

mappdf = mappdf[~(mappdf.value == 0)]

mappdf = mappdf.drop('value', axis = 1)

mappdf = mappdf.rename(columns = {"variable":"main_sector"})

mappdf.head()
master_frame = master_frame.merge(mappdf, how = 'left', on ='category_list')

master_frame.head()
#List of primary sectors which have no main sectors in the master_frame

print(master_frame[master_frame.main_sector.isnull()].category_list.unique())
#Number of rows with NaN masin_sector value

len(master_frame[master_frame.main_sector.isnull()])
D1 = master_frame[(master_frame['country_code'] == 'USA') & 

             (master_frame['raised_amount_usd'] >= 5) & 

             (master_frame['raised_amount_usd'] <= 15)]

D1.head()
D1gr = D1[['raised_amount_usd','main_sector']].groupby('main_sector').agg(['sum', 'count']).rename(

    columns={'sum':'Total_amount','count' : 'Total_count'})

D1gr.head()
D1 = D1.merge(D1gr, how='left', on ='main_sector')

D1.head()
frame = pd.DataFrame(D1gr).reset_index(col_level=1)



frame.columns = frame.columns.get_level_values(1)

frame.head()
D1 = D1.merge(frame, how='left', on ='main_sector')

D1.head()
D2 = master_frame[(master_frame['country_code'] == 'GBR') & 

             (master_frame['raised_amount_usd'] >= 5) & 

             (master_frame['raised_amount_usd'] <= 15)]
D2gr = D2[['raised_amount_usd','main_sector']].groupby('main_sector').agg(['sum', 'count']).rename(

    columns={'sum':'Total_amount','count' : 'Total_count'})
D2gr.head()
frame1 = pd.DataFrame(D2gr).reset_index(col_level=1)



frame1.columns = frame1.columns.get_level_values(1)

frame1.head()
D2 = D2.merge(frame1, how='left', on ='main_sector')

D2.head()
D3 = master_frame[(master_frame['country_code'] == 'IND') & 

             (master_frame['raised_amount_usd'] >= 5) & 

             (master_frame['raised_amount_usd'] <= 15)]
D3gr = D3[['raised_amount_usd','main_sector']].groupby('main_sector').agg(['sum', 'count']).rename(

    columns={'sum':'Total_amount','count' : 'Total_count'})

D3gr.head()
frame2 = pd.DataFrame(D3gr).reset_index(col_level=1)



frame2.columns = frame2.columns.get_level_values(1)

frame2.head()
D3 = D3.merge(frame2, how='left', on ='main_sector')

D3.head()
#Total number of investments (count)

print(D1.raised_amount_usd.count())

print(D2.raised_amount_usd.count())

print(D3.raised_amount_usd.count())
#Total amount of investment (USD)

print(round(D1.raised_amount_usd.sum(), 2))

print(round(D2.raised_amount_usd.sum(), 2))

print(round(D3.raised_amount_usd.sum(), 2))
#Top sector, second-top, third-top for D1 (based on count of investments)

#Number of investments in the top, second-top, third-top sector in D1

frame
#Top sector, second-top, third-top for D2 (based on count of investments)

#Number of investments in the top, second-top, third-top sector in D2

frame1
#Top sector, second-top, third-top for D2 (based on count of investments)

#Number of investments in the top, second-top, third-top sector in D3

frame2
#For the top sector USA , which company received the highest investment?

company = D1[D1['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector USA , which company received the highest investment?

company = D1[D1['main_sector']=='Social, Finance, Analytics, Advertising']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))

#For the top sector GBR , which company received the highest investment?

company = D2[D2['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector GBR , which company received the highest investment?

company = D2[D2['main_sector']=='Social, Finance, Analytics, Advertising']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))
#For the top sector IND , which company received the highest investment?

company = D3[D3['main_sector']=='Others']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))



#For the second top sector IND , which company received the highest investment?

company = D3[D3['main_sector']=='News, Search and Messaging']

company = company.pivot_table(values = 'raised_amount_usd', index = 'company_permalink', aggfunc = 'sum')

company = company.sort_values(by = 'raised_amount_usd', ascending = False).head()

print(company.head(1))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))



sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master_frame)

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))



sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=D1)

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))



sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=D2)

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))



sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=D3)

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))



sns.boxplot(x='main_sector', y='Total_count', data=frame)

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.figure(figsize=(14, 8))



sns.boxplot(x='main_sector', y='Total_amount', data=frame)

plt.show()