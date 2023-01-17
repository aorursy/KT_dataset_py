import pandas as pd

from datetime import datetime

import re

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# 1. Initial data exploration using shape, describe() and info()



file_path = "/kaggle/input/indian-startup-funding/startup_funding.csv"

data = pd.read_csv(file_path)

data.head()
data.shape
data.columns
data.info()
# 2. The data types of data in few columns are not correct to do statistical analysis, hence it needs to be fixed using dtype, astype()



data.rename(columns={"Date dd/mm/yyyy": "Date", 'Startup Name': 'StartupName', 'Industry Vertical':'IndustryVertical',

       'SubVertical': 'SubVertical', 'City  Location': 'CityLocation', 'Investors Name': 'InvestorsName', 'InvestmentnType':'InvestmentnType',

       'Amount in USD':'AmountUSD', 'Remarks':'Remarks'}, inplace=True)

data.head()
# 3. Few values 'Date' column are not in correcdt date format checking those rows

data[~data.Date.str.contains('(\d{2})[/](\d{2})[/](\d{4})')]
#incorrect_date = data[~data.Date.str.contains('(\d{2})[/](\d{2})[/](\d{4})')].index # [192, 2571, 2606, 2775, 2776, 2831, 3011, 3029]

data.loc[ 192, 'Date'] = '05/07/2018' 

data.loc[2571, 'Date'] = '01/07/2015' 

data.loc[2606, 'Date'] = '10/07/2015'

data.loc[2775, 'Date'] = '12/05/2015' 

data.loc[2776, 'Date'] = '12/05/2015'

data.loc[2831, 'Date'] = '13/04/2015' 

data.loc[3011, 'Date'] = '15/01/2015'

data.loc[3029, 'Date'] = '22/01/2015'
# 4. Extracting date, month, year from string values in 'Date' column



date_expand = data['Date'].str.extract(r'(\d{2})/?(\d{2})/?(\d{4})')

data['Year'] = date_expand[2]

data['Month'] = date_expand[1]

data['NewDate'] = date_expand[0]+'/'+date_expand[1]+'/'+date_expand[2]

data.head()
data['Date'] = pd.to_datetime(data['Date'])#['Date']

data.head()
# 5. Converting datatype of values in 'AmountUSD' column from string to float. Marking Undisclosed values to 'nan' and then converting into float type



data.loc[data['AmountUSD'].isin(['undisclosed', 'unknown', 'Undisclosed']), 'AmountUSD'] = 'nan'



data['AmountUSD'] = data['AmountUSD'].astype(str)

data['NewAmountUSD'] = data['AmountUSD'].apply(lambda x : re.sub("[^0-9]", "", x))

data.loc[data['NewAmountUSD']=='', 'NewAmountUSD'] = 0 #'nan' # replace with average of funding provided that months 

data['NewAmountUSD'] = data['NewAmountUSD'].astype(float)

data.head()
# 6. Cleaning column 'CityLocation'



data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Noida', '\\xc2\\xa0Noida']), 'CityLocation'] = 'Noida'

data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Bangalore', '\\xc2\\xa0Bangalore', 'Bangalore']), 'CityLocation'] = 'Bengaluru'

data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0New Delhi', '\\xc2\\xa0New Delhi']), 'CityLocation'] = 'New Delhi'

data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Gurgaon', 'Gurugram']), 'CityLocation'] = 'Gurgaon'

data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Mumbai', '\\xc2\\xa0Mumbai']), 'CityLocation'] = 'Mumbai'

# len(data['CityLocation'].unique())
# 7. Cleanning column 'IndustryVertical'



data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0News Aggregator mobile app", 'IndustryVertical'] = 'News Aggregator mobile app'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Jewellery Store", 'IndustryVertical'] = 'Online Jewellery Store'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Fashion Info Aggregator App", 'IndustryVertical'] = 'Fashion Info Aggregator App'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Study Notes Marketplace", 'IndustryVertical'] = 'Online Study Notes Marketplace'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Warranty Programs Service Administration", 'IndustryVertical'] = 'Warranty Programs Service Administration'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Pre-School Chain", 'IndustryVertical'] = 'Pre-School Chain'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Premium Loyalty Rewards Point Management", 'IndustryVertical'] = 'Premium Loyalty Rewards Point Management'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Contact Center Software Platform", 'IndustryVertical'] = 'Contact Center Software Platform'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Casual Dining restaurant Chain", 'IndustryVertical'] = 'Casual Dining restaurant Chain'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Grocery Delivery", 'IndustryVertical'] = 'Online Grocery Delivery'

data.loc[data['IndustryVertical'] == "Online home d\\\\xc3\\\\xa9cor marketplace", 'IndustryVertical'] = 'Online home decor marketplace'

data.loc[data['IndustryVertical'].isin(["ECommerce", "E-Commerce", "E-commerce", "Ecommerce"]), 'IndustryVertical'] = 'eCommerce'

data.loc[data['IndustryVertical'].isin(["Fin-Tech"]), 'IndustryVertical'] = 'FinTech'
# 8. Cleanning column 'InvestorsName'



data.loc[data['InvestorsName'].isin(['Undisclosed investors', 'Undisclosed', 'undisclosed investors', 'Undisclosed Investor', 'Undisclosed investors']), 'InvestorsName'] = 'Undisclosed Investors'

data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0Tiger Global", 'InvestorsName'] = 'Tiger Global'

data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0IndianIdeas.com", 'InvestorsName'] = 'IndianIdeas'

data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0IvyCap Ventures, Accel Partners, Dragoneer Investment Group", 'InvestorsName'] = 'IvyCap Ventures, Accel Partners, Dragoneer Investment Group'

data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0Goldman Sachs", 'InvestorsName'] = 'Goldman Sachs'
startup_data = data[['Date', 'Year', 'Month', 'StartupName', 'IndustryVertical', 'SubVertical', 'CityLocation', 'InvestorsName', 'InvestmentnType', 'NewAmountUSD']]

startup_data['Date'] = pd.to_datetime(startup_data.Date)

startup_data.set_index('Date', inplace=True)

startup_data.head()
funding_count_yr = pd.DataFrame(startup_data['Year'].value_counts())

funding_count_yr.rename(columns={"Year":"Number of Fundings"}, inplace=True)

funding_count_yr
funding_count_qtr = pd.DataFrame(data=startup_data['Year'].resample('QS').count())

funding_count_qtr.rename(columns={'Year':'Number of Fundings(Qtr)'}, inplace=True)

funding_count_qtr['QtrMonth'] = ['2015-1', '2015-4', '2015-7', '2015-10', '2016-1', '2016-4', '2016-7', '2016-10', '2017-1', '2017-4', '2017-7', '2017-10', '2018-1', '2018-4', '2018-7', '2018-10', '2019-1', '2019-4', '2019-7', '2019-10', '2020-1', '2020-4', '2020-7', '2020-10']

funding_count_qtr.head()
funding_total_yr = pd.DataFrame(startup_data.groupby(by=['Year'])['NewAmountUSD'].sum())

funding_total_yr.rename(columns={"NewAmountUSD":"Total Funding(USD-Bn)"}, inplace=True)

funding_total_yr = funding_total_yr.sort_values(by='Total Funding(USD-Bn)', ascending=False)

funding_total_yr
funding_total_qtr = pd.DataFrame(data=startup_data['NewAmountUSD'].resample('QS').sum())

funding_total_qtr.rename(columns={'NewAmountUSD':'Total Fundings(Qtr USD-Bn)'}, inplace=True)

funding_total_qtr['QtrMonth'] = ['2015-1', '2015-4', '2015-7', '2015-10', '2016-1', '2016-4', '2016-7', '2016-10', '2017-1', '2017-4', '2017-7', '2017-10', '2018-1', '2018-4', '2018-7', '2018-10', '2019-1', '2019-4', '2019-7', '2019-10', '2020-1', '2020-4', '2020-7', '2020-10']

funding_total_qtr.head()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,8))



sns.barplot(x=funding_count_yr.index, y=funding_count_yr['Number of Fundings'], data=funding_count_yr, ax=axes[0,0], orient='v')

sns.barplot(x=funding_count_qtr.index, y=funding_count_qtr['Number of Fundings(Qtr)'], data=funding_count_qtr, ax=axes[0,1], orient='v').set_xticklabels(rotation=90, labels=funding_total_qtr['QtrMonth'])



sns.barplot(x=funding_total_yr.index, y=funding_total_yr['Total Funding(USD-Bn)'], data=funding_total_yr, ax=axes[1,0], orient='v')

sns.barplot(x=funding_total_qtr.index, y=funding_total_qtr['Total Fundings(Qtr USD-Bn)'], data=funding_total_qtr, ax=axes[1,1], orient='v').set_xticklabels(rotation=90, labels=funding_total_qtr['QtrMonth'])



fig.tight_layout(pad=3)

plt.show()
fundings_count_city = pd.DataFrame(startup_data['CityLocation'].value_counts().sort_values(ascending=False)[:10])

fundings_count_city.rename(columns={'CityLocation':'Number of Fundings by City'}, inplace=True)

fundings_count_city.head()
funding_total_city = pd.DataFrame(startup_data.groupby('CityLocation')['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]

funding_total_city.rename(columns={'NewAmountUSD':'Total Funding by City(USD-Bn)'}, inplace=True)

funding_total_city.head()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))



sns.barplot(x=fundings_count_city.index, y=fundings_count_city['Number of Fundings by City'], data=fundings_count_city, ax=axes[0])

sns.barplot(x=funding_total_city.index, y=funding_total_city['Total Funding by City(USD-Bn)'], data=funding_total_city, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_city.index)



fig.tight_layout(pad=0.5)

plt.show()
fundings_count_industry = pd.DataFrame(startup_data['IndustryVertical'].value_counts().sort_values(ascending=False))[:15]

fundings_count_industry.rename(columns={'IndustryVertical':'Number of Fundings by Industry'}, inplace=True)

fundings_count_industry.head()
funding_total_industry = pd.DataFrame(startup_data.groupby('IndustryVertical')['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]

funding_total_industry.rename(columns={'NewAmountUSD':'Total Funding by Industry(USD-Bn)'}, inplace=True)

funding_total_industry.head()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))



sns.barplot(x=fundings_count_industry.index, y=fundings_count_industry['Number of Fundings by Industry'], data=fundings_count_industry, ax=axes[0]).set_xticklabels(rotation=90, labels=fundings_count_industry.index)

sns.barplot(x=funding_total_industry.index, y=funding_total_industry['Total Funding by Industry(USD-Bn)'], data=funding_total_industry, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_industry.index)



fig.tight_layout(pad=1)

#plt.xticks(rotation=90)

plt.show()
funding_count_investor = pd.DataFrame(startup_data['InvestorsName'].value_counts()).sort_values(by='InvestorsName', ascending=False)[:10]

funding_count_investor.rename(columns={'InvestorsName': 'Number of Investments by Investor'}, inplace=True)

funding_count_investor.drop(funding_count_investor[funding_count_investor.index == 'Undisclosed Investors'].index, inplace=True)

funding_count_investor.head()
funding_total_investor = pd.DataFrame(startup_data.groupby(['InvestorsName'])['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]

funding_total_investor.rename(columns={'NewAmountUSD':'Total Funding by Investor(USD-Bn)'}, inplace=True)

funding_total_investor.head()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))



sns.barplot(x=funding_count_investor.index, y=funding_count_investor['Number of Investments by Investor'], data=funding_count_investor, ax=axes[0]).set_xticklabels(rotation=90, labels=funding_count_investor.index)

sns.barplot(x=funding_total_investor.index, y=funding_total_investor['Total Funding by Investor(USD-Bn)'], data=funding_total_investor, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_investor.index)



fig.tight_layout(pad=1)

plt.show()
startup_data[startup_data['InvestorsName'].isin(['Westbridge Capital', 'Softbank'])]
funding_count_company = pd.DataFrame(startup_data['StartupName'].value_counts()).sort_values(by='StartupName', ascending=False)[:15]

funding_count_company.rename(columns={'StartupName': 'Number of Investments by Investor'}, inplace=True)

funding_count_company.head()
funding_total_company = pd.DataFrame(startup_data.groupby('StartupName')['NewAmountUSD'].sum()).sort_values(by='NewAmountUSD', ascending=False)[:15]

funding_total_company.rename(columns={'NewAmountUSD': "Total amount Raised by Startup (USD-Bn)"}, inplace=True)

funding_total_company.head()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))



sns.barplot(x=funding_count_company.index, y=funding_count_company['Number of Investments by Investor'], data=funding_count_company, ax=axes[0]).set_xticklabels(rotation=90, labels=funding_count_company.index)

sns.barplot(x=funding_total_company.index, y=funding_total_company['Total amount Raised by Startup (USD-Bn)'], data=funding_total_company, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_company.index)



fig.tight_layout(pad=1)

plt.show()
plt.figure(figsize=(25,8))

sns.distplot(startup_data.loc[startup_data['NewAmountUSD']<=10000000.0, 'NewAmountUSD'])

plt.show()
funding_average = startup_data['NewAmountUSD'].mean()

funding_meadian = startup_data['NewAmountUSD'].median()

print(funding_average, funding_meadian)