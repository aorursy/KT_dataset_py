import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



#Load data -- cleaned date typos in text file

Raw_df = pd.read_csv("../input/startup_funding.csv")
#Data cleaning

#Make dates all same format

Raw_df['Date']=Raw_df['Date'].replace('12/05.2015','12/05/2015')

Raw_df['Date']=Raw_df['Date'].replace('13/04.2015','13/04/2015')

Raw_df['Date']=Raw_df['Date'].replace('15/01.2015','15/01/2015')

Raw_df['Date']=Raw_df['Date'].replace('22/01//2015','22/01/2015')

#Convert date column to date format

Raw_df['Date'] = pd.to_datetime(Raw_df['Date'], format='%d/%m/%Y')

#Make YrMo and Year columns for easy aggregates

Raw_df['YrMo'] = [str(x.year) + '-' + str(x.month).zfill(2) for x in Raw_df['Date']]

Raw_df['Year'] = [str(x.year) for x in Raw_df['Date']]



#Make investment amounts a number -- make empty (NaN) values zero and convert to $M

Raw_df['AmountInUSD'] = [float(str(x).replace(',',''))/1e6 for x in Raw_df['AmountInUSD']]

Raw_df.loc[np.isnan(Raw_df['AmountInUSD']), 'AmountInUSD'] = 0



#Fix InvestmentType

Raw_df['InvestmentType']=Raw_df['InvestmentType'].replace('Crowd funding','Crowd Funding')

Raw_df['InvestmentType']=Raw_df['InvestmentType'].replace('PrivateEquity','Private Equity')

Raw_df['InvestmentType']=Raw_df['InvestmentType'].replace('SeedFunding','Seed Funding')



#Fix city where possible -- if city is blank or it's outside India, assume "India"

Raw_df['CityLocation'] = ['India' if type(x)==float else x for x in Raw_df['CityLocation']]

Raw_df['CityMod']=Raw_df['CityLocation'].replace('Bangalore / Palo Alto','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Bangalore / San Mateo','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Bangalore/ Bangkok','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Bangalore / SFO','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Bangalore / USA','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('bangalore','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Chennai/ Singapore','Chennai')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Dallas / Hyderabad','Hyderabad')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Hyderabad/USA','Hyderabad')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Gurgaon / SFO','Gurgaon')

Raw_df['CityMod']=Raw_df['CityMod'].replace('India / US','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('London','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Mumbai / Global','Mumbai')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Mumbai / NY','Mumbai')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Mumbai / UK','Mumbai')

Raw_df['CityMod']=Raw_df['CityMod'].replace('New Delhi/ Houston','New Delhi')

Raw_df['CityMod']=Raw_df['CityMod'].replace('New Delhi / US','New Delhi')

Raw_df['CityMod']=Raw_df['CityMod'].replace('New Delhi / California','New Delhi')

Raw_df['CityMod']=Raw_df['CityMod'].replace('New York/ India','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Noida / Singapore','Noida')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Pune / Dubai','Pune')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Pune / US','Pune')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Pune/Seattle','Pune')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Pune / Singapore','Pune')

Raw_df['CityMod']=Raw_df['CityMod'].replace('Seattle / Bangalore','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('SFO / Bangalore','Bangalore')

Raw_df['CityMod']=Raw_df['CityMod'].replace('US/India','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('USA/India','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('US','India')

Raw_df['CityMod']=Raw_df['CityMod'].replace('USA','India')



#Fix industry where possible

Raw_df['IndustryVertical']=Raw_df['IndustryVertical'].replace('ECommerce','eCommerce')

Raw_df['IndustryVertical']=Raw_df['IndustryVertical'].replace('E-Commerce & M-Commerce platform','eCommerce')

Raw_df['IndustryVertical']=Raw_df['IndustryVertical'].replace('Ecommerce Marketplace','Online Marketplace')
#Analyses

#Count # of investments as function of month-yr

PosFund_df = Raw_df.loc[Raw_df['AmountInUSD'] > 0,['YrMo','AmountInUSD']]

fundCountsByMo = PosFund_df['YrMo'].value_counts()

FundCounts_df = pd.DataFrame(fundCountsByMo).sort_index()

ax = sns.barplot(FundCounts_df.YrMo, FundCounts_df.index, orient="h", color='b')

ax.set(xlabel='Investment Count', title='# of Startup Investments ($>0)')

sns.plt.show()
#Calculate # invested each month

FundTotal_df = Raw_df[['YrMo','AmountInUSD']].groupby('YrMo').sum()

ax = sns.barplot(FundTotal_df.AmountInUSD, FundTotal_df.index, orient="h", color='b')

ax.set(xlabel='Investment $M', title='Total $M Invested over time')

sns.plt.show()
#Now break it down by type of funding

#Make df such there's an entry for each investment type

FundTotal_df = pd.pivot_table(Raw_df,index=['YrMo'],values=['AmountInUSD'],columns=['InvestmentType'], aggfunc=np.sum, fill_value=0)['AmountInUSD'].reset_index()



fig, ax = plt.subplots()

ax.plot(FundTotal_df['Crowd Funding'], 'b', label='Crowd Funding')

ax.plot(FundTotal_df['Debt Funding'], 'r', label='Debt Funding')

ax.plot(FundTotal_df['Seed Funding'], 'k', label='Seed Funding')

ax.plot(FundTotal_df['Private Equity'], 'g', label='Private Equity')

ax.set(xlabel='Months from Jan 2015', ylabel='Investment $M', title='Total $M Invested over time by source type')

legend = ax.legend(loc='upper center', shadow=True)

plt.show()



fig,ax = plt.subplots()

ax.plot(FundTotal_df['Crowd Funding'], 'b', label='Crowd Funding')

ax.plot(FundTotal_df['Debt Funding'], 'r', label='Debt Funding')

ax.plot(FundTotal_df['Seed Funding'], 'k', label='Seed Funding')

ax.set(xlabel='Months from Jan 2015', ylabel='Investment $M', title='Total $M Invested over time by source type')

legend = ax.legend(loc='upper center', shadow=True)

plt.show()

#Now break down investment counts by type of funding

#Make df such there's an entry for each investment type

FundTotal_df = pd.pivot_table(Raw_df,index=['YrMo'],values=['AmountInUSD'],columns=['InvestmentType'], aggfunc='count', fill_value=0)['AmountInUSD'].reset_index()



fig, ax = plt.subplots()

ax.plot(FundTotal_df['Crowd Funding'], 'b', label='Crowd Funding')

ax.plot(FundTotal_df['Debt Funding'], 'r', label='Debt Funding')

ax.plot(FundTotal_df['Seed Funding'], 'k', label='Seed Funding')

ax.plot(FundTotal_df['Private Equity'], 'g', label='Private Equity')

ax.set(xlabel='Months from Jan 2015', ylabel='Investment $M', title='# Investments over time by source type')

legend = ax.legend(loc='upper right', shadow=True)

plt.show()
#Now break by city / region

#FundTotalCity shows >99% of investment is within the top 10 cities

CityTotals_df = Raw_df[['CityMod','AmountInUSD']].groupby('CityMod').sum().sort_values('AmountInUSD',ascending=False)

CityList = list(CityTotals_df.index[:5])

FundTotalCity_df = Raw_df[['YrMo','CityMod','AmountInUSD']]

FundTotalCity_df.loc[:,'CityMod'] = [x if x in CityList else 'Other' for x in FundTotalCity_df['CityMod']]

FundTotalCity_df = pd.pivot_table(FundTotalCity_df,index=['YrMo'],values=['AmountInUSD'],columns=['CityMod'], aggfunc=np.sum, fill_value=0)['AmountInUSD'].reset_index()



fig, ax = plt.subplots()

ax.plot(FundTotalCity_df[CityList[0]],'b',label=CityList[0])

ax.plot(FundTotalCity_df[CityList[1]],'r',label=CityList[1])

ax.plot(FundTotalCity_df[CityList[2]],'k',label=CityList[2])

ax.plot(FundTotalCity_df[CityList[3]],'g',label=CityList[3])

ax.plot(FundTotalCity_df[CityList[4]],'c',label=CityList[4])

ax.plot(FundTotalCity_df['Other'],'k*',label='Other')

legend = ax.legend(loc='upper center', shadow=True)

ax.set(xlabel='Months from Jan 2015', ylabel='Investment $M')

plt.show()

print("Investment has been predominately in Bangalore")

#Now focus on IndustryVertical

IndustryTotals_df = Raw_df[['IndustryVertical','AmountInUSD']].groupby('IndustryVertical').sum().sort_values('AmountInUSD',ascending=False)

IndustryList = list(IndustryTotals_df.index[:5])

FundTotalIndustry_df = Raw_df[['YrMo','IndustryVertical','AmountInUSD']]

FundTotalIndustry_df.loc[:,'IndustryVertical'] = [x if x in IndustryList else 'Other' for x in FundTotalIndustry_df['IndustryVertical']]

FundTotalIndustry_df = pd.pivot_table(FundTotalIndustry_df,index=['YrMo'],values=['AmountInUSD'],columns=['IndustryVertical'], aggfunc=np.sum, fill_value=0)['AmountInUSD'].reset_index()



fig, ax = plt.subplots()

ax.plot(FundTotalIndustry_df[IndustryList[0]],'b',label=IndustryList[0])

ax.plot(FundTotalIndustry_df[IndustryList[1]],'r',label=IndustryList[1])

ax.plot(FundTotalIndustry_df[IndustryList[2]],'k',label=IndustryList[2])

ax.plot(FundTotalIndustry_df[IndustryList[3]],'g',label=IndustryList[3])

ax.plot(FundTotalIndustry_df[IndustryList[4]],'c',label=IndustryList[4])

ax.plot(FundTotalIndustry_df['Other'],'k*',label='Other')

legend = ax.legend(loc='upper center', shadow=True)

ax.set(xlabel='Months from Jan 2015', ylabel='Investment $M')

plt.show()

print('In the past 1.5 years, investment has focused primarily on eCommerce, Consumer Internet and Technology')
#Investor analysis, see how much specific firms are investing in India

#For investor analysis, each entry could have multiple investors, so it has to be separated

InvestorTotals_df = Raw_df[['YrMo','Year','InvestorsName','AmountInUSD']]

InvestorTotals_df['AvgDeal'] = [x/len(str(y).split(',')) for (x,y) in zip(InvestorTotals_df['AmountInUSD'],InvestorTotals_df['InvestorsName'])]

Investor_df = InvestorTotals_df['InvestorsName'].str.split(',').apply(pd.Series,1).stack()

Investor_df.index = Investor_df.index.droplevel(-1)

Investor_df.name = 'Investor'

del(InvestorTotals_df['InvestorsName'])

InvestorTotals_df = InvestorTotals_df.join(Investor_df)

#Clean up investor names

InvestorTotals_df['Investor']=[str(x).strip() for x in InvestorTotals_df['Investor']]

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Alibaba Group','Alibaba')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Softbank','SoftBank Group')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('SoftBank Group Corp','SoftBank Group')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Tiger Global Management','Tiger Global')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Sequoia India','Sequoia Capital')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Sequoia Capital India Advisors','Sequoia Capital')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Sequoia Capital.','Sequoia Capital')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Foxconn Technology Group','Foxconn')

InvestorTotals_df['Investor']=InvestorTotals_df['Investor'].replace('Greenoaks Capital Partners','Greenoaks Capital')

#Now analyze -- first overall investment over entire time period

InvestorTotal_df = InvestorTotals_df[InvestorTotals_df['Year'] == '2015']

InvestorTotal_df = InvestorTotal_df[['Investor','AmountInUSD','AvgDeal']].groupby('Investor').sum().sort_values('AvgDeal',ascending=False).reset_index()

TotalDeals = InvestorTotal_df['AvgDeal'].sum()

InvestorTotal_df['DealPortion']=[x/TotalDeals for x in InvestorTotal_df['AvgDeal']]

InvestorTotal_df['CumPortion']=np.cumsum(InvestorTotal_df['DealPortion'])

InvestorCount = len(InvestorTotal_df['Investor'])

fig, ax=plt.subplots()

ax.plot(range(1,151),InvestorTotal_df['CumPortion'][:150])

ax.set(xlabel='Investor count', ylabel='Fraction of total investment', title='Investment concentration by investors (Jan 2015 - PT)')

plt.ylim([0,1])

plt.show()

print("Out of ",InvestorCount," investors, half of the investment comes from the top 15")

InvestorTotal_df = InvestorTotal_df[:20]

ax = sns.barplot(InvestorTotal_df.AvgDeal, InvestorTotal_df.Investor, orient="h", color='b')

ax.set(xlabel='Investment $M', title='Total $M Investment by Investor (2015)')

plt.xlim([0,1000])

sns.plt.show()
#Repeat analysis from Jan 2016 onward

InvestorTotal_df = InvestorTotals_df[InvestorTotals_df['Year'] != '2015']

InvestorTotal_df = InvestorTotal_df[['Investor','AmountInUSD','AvgDeal']].groupby('Investor').sum().sort_values('AvgDeal',ascending=False).reset_index()

TotalDeals = InvestorTotal_df['AvgDeal'].sum()

InvestorTotal_df['DealPortion']=[x/TotalDeals for x in InvestorTotal_df['AvgDeal']]

InvestorTotal_df['CumPortion']=np.cumsum(InvestorTotal_df['DealPortion'])

InvestorCount = len(InvestorTotal_df['Investor'])

fig, ax=plt.subplots()

ax.plot(range(1,151),InvestorTotal_df['CumPortion'][:150])

ax.set(xlabel='Investor count', ylabel='Fraction of total investment', title='Investment concentration by investors (Jan 2016 - PT)')

plt.ylim([0,1])

plt.show()

print("Out of ",InvestorCount," investors, half of the investment comes from the top 12")

InvestorTotal_df = InvestorTotal_df[:20]

ax = sns.barplot(InvestorTotal_df.AvgDeal, InvestorTotal_df.Investor, orient="h", color='b')

ax.set(xlabel='Investment $M', title='Total $M Investment by Investor (2016-PT)')

plt.xlim([0,2000])

sns.plt.show()
