import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

print('Libs Imported')
data = pd.read_csv("../input/startup_funding.csv")
data.info()
data.describe()
data.head()
data['Date'][data['Date']=='12/05.2015'] = '12/05/2015'
data['Date'][data['Date']=='13/04.2015'] = '13/04/2015'
data['Date'][data['Date']=='15/01.2015'] = '15/01/2015'
data['Date'][data['Date']=='22/01//2015'] = '22/01/2015'
data['Date'] = pd.to_datetime(data['Date'])
import re
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['MY'] = (pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.month)
data.head()
totalStartupCount = sum(data['StartupName'].value_counts())
print("Total Number of startUps:", totalStartupCount)
#lets check first in which year how many funding is received by all startups together
yearCount = data['Year'].value_counts()
print(yearCount)
plt.figure(figsize=(12,6))
sns.barplot(yearCount.index, yearCount.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Year of Funding', fontsize=12)
plt.ylabel('Number of Fundings', fontsize=12)
plt.title("Year Distribution", fontsize=16)
plt.show()
monthCounts = data['Month'].value_counts()
print(monthCounts)
plt.figure(figsize=(12,6))
sns.barplot(monthCounts.index, monthCounts.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Month of Funding', fontsize=12)
plt.ylabel('Number of Fundings', fontsize=12)
plt.title("Month Distribution", fontsize=16)
plt.show()
monthYearCount = data['MY'].value_counts()
print(monthYearCount)
plt.figure(figsize=(12,6))
sns.barplot(monthYearCount.index, monthYearCount.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('YYYYDD of Funding', fontsize=12)
plt.ylabel('Number of Fundings', fontsize=12)
plt.title("Year-Month Distribution", fontsize=16)
plt.show()
#Now lets check which location recieved higest startup
cityCount = data['CityLocation'].value_counts()[:10]
print(cityCount)
plt.figure(figsize=(12,6))
sns.barplot(cityCount.index, cityCount.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Location of Startups", fontsize=16)
plt.show()
data['InvestmentType'][data['InvestmentType']=='SeedFunding'] = 'Seed Funding'
data['InvestmentType'][data['InvestmentType']=='PrivateEquity'] = 'Private Equity'
data['InvestmentType'][data['InvestmentType']=='Crowd funding'] = 'Crowd Funding'
investTypes = data['InvestmentType'].value_counts()
print(investTypes)
plt.figure(figsize=(12,6))
sns.barplot(investTypes.index, investTypes.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Funding Type', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Funding types distribution", fontsize=16)
plt.show()
investorCounts = data['InvestorsName'].value_counts()[:10]
print(investorCounts)
plt.figure(figsize=(12,6))
sns.barplot(investorCounts.index, investorCounts.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Investors', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Investors distribution", fontsize=16)
plt.show()
# Just take a look what type of values we have
i = 0
for investor in data['InvestorsName']:
    print(investor)
    if(i==10):
        break
    i += 1
investorNames = []
for investor in data['InvestorsName']:
    for inv in str(investor).split(","):
        if inv != "":
            investorNames.append(inv.strip().lower())

startUpInvestors = pd.Series(investorNames).value_counts()[:20]
print(startUpInvestors)
            
plt.figure(figsize=(12,6))
sns.barplot(startUpInvestors.index, startUpInvestors.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Investors', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Investors distribution", fontsize=16)
plt.show()
data['IndustryVertical'][data['IndustryVertical']=='eCommerce'] = 'ECommerce'
industryCounts = data['IndustryVertical'].value_counts()[:15]
print(industryCounts)
plt.figure(figsize=(12,6))
sns.barplot(industryCounts.index, industryCounts.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Industry', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Industry distribution", fontsize=16)
plt.show()
subVerticalCounts = data['SubVertical'].value_counts()[:5]
print(subVerticalCounts)
plt.figure(figsize=(12,6))
sns.barplot(subVerticalCounts.index, subVerticalCounts.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Sub-vertical', fontsize=12)
plt.ylabel('Number of Startups', fontsize=12)
plt.title("Sub-Vertcial distribution", fontsize=16)
plt.show()
data["AmountInUSD"] = data["AmountInUSD"].apply(lambda x: float(str(x).replace(",","")))
data["AmountInUSD"] = pd.to_numeric(data["AmountInUSD"])
data.head()
# Now lets Find out what is highest amount of funding a startup
highestFund = data['AmountInUSD'].max()
print("Highest Funding amount is:", highestFund)
# Now Lets check who received this fund
print("Highest Fund Receivers\n")
data[data['AmountInUSD'] == highestFund]
# As we know Paytm and Flipkart are the Highest Fund receiver
data[data.StartupName == 'Paytm']
data[data.StartupName == 'Flipkart']
# Now lets Find out what is Lowest amount of funding a startup
lowestFund = data['AmountInUSD'].min()
print("Lowest Funding amount is:", lowestFund)
# Now Lets check who received this fund
print("Lowest Fund Receivers\n")
data[data['AmountInUSD'] == lowestFund]
# Now let's check what is average funding received by startup
print("Average Indian startups received funding of : ",data["AmountInUSD"].dropna().sort_values().mean())