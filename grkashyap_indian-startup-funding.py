#Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#Method to format AmountInUSD column values
def convertAmounts(amt):
    if amt != np.nan:
        amt = amt.replace(',','')
        if len(amt) > 0:
            return float(amt)
        else:
            return 0
    return 0
#Read input csv file
df = pd.read_csv('../input/startup_funding.csv',converters={'AmountInUSD':convertAmounts})
#Check info of dataset
df.info()
#Check head of dataset
df.head()
#AmountInUSD has some null values. Replace null values with mean
mean_of_amount = int(np.mean(df['AmountInUSD']))
df['AmountInUSD'] = df['AmountInUSD'].apply(lambda x:mean_of_amount if x == 0 else x )
#Replace . with / and // with / of the date values so that it would be easy to extract year
df['Date'] = df['Date'].apply(lambda dt:dt.replace(".","/"))
df['Date'] = df['Date'].apply(lambda dt:dt.replace("//","/"))
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['year'] = df['Date'].apply(lambda dt:dt.year)
#Check whether new column is added 
df.info()
#The dataset has data for three years only
df['year'].value_counts()
#Check head of dataset
df.head()
#Get count of startups which received funding group by year
by_year_cmp = df[['StartupName','year']].groupby(by='year').count()
#Plot no.of companies which received funding groupby year
sns.barplot(x='year',y='StartupName',data=by_year_cmp.reset_index())
plt.title('No.of startups which received funding group by year')
plt.xlabel('Year')
plt.ylabel('Count of startups')
plt.tight_layout()
#Calculate the mean of funding amount group by year
by_year = df.groupby(by='year')['AmountInUSD'].mean().astype('int64')
#The dataset has data for three years only. Hence, creating a list for the available years
years = ['2015','2016','2017']

#Plot average funding amount group by year
sns.barplot(x=years,y=by_year.values)
plt.xlabel('Year')
plt.ylabel('Average amount (in Millions USD)')
plt.title('Average amount received by startups (in Millions USD)')
plt.tight_layout()
#Drop NA values from industry verticals
#Convert all available values to uppercase so that it would be easy to retrieve unique records
df['IndustryVertical'] = df['IndustryVertical'].dropna()
industries_vertical = df['IndustryVertical'].apply(lambda iv:str(iv).upper())
#Create new numpy arrays to store industries which received funding and its respective counts
industries = industries_vertical.dropna().unique()[:10]
industries_count = industries_vertical.value_counts()[:10]
#Plot industry verticals and count
plt.figure(figsize=(12,8))
sns.barplot(x=industries_count, y=industries,)
#plt.xticks(rotation=45)
plt.ylabel('Industry Verticals')
plt.xlabel('Count')
plt.title('Top 5 industry verticals which received most funding')
#plt.tight_layout()
#Drop NA values from investor names and convert the values to upper case
df['InvestorsName'] = df['InvestorsName'].dropna()
investors = df['InvestorsName'].apply(lambda iv:str(iv).upper())
investors_count = investors.value_counts()[:10]
investors = investors.dropna().unique()[:10]
#Plot investor names and count
plt.figure(figsize=(12,8))
sns.barplot(y=investors,x=investors_count)
plt.xticks(rotation=90)
plt.ylabel('Investors')
plt.xlabel('Count')
plt.title('Top 10 Investors and the count of companies')
#Create a new dataframe which holds investors and amount
by_investors_df = df[['InvestorsName','AmountInUSD']]
#Convert investor names to uppercase
by_investors_df['InvestorsName'] = by_investors_df['InvestorsName'].apply(lambda iv:str(iv).upper())
#Group by investornames and retrieve the sum of amounts
by_investors = by_investors_df.groupby(by='InvestorsName').sum().astype('int64')
#Sort the calculates sum in descending order
by_investors.sort_values(by='AmountInUSD',ascending=False,inplace=True)
#Plot top-10 investors and corresponding invested amount
top_investors_name = by_investors.reset_index()[:10]['InvestorsName']
top_invested_amount = by_investors.reset_index()[:10]['AmountInUSD']
plt.figure(figsize=(12,8))
sns.barplot(y=top_investors_name,x=top_invested_amount)
plt.ylabel('Investors')
plt.xlabel('Average Invested Amount (in Millions USD)')
plt.title('Top 10 Investors and the average amount invested')
#As per data there are different values for OLA and Flipkart.
#Standardize them so that it will be easy to group them
df['StartupName'] = df['StartupName'].apply(lambda x:x[0:3] if x.replace(' ','').lower() == 'olacabs' else x)
df['StartupName'] = df['StartupName'].apply(lambda x:x[0:x.index('.')] if x == 'Flipkart.com' else x)
#df['StartupName'] = df['StartupName'].apply(lambda x:x[0:3] if x == 'Olacabs' else x)
#Create a new dataframe which holds startups and amount
top_companies_df = df[['StartupName','AmountInUSD']]
#Convert startup names to uppercase
top_companies_df['StartupName'] = top_companies_df['StartupName'].apply(lambda iv:str(iv).upper())
#Group by startupname and retrieve the sum of amounts
top_companies = top_companies_df.groupby(by='StartupName').sum().astype('int64')
#Sort the calculates sum in descending order
top_companies.sort_values(by='AmountInUSD',ascending=False,inplace=True)
#Plot the top-10 startups which received highest funding
top_companies_name = top_companies.reset_index()[:10]['StartupName']
top_companies_amount = top_companies.reset_index()[:10]['AmountInUSD']
plt.figure(figsize=(12,8))
sns.barplot(y=top_companies_name,x=top_companies_amount)
plt.ylabel('Companies')
plt.xlabel('Average investment amount received (in Millions USD)')
plt.title('Top 10 Companies and average amount received')
#Function to take care of cities which has / in them. Ex: Bangalor / California.....
def format_cities(city):
    city = city.replace(" ","")
    if "/" in city:
        print(city)
        print(city[0:city.index("/")].upper())
        return city[0:city.index("/")].upper()
    else:
        return city.upper()
#Somehow nan values in dataset are stored as float datatype. convert them to np.nan so that
#it would be easy to remove them
df['CityLocation'] = df['CityLocation'].apply(lambda x:np.nan if str(x) == 'nan' else x)

#Now drop nan values easily
df['CityLocation'] = df['CityLocation'].dropna()
#Create a new dataframe which holds Cities and amount
top_cities_df = df[['CityLocation','AmountInUSD']]
#Group by startupname and retrieve the sum of amounts
top_cities = top_cities_df.groupby(by='CityLocation').sum().astype('int64')
#Sort the calculates sum in descending order
top_cities.sort_values(by='AmountInUSD',ascending=False,inplace=True)
#Plot city and amount
top_cities_name = top_cities.reset_index()[:10]['CityLocation']
top_cities_amount = top_cities.reset_index()[:10]['AmountInUSD']
plt.figure(figsize=(12,8))
sns.barplot(y=top_cities_name,x=top_cities_amount)
plt.ylabel('Cities')
plt.xlabel('Average investment amount received (in Millions USD)')
plt.title('Top 10 Cities and average amount received')