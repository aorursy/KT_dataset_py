# import the libraries first
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 1: Data Cleaning 1\n')

# ********************* The code below this line until the next star marked line is for checkpoint 1: Data Cleaning 1 ************************

pathCompany = '../input/sparkfunds/companies.txt'
pathRounds2 = '../input/sparkfunds/rounds2.csv'
pathMapping = '../input/sparkfunds/mapping.csv'


# so, for any join we need to use 'permalink' from companies and 'company_permalink' from rounds2
# in order to join these two dataframes on the common id, lets convert the values to lowercase and trim any trailing whitespaces

companies = pd.read_csv(pathCompany,encoding='latin',sep='\t')
#companies.describe()
#companies.head()

# from the above, we notice that the unique column is 'permalink' (as opposed to name, since name can be duplicate)
# also, it can also be case sensitive, but we are only interested in comparing names in one case, so making it lower for 
# both data frames below
companies['permalink'] = companies['permalink'].str.lower().str.strip()

rounds2 = pd.read_csv(pathRounds2,encoding='latin')
#rounds2.head()
#rounds2.describe()
rounds2['company_permalink'] = rounds2['company_permalink'].str.lower().str.strip()



# from the above, we can conclude that unique id from companies dataframe is 'permalink'
# that from rounds2 dataframe is 'company_permalink'

# lets drop duplicate rows (if any) from the two dataframes and store them in new dataframes
# please note that we are not changing the original dataframes but creating newer ones just in case we might need them in future

# 1. remove duplicates from rounds2 and store in a different data frame
rounds2Unique = rounds2.drop_duplicates(keep=False, inplace=False)

# 2. remove duplicates from companies and store in a different data frame
companiesUnique = companies.drop_duplicates(keep=False, inplace=False)

# get the count of unique companies in rounds2
uniqueRounds2Count = rounds2Unique['company_permalink'].nunique()
print('The number of unique companies present in rounds2 dataset is: ', uniqueRounds2Count)

# get the count of unique companies in companies
uniqueCompaniesCount = companiesUnique['permalink'].nunique()
print('The number of unique companies present in companies dataset is: ', uniqueCompaniesCount)

# number of unique companies is equal to 66368 from the above dataframe definition
# and number of unique rounds2 companies is 66370

# for all the future purposes, we will use companiesUnique & rounds2Unique

leftMergedDataFrame = pd.merge(rounds2Unique,companiesUnique,how='left',right_on='permalink',left_on='company_permalink', indicator=True)
leftMergedDataFrame.isnull().sum()

# the above statement results in a non zero value hence we can conclude that there are companies in rounds2 which are not there in companies

# let's get observations whose merge key only appears in ‘left’ DataFrame
onlyLeft = leftMergedDataFrame[leftMergedDataFrame['_merge'] != 'both']
print('Total number of entries present in rounds2 but not in companies is: ',len(onlyLeft))

print('Total number of rows in rounds2Unique is: ', len(rounds2Unique))

# now create a master_frame such that all variables (columns) in the companies frame are added to the rounds2 data frame
master_frame = pd.merge(rounds2Unique,companiesUnique,how='inner',right_on='permalink',left_on='company_permalink')
master_frame_unique = master_frame.drop_duplicates(keep=False, inplace=False)
countOfObservations = len(master_frame_unique.index)

print('The total number of observations present in master_frame is: ',countOfObservations)

# ********************************* Data Cleaning for master_frame *******************************************

print('\nLets now clean the master_frame by checking if there are columns with blank/null/NaN values. If found, clean the master_frame.\n')
# we must check if there are columns with blank/null/NaN values and also see what fraction of the total dataset do they occupy
# based on this fraction, we should decide which columns to keep and which ones to drop from the dataframe.

# lets check if master_frame has missing data anywhere
master_frame.isnull().any()

# and get their sum
master_frame.isnull().sum()

# and in terms of percentage (rounded off to 2 decimal values)
round((master_frame.isnull().sum()*100/len(master_frame.index)),2)

# we know for a fact that the columns: funding_round_code,founded_at, homepage_url,state_code,region & city are of very little significance for now
# lets remove them
master_frame = master_frame.drop('funding_round_code', axis=1)
master_frame = master_frame.drop('founded_at', axis=1)
master_frame = master_frame.drop('homepage_url', axis=1)
master_frame = master_frame.drop('state_code', axis=1)
master_frame = master_frame.drop('region', axis=1)
master_frame = master_frame.drop('city', axis=1)

# also, the column 'company_permalink' is duplicate in the merged dataframe, lets drop that one too
master_frame = master_frame.drop('company_permalink', axis=1)

# now lets check how does the master_frame now look like after the above removals
round((master_frame.isnull().sum()*100/len(master_frame.index)),2)

# the column 'raised_amount_usd' is of utmost significance for our data analysis
# lets take off any rows with NaN values in 'raised_amount_usd' from master_frame
master_frame = master_frame[~np.isnan(master_frame['raised_amount_usd'])] 

# now lets check how does the master_frame now look like
round((master_frame.isnull().sum()*100/len(master_frame.index)),2)

# we still have 6 columns with missing values, the question here is: do we delete such rows? 
# well, to answer that, lets first check if we have any row(s) with all of fields as blank
master_frame[master_frame.isnull().sum(axis=1) > 6].shape

# clearly, there are no rows which satisfy the above condition : the above statement resulted in (0,9)
# how does the master_frame now look like
round((master_frame.isnull().sum()*100/len(master_frame.index)),2)

# the above result shows there are still two columns namely "category_list" & "country_code" with blank values
# since we know that these are string columns, we can simply replace the blanks, or in other words
# impute them with suitable values so that the further data manipulation/filter operations are easy to perform

#lets do that now
# impute blank country_code with a user defined code as 'UNKWN'
master_frame.loc[pd.isnull(master_frame['country_code']), ['country_code']] = 'UNKWN'

# impute blank category_list with a user defined value as 'undefined'
master_frame.loc[pd.isnull(master_frame['category_list']), ['category_list']] = "undefined"

# lets check how does the master_frame now look like
round((master_frame.isnull().sum()*100/len(master_frame.index)),2)

master_frame.head(25)

# we do see that the column category_list has values containing pipe symbol '|' as the separator
# assuming we are only interested in the first of such categories, lets clean the category_list column
# and sotre the cleaned values as primary_sector

# function to strip additional values and retian only the first value
def takeFirstValueAfterBar(value):
    values = value.split('|')
    if len(values) > 1:
        return values[0]
    else:
        return value
    

master_frame['primary_sector'] = master_frame['category_list'].apply(takeFirstValueAfterBar)
master_frame['primary_sector'] = master_frame['primary_sector'].str.lower() # make it lowercase for ease of data manipulation

# check the values again

print('\nAfter all the cleaning, master_frame data now looks like below. \n')
print(master_frame.head())

# all good now!
print('\nAll good now!')

print('\nCheckpoint 1 ends here. \n')

# ****************************************************************************************************************************

print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 2: Funding Type Analysis\n')
# ********************* The code below this line until the next star marked line is for checkpoint 2: Funding Type Analysis ************************
ventureDF = master_frame[master_frame['funding_round_type']=='venture']
angelDF = master_frame[master_frame['funding_round_type']=='angel']
seedDF = master_frame[master_frame['funding_round_type']=='seed']
peDF = master_frame[master_frame['funding_round_type']=='private_equity']

ventureAvgFunding = ventureDF["raised_amount_usd"].mean(skipna = True)
angelAvgFunding = angelDF["raised_amount_usd"].mean(skipna = True)
seedAvgFunding = seedDF["raised_amount_usd"].mean(skipna = True)
peAvgFunding = peDF["raised_amount_usd"].mean(skipna = True)

print('Average funding amount of venture type is ', ventureAvgFunding)
print('Average funding amount of angel type is ', angelAvgFunding)
print('Average funding amount of seed type is ', seedAvgFunding)
print('Average funding amount of private equity type is ', peAvgFunding)

# get the Average funding in millions to know which one is the best suited for Spark Funds
ventureAvgFundingInMillions = ventureAvgFunding/1000000
angelAvgFundingInMillions = angelAvgFunding/1000000
seedAvgFundingInMillions = seedAvgFunding/1000000
peAvgFundingInMillions = peAvgFunding/1000000

print('Average Venture Funding In Millions is: ', ventureAvgFundingInMillions)
print('Avergae Angel Funding In Millions is: ', angelAvgFundingInMillions)
print('Average Seed Funding In Millions is: ', seedAvgFundingInMillions)
print('Average Private Equity Funding In Millions is: ', peAvgFundingInMillions)

# Since we know the window for investemtn for Spark Funds is between 5 & 15 millions, 
# the best suited investment type from the above values is Venture Type (11.75 millions), rest all are out of the investment window
print('\nAt this point, it looks like the most suitable type of funding for Spark Funds amongst venture, angel, seed & private_equity is Venture Type.\n')

# lets do some chart plotting to visualize this data

print('\nLets do some chart plotting to visualize this data and to support our claim. \n')

# first set the theme, I used the thems mentioned here: https://python-graph-gallery.com/104-seaborn-themes/
sns.set_style("darkgrid")

# lets see the boxplot of a 'raised_amount_usd' plotted against 'funding_round_type', stats shown here are mean
plt.figure(figsize=(25, 10))
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master_frame)
plt.yscale('symlog') # source: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html, value : {"linear", "log", "symlog", "logit", ...}
plt.show()


# after trying out various possible values for yscale, I settled for "symlog"

# the box plot too showed similar results. angel & seed and not even close to a million, and private_equity overshoots the range (5 to 15 mil)
# looks like our bet is Venture Type

# lets reduce the scope of funding to only 4 types as expected, i.e. 'angel','seed','venture' & 'private_equity' and replot the data
master_frame = master_frame[master_frame['funding_round_type'].isin(['venture', 'angel', 'seed', 'private_equity'])]
master_frame_with_4_funding_types =  master_frame.copy() # this will be used in checkpoint 6

plt.figure(figsize=(15, 10))
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master_frame)
plt.yscale('symlog') # source: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html, value : {"linear", "log", "symlog", "logit", ...}
plt.show()


# lets now also check the funding type values as counts to see the concentration
plt.figure(figsize=(10, 8))
sns.countplot(y="funding_round_type", data=master_frame)
plt.show()


# since we know the funding window is between 5 to 15 mils, lets draw the countplot
# and juxtapose it with a barplot of raised amount v funding type 
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.countplot(x="funding_round_type", data=master_frame)
plt.title("Total number of Investments")

plt.subplot(1, 2, 2)
sns.barplot(y='raised_amount_usd', x="funding_round_type", data=master_frame, estimator=np.mean)
plt.title("Average")
plt.axhline(y=5000000, linewidth=3, color = 'm') # source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axhline.html
plt.axhline(y=15000000, linewidth=3, color = 'm') # valid color values: one of ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``, source: https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html
plt.show()

# we can safely conclude, based on the above plots and arrived numbers, that the safest bet for Spark Funds is 'Venture Type'

print('\nBased on the above plots and arrived numbers, we can safely conclude that the most suitable investment type for Spark Funds is Venture Type.\n')

# lets filter our dataframe to contain only venture type funding
master_frame = master_frame[master_frame['funding_round_type'].isin(['venture'])]
master_frame.head()


print('\nCheckpoint 2 ends here. \n')
# ****************************************************************************************************************************

print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 3: Country Analysis\n')
# ********************* The code below this line until the next star marked line is for checkpoint 3: Country Analysis ************************

# we are expected to list top 9 countries in the order of they receiving the funds
# lets arrange the master_frame in descending order based on the total funding received (we are talking groupby here)
master_frame.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False).head(25)
master_frame.head()

# looking at the data above, we can conclude that the top 9 country codes are as follows:
# 'USA', 'CHN', 'GBR', 'IND', 'CAN', 'FRA', 'ISR', 'DEU', 'JPN', excluding the country code 'UNKWN'

# our required dataframe top9 can be created as below
top9Countries = ['USA', 'CHN', 'GBR', 'IND', 'CAN', 'FRA', 'ISR', 'DEU', 'JPN']
top9CountriesSeries = master_frame['country_code'].isin(top9Countries)
top9 = master_frame[top9CountriesSeries]
print('\nThe top 9 countries in the descending order of fundings they received are as follows:')
print(top9.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False))

# *** Below section uses web scraping to get a list of English speaking countires from the url specified in the assignment *************************

# url for getting list of Englsh speaking countries as per the assignment is:
# https://en.wikipedia.org/wiki/List_of_territorial_entities_where_English_is_an_official_language
import requests
from bs4 import BeautifulSoup
import re
from pprint import pprint
import lxml.html as lh
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_territorial_entities_where_English_is_an_official_language"
countryData = requests.get(url)
soup = BeautifulSoup(countryData.text, 'html.parser')

#Store the contents of the website under doc
doc = lh.fromstring(countryData.content)

#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')

#Check the length of the first 150 rows
#[len(T) for T in tr_elements[:150]]

# the above results has different column values as 6,4,3 pertaining to different table strucutres



# store the column headers
#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    #print(i,name)
    col.append((name,[]))



# our first row is the header, data is stored on the second row onwards

def createDataFrame(colCount):
    for j in range(1,len(tr_elements)):
        #T is our j'th row
        T=tr_elements[j]
        
    
    
        #If row is not of size 6, the //tr data is not from our table 
        if len(T)!=colCount:
            break
    
        #i is the index of our column
        i=0

        #Iterate through each element of the row
        for t in T.iterchildren():
            data=t.text_content() 
            #Check if row is empty
            if i>0:
            #Convert any numerical value to integers
                try:
                    data=int(data)
                except:
                    pass
            #Append the data to the empty list of the i'th column
            col[i][1].append(data)
            #Increment i for the next column
            i+=1

    
# call the method for tables having 6 columns
createDataFrame(6)

#[len(C) for (title,C) in col]

# this shows every column in our tables with 6 columns has exactly 62 records

# lets create a data frame of countries from the above data
countriesDictionary ={title:column for (title,column) in col}
countriesDF =pd.DataFrame(countriesDictionary)

#countriesDF.head(100)

# please note that we have got to remove the row at index 4 since it is the header row for the next table in our wiki page

modCountriesDF = countriesDF.drop(4)

# we see the Singapore population has [21] appended to it, lets strip it of as it is just a link on actual page
def takeFirstValueAfterSplit(value,char):
    values = value.split(char)
    if len(values) > 1:
        return values[0]
    else:
        return value
    


# modCountriesDF.head(50)

# strip the population of \n
modCountriesDF['Population\n'] = modCountriesDF['Population\n'].map(lambda x: str(x).replace('\n',''))
modCountriesDF['Population\n'] = modCountriesDF['Population\n'].map(lambda x: str(x).replace(',',''))


modCountriesDF['Population\n'] = modCountriesDF['Population\n'].apply(lambda x: takeFirstValueAfterSplit(x,'['))
modCountriesDF['Population\n'] = modCountriesDF['Population\n'].apply(lambda x: takeFirstValueAfterSplit(x,'+'))

modCountriesDF['Population\n'] = pd.to_numeric(modCountriesDF['Population\n'])

#modCountriesDF.head(50)

sortedCountriesDataFrame = modCountriesDF.sort_values(by='Population\n', axis=0, ascending=False, inplace=False)

sortedCountriesDataFrame.head(60)

listOfEnglishSpeakingCountries = sortedCountriesDataFrame['Alpha-3 code\n'].map(lambda x: str(x).replace('\n',''))

print('\nTop English speaking countries based on their poluation arranged in descending order is')
print(listOfEnglishSpeakingCountries)

# please note that we have only taken the countires where English is either a de-facto or dejure language form the tables with 6 columns
# this is because other tables do not have country codes

# ****************************************** This section ends here ***********************************************************

# from the above we can safely conclude that the Top 3 English speaking countires where investments are being made
# in the Venture Type are as follows:
"""
USA
GBR
IND
"""


# lets create a dataframe that only has these 3 countries which will be used for further Data Analysis

# our required dataframe can be created as below
top3Countries = ['USA', 'GBR', 'IND']
top3CountriesSeries = top9['country_code'].isin(top3Countries)
top3_english_speaking_countries = top9[top3CountriesSeries]

print('\nThe top 3 countries in the descending order of fundings they received are as follows:')
print(top3_english_speaking_countries.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False))


# lets plot sum across top 3 english official language country_code for better data visualization
print('\nFor better data visualization across top 3 english official language countries, lets plot count & sum in juxtaposed sub-plots.\n')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.countplot(x="country_code", data=top3_english_speaking_countries)
plt.title("Count")
plt.subplot(1, 2, 2)
sns.barplot(x="country_code", y="raised_amount_usd", data=top3_english_speaking_countries, estimator=sum)
plt.title("Sum")
plt.show()


print('\nCheckpoint 3 ends here. \n')

# ****************************************************************************************************************************

print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 4: Sector Analysis 1\n')
# ********************* The code below this line until the next star marked line is for checkpoint 4: Sector Analysis 1 ************************

#Let's load the mapping csv file into a dataframe and start with sector wise analysis
mapping = pd.read_csv(pathMapping, encoding = "latin")


# looking at the data, lets drop the row with category_list as NaN, i.e. row at index 0
mapping = mapping.drop(0)

# lets also add a column which takes care of taking first value after splitting from | symbol
mapping['category_list'] = mapping['category_list'].apply(lambda x: takeFirstValueAfterSplit(x,'|'))
mapping['category_list'] = mapping['category_list'].str.lower() # make it lowercase for ease of data manipulation


# lets create a pivot based on category list to its main sector
mapping = pd.melt(mapping,id_vars=["category_list"])
mapping = mapping[mapping.value != 0]

# rename columns as per our need and convert primary sector column to lower for easy data manipulations later
mapping = mapping.drop('value', axis=1)
mapping.rename(columns={'variable':'main_sector','category_list':'primary_sector'}, inplace=True) # inplace=True modifies the original dataframe
mapping['primary_sector'] = mapping['primary_sector'].str.lower()


print('Primary Sector mapped to its Main Sector is as follows:\n')
print(mapping.head())
print('\n')


# lets drop duplicate rows from mapping if any
mapping = mapping.drop_duplicates(keep=False, inplace=False)

# merge the above two dataframes to get expected results as per checkpoint 4
top3_english_merged = pd.merge(top3_english_speaking_countries, mapping, how='left', on='primary_sector', indicator=True)

# lets drop duplicate rows from top3_english_merged if any
top3_english_merged = top3_english_merged.drop_duplicates(keep=False, inplace=False)


# if we just take values which are present on left dataframe
top3_english_left = top3_english_merged[top3_english_merged['_merge'] != 'both']



# Lets see the unique primary sector name
top3_english_left.primary_sector.sort_values(ascending=True).unique()

# and that for mapping 
mapping.primary_sector.sort_values(ascending=True)

# after looking at the results, it is only fair to say that the primary_sector column have some corrupt data
# where we see a '0', it must actually be 'na', for e.g. '0notechnology' should be 'nanotechnology'

# lets clean that up first
import re
mapping['primary_sector'] = mapping.primary_sector.apply(lambda x: re.sub('[0]', 'na', str(x)))

mapping.primary_sector.sort_values(ascending=True)

# lets perform the merge again with this clean data
# merge the above two dataframes to get expected results as per checkpoint 4
top3_english_merged = pd.merge(top3_english_speaking_countries, mapping, how='left', on='primary_sector', indicator=True)


# if we just take values which are present on left dataframe
top3_english_left = top3_english_merged[top3_english_merged['_merge'] != 'both']


# Lets see the unique primary sector name again after clean-up
top3_english_left.primary_sector.sort_values(ascending=True).unique()


# since we did a left join, we can remove all rows that does not have 'both' indicator.
top3_english_merged = top3_english_merged[top3_english_merged['_merge'] == 'both']
top3_english_merged = top3_english_merged.drop('_merge', axis=1)

print('\n The data of top 3 English Speaking Countries after merging with Mappings data is as follows: \n')
print(top3_english_merged.head())

print('\nCheckpoint 4 ends here. \n')

# ****************************************************************************************************************************

print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 5: Sector Analysis 2\n')
# ********************* The code below this line until the next star marked line is for checkpoint 5: Sector Analysis 2 ************************

# for Checkpoint 5, we are talking about dividing the investment into various subgroups

# lets start by dropping now drop all the records whose investment is outside our window of 5 to 15 mils
top3_english_merged = top3_english_merged.drop(top3_english_merged[(top3_english_merged.raised_amount_usd < 5000000)].index)
top3_english_merged = top3_english_merged.drop(top3_english_merged[(top3_english_merged.raised_amount_usd > 15000000)].index)

# name the dataframes for top 3 countires as D1, D2 & D3 as asked in the assignment
D1 = top3_english_merged[top3_english_merged['country_code'] == 'USA']
D2 = top3_english_merged[top3_english_merged['country_code'] == 'GBR']
D3 = top3_english_merged[top3_english_merged['country_code'] == 'IND']

print('\nBased on the dataframes D1, D2 & D3 created for countries C1, C2 & C3 (i.e. USA, GBR & IND), we can conclude the below: \n')
# get the count & sum of investments in C1, i.e. USA
countOfInvestmentsInUSA = D1['raised_amount_usd'].count()
totalInvestmentsInUSA = D1['raised_amount_usd'].sum()

print('Total number of Investments made in USA is: ',countOfInvestmentsInUSA)
print('Total amount of Investments made in USA is: ',totalInvestmentsInUSA)

# get the count & sum of investments in C2, i.e. GBR
countOfInvestmentsInGBR = D2['raised_amount_usd'].count()
totalInvestmentsInGBR = D2['raised_amount_usd'].sum()

print('Total number of Investments made in GBR is: ',countOfInvestmentsInGBR)
print('Total amount of Investments made in GBR is: ',totalInvestmentsInGBR)


# get the count & sum of investments in C3, i.e. IND
countOfInvestmentsInIND = D3['raised_amount_usd'].count()
totalInvestmentsInIND = D3['raised_amount_usd'].sum()

print('Total number of Investments made in IND is: ',countOfInvestmentsInIND)
print('Total amount of Investments made in IND is: ',totalInvestmentsInIND)


# to get other values as best,second best & third best sectors, it would be handy if we create a pivot table for different countries
print('\nTo get other values as best,second best & third best sectors, it would be handy if we create a pivot table and group by for different countries.')
print('\nHere we go!\n')
# lets start with C1, i.e. 'USA' and carry on for other countries
print('\nPivot for USA\n')
print(D1.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'}))

# to get only sector info lets do a group by
print('\nGroup By Data for USA\n')
print(D1.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False))


# For GBR
print('\nPivot for GBR\n')
print(D2.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'}))

# to get only sector info lets do a group by
print('\nGroup By Data for GBR\n')
print(D2.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False))


# For IND
print('\nPivot for IND\n')
print(D3.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'}))

# to get only sector info lets do a group by
print('\nGroup By Data for IND\n')
print(D3.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False))

print('\nBased on the above pivots and group by data, we can safely conclude the below for USA, GBR & IND:\n')

print('Sector-wise data for USA:')
print('Top Sector name (no. of investment-wise): Others')
print('Second Sector name (no. of investment-wise): Social, Finance, Analytics, Advertising')
print('Third Sector name (no. of investment-wise): Cleantech / Semiconductors')
print('Number of investments in top sector (3) is 2950')
print('Number of investments in second sector (4) is 2714')
print('Number of investments in top sector (3) is 2350')

print('\nSector-wise data for GBR:')
print('Top Sector name (no. of investment-wise): Others')
print('Second Sector name (no. of investment-wise): Social, Finance, Analytics, Advertising')
print('Third Sector name (no. of investment-wise): Cleantech / Semiconductors')
print('Number of investments in top sector (3) is 147')
print('Number of investments in second sector (4) is 133')
print('Number of investments in top sector (3) is 130')

print('\nSector-wise data for IND:')
print('Top Sector name (no. of investment-wise): Others')
print('Second Sector name (no. of investment-wise): Social, Finance, Analytics, Advertising')
print('Third Sector name (no. of investment-wise): News, Search and Messaging')
print('Number of investments in top sector (3) is 110')
print('Number of investments in second sector (4) is 60')
print('Number of investments in top sector (3) is 52')


# now the task is to get the companies by the order of they receiving the investment

print('\nLets show the companies data by the order of they receiving the investment as follows:\n')
# lets start with C1: USA

# for sector 'Others'
print('USA Investment data for Others')
print(D1[D1['main_sector'] == "Others" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in the USA for 'Others' was made in company with permalink: /organization/virtustream

# for sector 'Social, Finance, Analytics, Advertising'
print('\nUSA Investment data for Social, Finance, Analytics, Advertising')
print(D1[D1['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in the USA for 'Social, Finance, Analytics, Advertising' was made in company with permalink: /organization/shotspotter


# now repeat the above for C2: GBR

# for sector 'Others'
print('\nGBR Investment data for Others')
print(D2[D2['main_sector'] == "Others" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in the GBR for 'Others' was made in company with permalink: /organization/electric-cloud

# for sector 'Social, Finance, Analytics, Advertising'
print('\nGBR Investment data for Social, Finance, Analytics, Advertising')
print(D2[D2['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in the GBR for 'Social, Finance, Analytics, Advertising' was made in company with permalink: /organization/celltick-technologies


# repeat the above for C3: IND

# for sector 'Others'
print('\nIND Investment data for Others')
print(D3[D3['main_sector'] == "Others" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in India for 'Others' was made in company with permalink: /organization/firstcry-com

# for sector 'Social, Finance, Analytics, Advertising'
print('\nIND Investment data for Social, Finance, Analytics, Advertising')
print(D3[D3['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5))
# so the highest investment in India for 'Social, Finance, Analytics, Advertising' was made in company with permalink: /organization/manthan-systems


# now lets refer to companies dataframe to get the country name from the permalink just obtained for the above 6 cases

companiesUnique.head(50)

USACompanyHighestInvtInOthers = companiesUnique.loc[companiesUnique['permalink'] == '/organization/virtustream', 'name'].iloc[0]
USACompanyHighestInvtSocial = companiesUnique.loc[companiesUnique['permalink'] == '/organization/shotspotter', 'name'].iloc[0]

GBRCompanyHighestInvtInOthers = companiesUnique.loc[companiesUnique['permalink'] == '/organization/electric-cloud', 'name'].iloc[0]
GBRCompanyHighestInvtSocial = companiesUnique.loc[companiesUnique['permalink'] == '/organization/celltick-technologies', 'name'].iloc[0]

IndianCompanyHighestInvtInOthers = companiesUnique.loc[companiesUnique['permalink'] == '/organization/firstcry-com', 'name'].iloc[0]
IndianCompanyHighestInvtSocial = companiesUnique.loc[companiesUnique['permalink'] == '/organization/manthan-systems', 'name'].iloc[0]

print('\nBased on the above data and combining it with companies dataframe, we can safely conclude the below:')

print('\nThe highest investment in the USA for Others was made in the company: ', USACompanyHighestInvtInOthers)
print('\nThe highest investment in the USA for Social, Finance, Analytics, Advertising was made in the company: ', USACompanyHighestInvtSocial)

print('\nThe highest investment in the GBR for Others was made in the company: ', GBRCompanyHighestInvtInOthers)
print('\nThe highest investment in the GBR for Social, Finance, Analytics, Advertising was made in the company: ', GBRCompanyHighestInvtSocial)

print('\nThe highest investment in India for Others was made in the company: ', IndianCompanyHighestInvtInOthers)
print('\nThe highest investment in India USA for Social, Finance, Analytics, Advertising was made in the company: ', IndianCompanyHighestInvtSocial)

print('\nCheckpoint 5 ends here. \n')

# *********************************************************************************************************************************************


print('****************************************************************************************************************************')
print('\nThe answers below until the next star marked line are specific for checkpoint 6: Plots\n')
# ********************* The code below this line until the next star marked line is for checkpoint 6: Plots ************************

# task 1: A plot showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type.
# task 2: A plot showing the top 9 countries against the total amount of investments of funding type FT. This should make the top 3 countries (Country 1, Country 2, and Country 3) very clear.
# task 3: A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart (for the chosen investment type FT). 


fractionSeries = master_frame_with_4_funding_types.funding_round_type.value_counts(normalize=True)

# task 1:
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
fractionSeries.plot(kind='bar') 
plt.title("Fraction of total number of Investments")

plt.subplot(1, 2, 2)
sns.barplot(y='raised_amount_usd', x="funding_round_type", data=master_frame_with_4_funding_types, estimator=np.mean)

# add margin lines between 5 and 15 mils to get better visualization of which funding type falls within
plt.axhline(y=5000000, linewidth=3, color = 'k') # source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axhline.html
plt.axhline(y=15000000, linewidth=3, color = 'k') # valid color values: one of ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``, source: https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html
plt.title("Average")
plt.show()

# from the above sub-plots, it is absolutely clear that the best suited investment funding round type is "venture"

# task 2:
plt.figure(figsize=(20, 10))
sns.barplot(x="country_code", y="raised_amount_usd", data=top9, estimator=sum)
plt.title("Country-wise funding")
plt.show()

# clearly the top 3 English Speaking Countries doing heaviest investments on funding type "Venture" are: USA, GBR & IND
# remember, as per our previous analysis of English Speaking Countries, China (CHN) isn't an English Speaking Country
# hence we have excluded it

# task 3.
frames = [D1, D2, D3]
result = pd.concat(frames)
plt.figure(figsize=(20, 10))
sns.barplot(y='country_code', x='raised_amount_usd', hue="main_sector", data=result, estimator=np.sum)
plt.show()

print('\nFrom the above plots it is now clear that top 3 sectors for:')
print('\nUSA are:\n 1. Others.\n 2. Social, Finance, Analytics, Advertising.\n 3. Cleantech / Semiconductors.')
print('\nGBR are:\n 1. Others.\n 2. Social, Finance, Analytics, Advertising.\n 3. Cleantech / Semiconductors.')
print('\nIND are:\n 1. Others.\n 2. Social, Finance, Analytics, Advertising.\n 3. News, Search and Messaging.')

print('\nThe above findings validate our claims made in Checkpoint 5.')

print('\nCheckpoint 6 ends here. \n')

print('****************************************************************************************************************************')
# ********************************* Checkpoint 6 ends here **********************************************
# ************************ The Python code for Spark Funds Investment Assignment ends here *************************
