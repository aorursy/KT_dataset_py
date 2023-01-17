# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url

from bs4 import BeautifulSoup # to parse the html data and find what we want

import re # Regular expressions library, it may be useful 

print('Setup complete!')
# We need the url of the page we are gonna scrape 

url = 'https://www.kaggle.com/rankings?group=datasets&page=1&pageSize=20'

response = requests.get(url) # Get content of page
# Parse the webpage text as html

page_html = BeautifulSoup(response.text, 'html.parser') 
"""

divsContainer = page_html.find('div', attrs={'class':'site-content'}) # Find table with id = pokedex

divsContainer.find_all('div')

"""
response.text
jsonStr = response.text # Get all text from url

jsonStr = jsonStr.split('"list":[')[1] # Get everything after "list":[

jsonStr = jsonStr.split(');')[0] # Get everything before ); 

jsonStr # The json data of the 20 first rows
# Using regular expressions we will take all the links to the profiles of the 20 first dataset rankers

usernames = re.findall('userUrl":"\/(\w+)","tier', jsonStr) # Capture group of word with the username between userUrl":"\ and ","tier

len(usernames)
usernames
baseUrl = 'https://www.kaggle.com/'

datasetsUrl = '/datasets'

top20datasetsProfiles = []

for username in usernames:

    top20datasetsProfiles.append(baseUrl+username) 
top20datasetsProfiles
resp2 = requests.get(top20datasetsProfiles[0]) # Get content of page

#resp2.text
# All information we want is between "userId": and ); 

jsonStr2 = resp2.text # Get all text from url

jsonStr2 = jsonStr2.split('"userId":')[1] # Get everything after "list":[

jsonStr2 = jsonStr2.split(');')[0] # Get everything before ); 
jsonStr2
# Data to extract 



# Bio info

displayName = []

country = []

region = []

city = []

gitHubUserName = []

twitterUserName = []

linkedInUrl = []

websiteUrl = []

occupation = []

organization = []

userJoinDate = []



# Datasets info

tier = []

totalResults = []

rankCurrent = []

rankHighest = []



totalGoldMedals = []

totalSilverMedals = []

totalBronzeMedals = []

bio = jsonStr2.split('"perf')[0]

bio
datasetSummary = jsonStr2.split('datasetsSummary')[1]

datasetSummary
all_data = re.findall('(?<=,)[^,]+(?=,)', jsonStr2) # all info between commas

all_data
all_data[0].split(':"')[1].split('"')[0] # Take the second part between ""
def dataSplits(str1):

    try:

        str1 = str1.split(':"')[1].split('"')[0]

    except:

        str1 = None

    return str1

        
def getArrayAllContent(str2):

    str2 = re.findall('(?<=,)[^,]+(?=",)', str2)

    return str2
bioA = getArrayAllContent(bio)

bioA
"""

for profile in top20datasetsProfiles[:2]:

    resp2 = requests.get(profile) # Get content of page

    # All information we want is between "userId": and ); 

    jsonStr2 = resp2.text # Get all text from url

    jsonStr2 = jsonStr2.split('"userId":')[1] # Get everything after "list":[

    jsonStr2 = jsonStr2.split(');')[0] # Get everything before ); 

    

    bio = jsonStr2.split(',"perf')[0]

    datasetSummary = jsonStr2.split('datasetsSummary')[1]

    

    bioArr = getArrayAllContent(bio)

    datasetSummaryArr = getArrayAllContent(datasetSummary)

    print(len(bioArr))

    #all_data = re.findall('(?<=,)[^,]+(?=,)', jsonStr2)

    print(dataSplits(bioArr[0]))

    displayName.append(dataSplits(bioArr[0]))

    country.append(dataSplits(bioArr[1]))

    region.append(dataSplits(bioArr[2]))

    city.append(dataSplits(bioArr[3]))

    gitHubUserName.append(dataSplits(bioArr[4]))

    twitterUserName.append(dataSplits(bioArr[5]))

    linkedInUrl.append(dataSplits(bioArr[6]))

    websiteUrl.append(dataSplits(bioArr[7]))

    occupation.append(dataSplits(bioArr[8]))

    organization.append(dataSplits(bioArr[9]))

    userJoinDate.append(dataSplits(bioArr[11]))

"""    

    



    
bio
for profile in top20datasetsProfiles:

    resp2 = requests.get(profile) # Get content of page

    # All information we want is between "userId": and ); 

    jsonStr2 = resp2.text # Get all text from url

    jsonStr2 = jsonStr2.split('"userId":')[1] # Get everything after "list":[

    jsonStr2 = jsonStr2.split(');')[0] # Get everything before ); 

    

    bio = jsonStr2.split('"perf')[0]

    datasetSummary = jsonStr2.split('datasetsSummary')[1]

    

    print(re.search('displayName":"([^,]+)",', bio).group(1))

    displayName.append(re.search('displayName":"([^,]+)",', bio).group(1))

    try:

        country.append(re.search('country":"([^,]+)",', bio).group(1))

    except:

        country.append(None)

        

    try:

        region.append(re.search('region":"([^,]+)",', bio).group(1))

    except: 

        region.append(None)

        

    try:

        city.append(re.search('city":"([^,]+)",', bio).group(1))

    except:

        city.append(None)

        

    try:

        gitHubUserName.append(re.search('gitHubUserName":"([^,]+)",', bio).group(1))

    except:

        gitHubUserName.append(None)

        

    try:

        twitterUserName.append(re.search('twitterUserName":"([^,]+)",', bio).group(1))

    except: 

        twitterUserName.append(None)

    

    try:

        linkedInUrl.append(re.search('linkedInUrl":"([^,]+)",', bio).group(1))

    except:

        linkedInUrl.append(None)

        

    try:

        websiteUrl.append(re.search('websiteUrl":"([^,]+)",', bio).group(1))

    except: 

        websiteUrl.append(None)

        

    try:    

        occupation.append(re.search('occupation":"([^,]+)",', bio).group(1))

    except:

        occupation.append(None)

        

    try:

        organization.append(re.search('organization":"([^,]+)",', bio).group(1))

    except:

        organization.append(None)

        

    userJoinDate.append(re.search('userJoinDate":"([^,]+)",', bio).group(1))

    

    tier.append(re.search('tier":"([^,]+)",', datasetSummary).group(1))

    totalResults.append(re.search('totalResults":([^,]+),', datasetSummary).group(1))

    rankCurrent.append(re.search('rankCurrent":([^,]+),', datasetSummary).group(1))

    rankHighest.append(re.search('rankHighest":([^,]+),', datasetSummary).group(1))



    totalGoldMedals.append(re.search('totalGoldMedals":([^,]+),', datasetSummary).group(1))

    totalSilverMedals.append(re.search('totalSilverMedals":([^,]+),', datasetSummary).group(1))

    totalBronzeMedals.append(re.search('totalBronzeMedals":([^,]+),', datasetSummary).group(1))
displayName
datasetSummary
top20KagglersDatasets = pd.DataFrame({

    'displayName':displayName,

    'country':country,

    'region':region,

    'city':city,

    'gitHubUserName':gitHubUserName,

    'twitterUserName':twitterUserName,

    'linkedInUrl':linkedInUrl,

    'websiteUrl':websiteUrl,

    'occupation':occupation,

    'organization':organization,

    'userJoinDate':userJoinDate,

    'tier':tier,

    'totalResults':totalResults,

    'rankCurrent':rankCurrent,

    'rankHighest':rankHighest,

    'totalGoldMedals':totalGoldMedals,

    'totalSilverMedals':totalSilverMedals,

    'totalBronzeMedals':totalBronzeMedals



})
top20KagglersDatasets
# Build csv

top20KagglersDatasets.to_csv('top20KagglersDatasets.csv', index=False)