# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url

from bs4 import BeautifulSoup # to parse the html data and find what we want

import re # Regular expressions library, it may be useful 

print('Setup complete!')
# We need the url of the page we are gonna scrape, we will start with the first page

url = 'https://www.kaggle.com/rankings.json?group=datasets&page=1&pageSize=20'

response = requests.get(url) # Get content of page
import json

# Get json object, take a look at how data is structured

json_resp = json.loads(response.text)

#json_resp # Uncomment to see structure 
#Counts for tier

json_resp['counts']
# We need to sum the first 3 tiers, the ones that count for the datasets ranking (grandmaster, master, expert)

rankingTiers = json_resp['counts'][0:3]

rankingTiers
# We need the number of rankers in dataset to know how many pages of 20 (every page has a fixed size of 20 users) we need to scrappe.

numberOfRankers = 0

for tier in rankingTiers:

    numberOfRankers += tier['count']

print(numberOfRankers)
# Round up using ceil as we need an extra page for the last ones

import math

numPagsToScrappe = math.ceil(numberOfRankers / 20)

numPagsToScrappe
# Get the kagglers users data 

usersList = json_resp['list']

usersList[0] # Show first user of the list
# Data to extract



currentRanking = []

displayName = []

thumbnailUrl = []

userId = []

userUrl = []

tier = []

points = []

joined = []

totalGoldMedals = []

totalSilverMedals = []

totalBronzeMedals = []



baseURL = 'https://www.kaggle.com/rankings.json?group=datasets&pageSize=20&page='



for page in range(1, numPagsToScrappe + 1): # Page query starts at 1, and its a range()  function so we need to add 1 

    # We need the url of the page we are gonna scrape

    pageToScrape = baseURL + str(page) # To acces the multiple pages

    resp = requests.get(pageToScrape) # Get content of page

    json_response = json.loads(resp.text) # Get JSON object 

    jsonRespListUsers = json_response['list'] # Get list of users of the JSON object

    

    for user in range(0, len(jsonRespListUsers)):

        currentRanking.append(jsonRespListUsers[user]['currentRanking'])

        displayName.append(jsonRespListUsers[user]['displayName'])

        #thumbnailUrl.append(jsonRespListUsers[user]['thumbnailUrl'])

        userId.append(jsonRespListUsers[user]['userId'])

        userUrl.append(jsonRespListUsers[user]['userUrl'])

        tier.append(jsonRespListUsers[user]['tier'])

        points.append(jsonRespListUsers[user]['points'])

        joined.append(jsonRespListUsers[user]['joined'])

        totalGoldMedals.append(jsonRespListUsers[user]['totalGoldMedals'])

        totalSilverMedals.append(jsonRespListUsers[user]['totalSilverMedals'])

        totalBronzeMedals.append(jsonRespListUsers[user]['totalBronzeMedals'])



        
# Create dataFrame with the information we have

topKagglersDatasets = pd.DataFrame({

    'displayName':displayName,

    'currentRanking':currentRanking,

    #'thumbnailUrl':thumbnailUrl,

    'userId':userId,

    'userUrl':userUrl,

    'tier':tier,

    'points':points,

    'userJoinDate':joined,

    'totalGoldMedals':totalGoldMedals,

    'totalSilverMedals':totalSilverMedals,

    'totalBronzeMedals':totalBronzeMedals



})
topKagglersDatasets.head(7)
topKagglersDatasets.shape
topKagglersDatasets.to_csv('topKagglersDatasets.csv', index=False)
userPage = 'https://www.kaggle.com/cdeotte'



userResp = requests.get(userPage)



jsonStr = userResp.text # Get all text from url

jsonStr = jsonStr.split('{"userId"')
jsonStr = '{"userId"' + jsonStr[1]
jsonStr = jsonStr.split(');')[0] # Get everything before ); 
bioObj = json.loads(jsonStr) # Create a JSON object containing all info
#bioObj # Uncomment to check all info about about user
bioObj['datasetsSummary'] # We could use all this information to complement the dataset

# This data contains a summary of datasets category and highlights of the top 3 most popular datasets (by number of votes)
bioObj['datasetsSummary']['highlights'][0]
# Data to extract in profile



country = []

region = []

city = []

gitHubUserName = []

twitterUserName = []

linkedInUrl = []

websiteUrl = []

occupation = []

organization = []



# Dataset summary info



totalResults = []

rankPercentage = []

rankCurrent = []

rankHighest = []



# Dataset summary -> highlights 



top1title = []

top1date = []

top1medal = []

top1score = []

top1url = []



top2title = []

top2date = []

top2medal = []

top2score = []

top2url = []



top3title = []

top3date = []

top3medal = []

top3score = []

top3url = []

"""

import time



baseUrl2 = 'https://www.kaggle.com'

for row in topKagglersDatasets['userUrl']:

    time.sleep(0.5) # You have to wait between request, try not to saturate server sending to much requests. You have to be polite in data mining.

    # Get profile URL

    profileUrl = baseUrl2 + row



    # Get content of profile URL

    userResp = requests.get(profileUrl)

    jsonStr2 = userResp.text # Get all text from url

    

    # Split text content to extract JSON content

    jsonStr2 = jsonStr2.split('{"userId"')

    jsonStr2 = '{"userId"' + jsonStr2[1]

    jsonStr2 = jsonStr2.split(');')[0] # Get everything before ); 

    

    # Create JSON object for easier manipulation

    bioObj = json.loads(jsonStr2) # Create a JSON object containing all data

    

    

    country.append(bioObj['country'])

    region.append(bioObj['region'])

    city.append(bioObj['city'])

    gitHubUserName.append(bioObj['gitHubUserName'])

    twitterUserName.append(bioObj['twitterUserName'])

    linkedInUrl.append(bioObj['linkedInUrl'])

    websiteUrl.append(bioObj['websiteUrl'])

    occupation.append(bioObj['occupation'])

    organization.append(bioObj['organization'])



    

    totalResults.append(bioObj['datasetsSummary']['totalResults'])

    rankPercentage.append(bioObj['datasetsSummary']['rankPercentage'])

    rankCurrent.append(bioObj['datasetsSummary']['rankCurrent'])

    rankHighest.append(bioObj['datasetsSummary']['rankHighest'])

    

    top1title.append(bioObj['datasetsSummary']['highlights'][0]['title'])

    top1date.append(bioObj['datasetsSummary']['highlights'][0]['date'])

    top1medal.append(bioObj['datasetsSummary']['highlights'][0]['medal'])

    top1score.append(bioObj['datasetsSummary']['highlights'][0]['score'])

    top1url.append(bioObj['datasetsSummary']['highlights'][0]['url'])



    top2title.append(bioObj['datasetsSummary']['highlights'][1]['title'])

    top2date.append(bioObj['datasetsSummary']['highlights'][1]['date'])

    top2medal.append(bioObj['datasetsSummary']['highlights'][1]['medal'])

    top2score.append(bioObj['datasetsSummary']['highlights'][1]['score'])

    top2url.append(bioObj['datasetsSummary']['highlights'][1]['url'])



    top3title.append(bioObj['datasetsSummary']['highlights'][2]['title'])

    top3date.append(bioObj['datasetsSummary']['highlights'][2]['date'])

    top3medal.append(bioObj['datasetsSummary']['highlights'][2]['medal'])

    top3score.append(bioObj['datasetsSummary']['highlights'][2]['score'])

    top3url.append(bioObj['datasetsSummary']['highlights'][2]['url'])

"""