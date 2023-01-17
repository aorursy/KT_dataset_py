# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url

print('Setup complete!')
# We need the url of the page we are gonna scrape, we will start with the first page

url = 'https://www.kaggle.com/rankings.json?group=notebooks&page=1&pageSize=20'

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
# We need the number of rankers in notebooks to know how many pages of 20 (every page has a fixed size of 20 users) we need to scrappe.

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
baseURL = 'https://www.kaggle.com/rankings.json?group=notebooks&pageSize=20&page='



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

topKagglersNotebooks = pd.DataFrame({

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
# First 7 rows of the dataframe

topKagglersNotebooks.head(7)
# We check the sizes of the dataframe

topKagglersNotebooks.shape
# Build csv

topKagglersNotebooks.to_csv('topKagglersNotebooks.csv', index=False)