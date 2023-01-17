#Install the required libraries as below

!pip install python-espncricinfo

!pip install grequests
from espncricinfo.summary import Summary

from espncricinfo.match import Match 

from espncricinfo.series import Series



import json

import requests

from bs4 import BeautifulSoup

from espncricinfo.exceptions import MatchNotFoundError, NoScorecardError



import pandas as pd
testlist = ['1119496', '1022357']
#To see the match URL we can use the below function within espncricinfo library

Match('1119496').match_url
#This functions helps to expand the list of dictonaries to columns in a dataframe.

def flatten(js):

    return pd.DataFrame(js).set_index(['text','name']).squeeze()
def getbattingdatafame(list1):

    df = pd.DataFrame()

    for x in list1:

        x1 = Match(x).html

        x2 = json.loads(x1.find_all('script')[13].get_text().replace("\n", " ").replace('window.__INITIAL_STATE__ =','').replace('&dagger;','wk').replace('&amp;','').replace('wkts;','wkts,').replace('wkt;','wkt,').strip().replace('};', "}};").split('};')[0])

        df1bat = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['1']['batsmen'])

        d1title = x2['gamePackage']['scorecard']['innings']['1']['title']

        df1bat['Team'] = d1title.split(' ')[0]

        df2bat = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['2']['batsmen'])

        d2title = x2['gamePackage']['scorecard']['innings']['2']['title']

        df2bat['Team'] = d2title.split(' ')[0]

        df1bat['Oppositionteam'] = d2title.split(' ')[0]

        df2bat['Oppositionteam'] = d1title.split(' ')[0]

        

        Finaldf_bat = pd.concat([df1bat.drop(['captain','commentary','runningScore','runningOver', 'stats','hasVideoId','href','isNotOut','roles','shortText','trackingName'], axis=1),

           df1bat.stats.apply(flatten)], axis=1).append(pd.concat([df2bat.drop(['captain','commentary','runningScore','runningOver', 'stats','hasVideoId','href','isNotOut','roles','shortText','trackingName'], axis=1),

                                                               df2bat.stats.apply(flatten)], axis=1))

        Finaldf_bat['city'] = Match(x).town_name

        Finaldf_bat['date'] = Match(x).date

        df=pd.concat([df,Finaldf_bat])

    return(df)
getbattingdatafame(testlist).head()
def getbowlingdatafame(list1):

    df = pd.DataFrame()

    for x in list1:

        x1 = Match(x).html

        x2 = json.loads(x1.find_all('script')[13].get_text().replace("\n", " ").replace('window.__INITIAL_STATE__ =','').replace('&dagger;','wk').replace('&amp;','').replace('wkts;','wkts,').replace('wkt;','wkt,').strip().replace('};', "}};").split('};')[0])

        df1bowl = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['1']['bowlers'])

        d1title = x2['gamePackage']['scorecard']['innings']['1']['title']

        df2bowl = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['2']['bowlers'])

        d2title = x2['gamePackage']['scorecard']['innings']['2']['title']

        df1bowl['Team'] = d2title.split(' ')[0]

        df2bowl['Team'] = d1title.split(' ')[0]

        df1bowl['Oppositionteam'] = d1title.split(' ')[0]

        df2bowl['Oppositionteam'] = d2title.split(' ')[0]

        

        Finaldf_bowl = pd.concat([df1bowl.drop(['captain','stats','hasVideoId','href','roles','trackingName'], axis=1),

                       df1bowl.stats.apply(flatten)], axis=1).append(pd.concat([df2bowl.drop(['captain','stats','hasVideoId','href','roles','trackingName'], axis=1),

                                                               df2bowl.stats.apply(flatten)], axis=1))

        Finaldf_bowl['city'] = Match(x).town_name

        Finaldf_bowl['date'] = Match(x).date

        df=pd.concat([df,Finaldf_bowl])

    return(df)
getbowlingdatafame(testlist).head()
print(Series('18808').years)

print(Series('18808').url)

print(Series('18808').name)