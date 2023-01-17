# import modules
import os
import re
import time
import difflib
import datetime
import dateutil
import requests
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TIRED OF THE CURRENT LEADERBOARD?
# WANT TO KNOW WHERE YOU ARE TRULY RANKED?
# WELL LOOK NO FURTHER! NOW YOU CAN MAKE YOUR OWN LEADERBOARD HOWEVER YOU WANT!

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# =============================================================================
# 1) scrape, extract and load the data
# 2) define filtering criterion to eliminate unwanted contestants
# 3) perform checks on each entry, accumulate entry IDs and then include/exclude as desired
# 4) put your own entry and see how you perform under your own rules!
# =============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# =============================================================================
# scraping the data from the leaderboard page:
# https://www.kaggle.com/c/home-credit-default-risk/leaderboard
# the url can be found by switching on devtools in your browser, reloading the page, and monitoring the network.
# the network element should be of the form leaderboard.json?include....etc
# click on it, and look at the headers - the Request URL under General is the endpoint we get the data from
# the request method is GET, so we simply use the requests library to acquire the table data.
# =============================================================================

lb = requests.get('https://www.kaggle.com/c/9120/leaderboard.json?includeBeforeUser=true&includeAfterUser=false').json()
# yeah, this is it! no painful parsing html with regex

# the data is nicely structured in json format, which makes it easy to turn it into a pandas dataframe
lbdf = pd.DataFrame(lb['beforeUser'])
lbdf = lbdf.append(pd.DataFrame(lb['afterUser']))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# defining the filtering criteria by which we can define what is allowed in the customised leaderboard
# you can choose whether to switch any of them on or off


filtering = {'recencyFilter' : False,
            'entriesFilter' : False,
            'thumbnailFilter' : False,
            'changeFilter' : False,
            'teamFilter' : False,
            'similarityFilter' : False,
            'tierFilter' : False,
            'randomFilter' : False,
            'nameFilter' : False}



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~s~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# declaring the parameters by which to filter contestants and entries by:

listOfThumbs = []
entriesCutoff = 11
similarityCutoff = 0.9

timeWeeks = 0
timeDays = 6
timeHours = 11
timeMinutes = 23
timeSeconds = 45

timeCutoff = time.time() - (timeWeeks*7*24*60*60 + timeDays*24*60*60 + timeHours*60*60 + timeMinutes*60 + timeSeconds)



# now you can change your filters to be random!
# you can even randomly pick the random filter for more randomness!

randomFilters = True

if randomFilters:
    for eachFilter in filtering.keys():
        print(eachFilter)
        print(filtering[eachFilter])
        
        filtering[eachFilter] = list(np.random.choice([True, False], 1))[0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# checking whether a contestant is still using the default thumbnail or not
# this is usually the case with brand new accounts
# we check each entry, and see whether the url for their picture matches the kaggle url where the default thumb is stored
# BEWARE! sneaky users might download the thumbnail and reupload it to another site, just to get around this check!


for eachEntry in range(len(lbdf)):
    #print(eachEntry)
    try:
        if (lbdf.iloc[eachEntry]['teamMembers'][0]['thumbnailUrl'] == 'https://storage.googleapis.com/kaggle-avatars/thumbnails/default-thumb.png'):
            #print('entry', eachEntry, 'has default thumbnail')
            listOfThumbs.append(eachEntry)
    except IndexError:
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# filtering by the number contestants in a team
# freshly made accounts generally will not be inside a team
# this reduces the contestant pool significantly, as there are many legitimate entries which are done alone
# but this is unfortunate for them. collaboration is good! enable this filter to weed out the solitary contestants

listOfTeams = []
for eachEntry in range(len(lbdf)):
    #print(len(lbdf.iloc[eachEntry]['teamMembers']))
    
    if (len(lbdf.iloc[eachEntry]['teamMembers']) > 1):
        listOfTeams.append(eachEntry)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# check for whether a profile display name is the same as the team name
# by default it will be very similar, if the entry has not been declared
# we use two similarity metrics:
# the first is the ratcliff-obershelp algorithm to find differences in strings, used by the builtin module difflib
# the other is the cosine similarity index

listOfSames = []
for eachEntry in range(len(lbdf)):
    
    for eachMember in lbdf.iloc[eachEntry]['teamMembers']:
        print(eachMember['displayName'][1:])
        print(difflib.SequenceMatcher(None, eachMember['displayName'][1:], lbdf.iloc[eachEntry]['teamName']).ratio())
        
        # now that we know how similar they are, we can decide how flexible or generous we feel today
        if (difflib.SequenceMatcher(None, eachMember['displayName'][1:], lbdf.iloc[eachEntry]['teamName']).ratio()) >= similarityCutoff:
            listOfSames.append(eachEntry)
        
    print(lbdf.iloc[eachEntry]['teamName'])
    print()

listOfSames = list(set(listOfSames))



# =============================================================================
# TODO: all this
# from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# =============================================================================


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# sorry "newbies", you are not wanted now!!
# enable this filter to throw out contestants who have not done enough kaggling - on this account atleast
# ironically this will eliminate myself aswell, but tough luck!
# git --gud

listOfNovices = []
for eachEntry in range(len(lbdf)):
    
    for eachMember in lbdf.iloc[eachEntry]['teamMembers']:
        print(eachEntry, eachMember['tier'])
        
        # simple check to see if the contestant is a novice (to kaggle! maybe you are an expert somewhere else)
        if (eachMember['tier'] == 'novice'):
            listOfNovices.append(eachEntry)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# this is to announce a new public initiative to promote self esteem and confidence in your models!
# therefore, we are not interested in including entries which keep changing their mind and submitting every chance they get!
# sorry, you should have picked a model, and STUCK WITH IT. hmph, always dropping a model as soon as another, better, smarter one comes along tsk tsk.
# for those who have just joined...whoops



# we get the current time, and then look back to see all the submissions done within the timeCutoff described above.

startTime = time.time()

newTimes = []
for eachEntry in range(len(lbdf)):
    #print(time.time())
    print(time.mktime(dateutil.parser.parse(lbdf['lastSubmission'].iloc[eachEntry]).timetuple()))
    
    # this will parse a datetime in the standard format yyyy-mm-dd HH:MM:SS.ms
    # into a parsed form: datetime.datetime(2018, 6, 9, 5, 58, 42, 826666, tzinfo=tzutc())
    # then into a timetuple: time.struct_time(tm_year=2018, tm_mon=6, tm_mday=9, tm_hour=5, tm_min=58, tm_sec=42, tm_wday=5, tm_yday=160, tm_isdst=0)
    # and lastly convert into unix epoch time 1528523922 , to easily compare to the current time 1529668413
    
    entryTime = time.mktime(dateutil.parser.parse(lbdf['lastSubmission'].iloc[eachEntry]).timetuple())
    if entryTime > timeCutoff:
        newTimes.append(eachEntry)

endTime = time.time()

print(endTime - startTime)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IN LIGHT OF THE NEW GENERAL DATA PROTECTION REGULATIONS, NAMES ARE NOT ALLOWED TO BE DISPLAYED
# PUBLICALLY IN COMPETITIONS. AS A RESULT OF THIS, ALL ENTRIES THAT CONTAIN PERSONALLY
# IDENTIFIABLE INFORMATION WILL BE IMMEDIATELY DISQUALIFIED. WE APOLOGISE FOR ANY INCONVENIENCE.
        

notNames = []

for eachEntry in range(len(lbdf)):
    print(lbdf.iloc[eachEntry]['teamName'])
    
    # searching the team names to see if they follow the format "Firstname Lastname"
    # using regex pattern to see if a capital letter followed by any amount of lowercase
    # at the start of the line, followed by a space and then a surname in the same form
    
    if re.match('^[A-Z][a-z]+ [A-Z][a-z]+$', lbdf.iloc[eachEntry]['teamName']) == None:
        notNames.append(eachEntry)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# displaying how many of the entries fall into each of the filtering categories

print('removals:')
print('similar names:'.ljust(40), len(listOfSames))
print('default thumbnails:'.ljust(40), len(listOfThumbs))
print('recent submissions:'.ljust(40), len(newTimes))
print('single person team:'.ljust(40), len(lbdf) - len(listOfTeams))
print('no leaderboard change:'.ljust(40), lbdf['change'].isnull().sum())
print(('less than ' + str(entriesCutoff) + ' entries:').ljust(40), lbdf[lbdf['entries'] < entriesCutoff].shape[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# actually doing the removals now, almost no turning back...

# this is a filter that didnt need a separate function to accumulate, but can just be applied
# if the entry has not changed position in the leaderboard yet, means you havent been around long enough!
# long term combatants only in this tournament!

# TODO: see what happened here


if filtering['changeFilter']:
    print('changeFilter applied')
    listOfThumbs = list(set(listOfThumbs).intersection(lbdf.index))
    lbdf = lbdf.drop(listOfThumbs)
    
if filtering['teamFilter']:
    print('teamFilter applied')
    lbdf = lbdf.loc[[i for i in listOfTeams if i in list(lbdf.index)], :]
    
if filtering['similarityFilter']:
    print('similarityFilter applied')
    lbdf[~lbdf.index.isin(listOfSames)]
    
if filtering['tierFilter']:
    print('tierFilter applied')
    for i in listOfNovices:
        if i in lbdf.index:
            print(i)
            lbdf = lbdf.drop(i)
    
if filtering['recencyFilter']:
    print('recencyFilter applied')
    lbdf = lbdf.loc[list(lbdf.index.difference(set(newTimes))), :]

if filtering['nameFilter']:
    print('nameFilter applied')
    lbdf[lbdf.index.isin(notNames)]

# ARE YOU FEELING LUCKY!
# *conveniently ignoring that we add our entry after removing everyone else* teehee
if filtering['randomFilter']:
    print('randomFilter applied')
    lbdf = lbdf.sample(np.random.randint(0, len(lbdf)))

# actually done the removals now, no turning back...

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

yourEntry = {'change' : None,
             'entries' : 1,
             'lastSubmission' : str(datetime.datetime.now()),
             'medal' : 'unobtainium',
             'rank' : 0,
             'score' : 0.75,
             'sourceKernelName' : 'customLeaderboard',
             'sourceKernelUrl' : 'http://192.168.1.1/',
             'teamId' : 0000000,
             'teamMembers' : [{'profileUrl': '/TheInjector',
                               'thumbnailUrl': 'https://en.wikipedia.org/wiki/Cicada_3301#/media/File:Cicada_3301_logo.jpg',
                               'tier': 'hackerman',
                               'displayName': 'Bobby T'}],
             'teamName' : "Robert'); DROP TABLE fakeKaggleLeaderboard;--"
            }

yourEntryDF = pd.DataFrame(yourEntry)
lbdf = lbdf.append(yourEntryDF).reset_index()

lbdf['rank'] = lbdf['rank'].apply(lambda x: int(x) if pd.notnull(x) else 99999)

lbdf['score'] = lbdf['score'].astype(str)
lbdf = lbdf.sort_values(['score', 'rank'], ascending=[False, False])
lbdf = lbdf.reset_index()

print(lbdf)


print('you placed', lbdf[lbdf['medal'] == yourEntry['medal']].index[0]+1, 'out of', len(lbdf))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# SCREW THESE 4 CONTESTANTS IN PARTICULAR.
lbdf = lbdf.sample(len(lbdf)-4)
