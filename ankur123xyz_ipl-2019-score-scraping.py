import pandas as pd
import json
import urllib3
from time import sleep
import warnings
warnings.filterwarnings("ignore")
http = urllib3.PoolManager() 
mat_data = pd.DataFrame()
periods = ["1","2"] # For the two innings in the match
pages = ["1","2","3","4","5","6","7","8","9","10"] # This refers to the no of requests for the inifinte scrolling
leagueId="8048" # League ID for IPL in espncricinfo
eventId="" #match ID - will be populated from the URLs later
dat_url = pd.read_csv("../input/ipl-2019-match-data/match_details.csv") # match_details extracted using scrapy
#All IPL 2019 URLs and match data in this file. Python script to extract this in my Github. Refer introduction
eventId_gp= [str(url).split("/")[6] for url in dat_url["url"]] # Extracting indiviual match ids from the URL
len_url= len(eventId_gp)
for count in range(len_url):
    eventId = eventId_gp[count]
    for period in periods:
        for page in pages:
            sleep(15) # Espncricinfo recommends a scraping delay of 15 seconds
            col_data = pd.DataFrame()
            match_dat= http.request('GET', 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId='+leagueId+'&eventId='+eventId+'&period=' +period+ '&page='+page+'&filter=full&liveTest=false')
            if(len(match_dat.data)<100):
                break
            data = json.loads(match_dat.data)
            df = pd.json_normalize(data['comments'])
            bowler=[]
            batsman=[]

            for bat,bowl in zip(df["currentBatsmen"],df["currentBowlers"]):
                batsman.append(bat[0]["name"])
                bowler.append(bowl[0]["name"])

            df["bowler"]= bowler
            df["batsman"] = batsman
            col_data = df.copy()    

            if(period=="1"):               
                df["innings"]=1
            else:
                df["innings"]=2

            if("matchWicket.text" in col_data.columns):
                col_data["matchWicket.text"].fillna("NA",inplace=True)
                col_data["run_out"]= ["Yes" if "run out" in wicket_text else "No" for wicket_text in col_data["matchWicket.text"]]
            else:
                col_data["matchWicket.text"]="NA"
                col_data["run_out"]="No"
                   
         
            col_data["match_id"] = eventId        
            mat_data = pd.concat([mat_data,col_data])   
# we are dropping the extra columns below. you can remove columns from the below list which you think are useful
mat_data.drop(["id","shortText","text","preText","postText","currentBatsmen","currentBowlers","currentInning.balls","currentInning.runs","currentInning.wickets","matchOver.maiden","matchOver.runs","matchOver.wickets","matchOver.totalRuns","matchOver.totalWicket","matchOver.runRate","matchOver.requiredRunRate","matchOver.batsmen","matchOver.bowlers","matchOver.teamShortName","matchOver.remainingOvers","matchOver.remainingBalls","matchOver.remainingRuns","matchWicket.id","matchWicket.batsmanRuns","matchWicket.batsmanBalls","matchWicket.text"],axis=1,inplace=True)
mat_data.to_csv("score.csv")

mat_data.head()
