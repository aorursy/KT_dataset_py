import pandas as pd
dfLiveCovid = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTSN2GPC-c61Ff7IXybM7bomLm3LCyuXY9QAT_io5tWb02k9-BfRG9AGtLFK65WBN1_xCctyIU3lpPM/pub?gid=0&single=true&output=csv', header=None)
dfLiveCovid.head()
dfLiveCovid.describe
dfHistoricalCovid = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTSN2GPC-c61Ff7IXybM7bomLm3LCyuXY9QAT_io5tWb02k9-BfRG9AGtLFK65WBN1_xCctyIU3lpPM/pub?gid=97730520&single=true&output=csv',
                  header=None)
dfHistoricalCovid.head()
dfHistoricalCovid.describe
usaNewDeathsHistory = dfHistoricalCovid[4:5]
usaNewDeathsHistory
usaNewDeathsToday = usaNewDeathsHistory.iloc[0,-2]
# usaDeathsT=pd.to_numeric(usaNewDeathsToday)
# usaDeathsT
type(usaNewDeathsToday)

# aa= pd.to_numeric(usaNewDeathsToday)
usaDeathsT = (usaNewDeathsToday.replace(',', ''))
usaNewDeathsHistory

yesterdayDeath = usaNewDeathsHistory.iloc[[0],[-4]]
yest = (yesterdayDeath.replace(',', ''))
yest= yest.apply(pd.to_numeric)
usaDeathsT=pd.to_numeric(usaDeathsT)
change = ((int(usaDeathsT)-yest)/usaDeathsT)*100
percentageChange = round(change,2)
percentageChange

yesterdayDeath = usaNewDeathsHistory.iloc[[0],[-4]]
yest = yesterdayDeath.to_numpy()
usaDeathsLast2Days = usaDeathsT+ int(yest )

usaDeathsLast2Days

dayBeforerYesterdayDeath = usaNewDeathsHistory.iloc[[0],[-6]]
dBYD = int(dayBeforerYesterdayDeath.to_numpy())
dBYD

dayBeforerTheDayBeforeYD = usaNewDeathsHistory.iloc[[0],[-8]]
dBTDBY = int(dayBeforerTheDayBeforeYD.to_numpy())
dBTDBY
other2Days = dBYD + dBTDBY
other2Days

change2Days = ((usaDeathsLast2Days-other2Days)/usaDeathsLast2Days)*100
percentageChange2Days = round(change2Days,2)
percentageChange2Days
usaDeathsT
from datetime import date 
todays = date. today()
todays
print ("USA deaths ", todays, usaDeathsT, 
       "\nPercentage change today against yesterday",percentageChange,"%\n"
      "Percentage change, sum 2 days against the previous 2 days",percentageChange2Days,"%"
      )