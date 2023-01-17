import pandas as pd 

import numpy as np

import trueskill as ts

from IPython.display import display

pd.set_option('display.max_rows', 300)

pd.set_option('display.max_columns', 50)
def loadCleanCompileResults(nameCol,raceName,appendScore=0,fleetName=None,reverseNameOrder=False):



    #load gold, silver fleets if they exist

    if fleetName is not None:

        dfResultsTemp = pd.read_csv('../input/' + raceName + '-' + fleetName + '.csv',encoding = "ISO-8859-1")

    else:

        dfResultsTemp = pd.read_csv('../input/' + raceName + '.csv',encoding = "ISO-8859-1")

        

    #create a ranking column with the column name as the race name

    dfResultsTemp[raceName] = dfResultsTemp.index + 1



    #make the racer's name the index

    if (len(nameCol)) == 2:

        dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp[nameCol[0]] + ' ' + dfResultsTemp[nameCol[1]])

    else:

        dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp[nameCol])

        

    if reverseNameOrder: 

        count = 0

        newIndex = []

        for name in dfResultsTemp.index:

            numNames = len(name.split(' '))

            newIndex.append(name.split(' ')[numNames-1] + ' ' + name.split(' ')[0])

            count += 1

        dfResultsTemp = dfResultsTemp.set_index([newIndex])

    



    dfResultsTemp.index = dfResultsTemp.index.str.lower()        

    dfResultsTemp.index = dfResultsTemp.index.str.replace(r"(\w)([A-Z])", r"\1 \2")

        

    dfResultsTemp.index = dfResultsTemp.index.str.title()

    dfResultsTemp.index = dfResultsTemp.index.str.replace('\([A-Z\ 0-9]*\)','')

    dfResultsTemp.index = dfResultsTemp.index.str.strip()

        

        

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Riccardo Andrea Leccese','Rikki Leccese')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Nicolas Parlier','Nico Parlier')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Alejandro Climent Hernã¥_ Ndez', 'Alejandro Climent Hernandez')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Alexandre Caizergues','Alex Caizergues')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Florian Trit.*','Florian Trittel Paul')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Jean Guillaume Rivaud','Jean-Guillaume Rivaud')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('^Kieran Le$','Kieran Le Borgne')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Marvin Baumeister.*','Marvin Baumeister Schoenian')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Theo De Ramecourt','Theo De-Ramecourt')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('James Johnson','James Johnsen')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('^Enrico$','Enrico Tonon')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Roman Liubimtsev','Roman Lyubimtsev')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Tomek Glazik','Tomasz Glazik')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Andrew Hansen','Andy Hansen')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Andrew Mc Manus','Andrew McManus')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Matthew Taggart','Matt Taggart')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Joey Pasqauli','Joey Pasquali')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Stefanus Viljoen','Stefaans Viljoen')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('^Alejandro Climent$','Alejandro Climent Hernandez')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('?','')

    dfResultsTemp.index = dfResultsTemp.index.str.replace('Jan Blaesiide','Jan Blaesino')  	        

       

    

    #append score adjusts positions for silver fleet and bronze fleets

    dfResultsTemp[raceName] = dfResultsTemp[raceName] + appendScore



       

    return dfResultsTemp[[raceName]]

    
def mergeResults(dfResults,dfResultsTemp,raceName):

    dfResults = pd.merge(dfResults,dfResultsTemp[[raceName]],left_index=True,right_index=True,how='outer')

    return dfResults
#initialize the results table

dfResults = pd.DataFrame()
raceName = '20160229-LakeTaupo-NZNationals'

dfResultsTemp = loadCleanCompileResults('HelmName',raceName)

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
raceName = '20160323-LaVentana-HydrofoilProTour'

dfResultsTemp = loadCleanCompileResults(['Name','LastName'],raceName,0,'Gold') 

dfResultsTempSilver = loadCleanCompileResults(['Name','LastName'],raceName,len(dfResultsTemp),'Silver')

dfResultsTemp = dfResultsTemp.append(dfResultsTempSilver)

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
raceName = '20160515-ShermanIsland-RippinTheRio'

dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
raceName = '20160516-MontPellier-IFKOSilverCup'

dfResultsTemp = loadCleanCompileResults('Name',raceName,0,None,True) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
##Load Cagliari Results



raceName = '20160522-Cagliari-EuropeanChampionships'



dfResultsTemp = loadCleanCompileResults('Name',raceName,0,'Platinum') 

dfResultsTempGold = loadCleanCompileResults('Name',raceName,len(dfResultsTemp),'Gold')

dfResultsTemp = dfResultsTemp.append(dfResultsTempGold)

dfResultsTempSilver = loadCleanCompileResults('Name',raceName,len(dfResultsTemp),'Silver') 

dfResultsTemp = dfResultsTemp.append(dfResultsTempSilver)



dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
##Load Brisbane Results



raceName = '20160710-Brisbane-SailBrisbane'

dfResultsTemp = loadCleanCompileResults('Skipper',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
###Load Italy results

raceName = '20160717-Gizzeria-IKAGoldCup'



dfResultsTemp = loadCleanCompileResults('Name',raceName,0,'Gold') 

dfResultsTempSilver = loadCleanCompileResults('Name',raceName,len(dfResultsTemp),'Silver')

dfResultsTemp = dfResultsTemp.append(dfResultsTempSilver)

dfResultsTempBronze = loadCleanCompileResults('Name',raceName,len(dfResultsTemp),'Silver')

dfResultsTemp = dfResultsTemp.append(dfResultsTempBronze)



dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
##Load SF results

raceName = '20160807-SanFrancisco-HydrofoilProTour'



dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)

##Load Mauritius results

raceName = '20160820-Mauritius-HydrofoilProTour'



dfResultsTemp = loadCleanCompileResults('HelmName',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)

##Load Fehnmanr Results - Men

raceName = '20160830-Fehnmarn-PringlesFoilWorldCup'



dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)
raceName = '20160911-Denmark-NationalChampionship'

dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)



##Load Weifang Results - Men

raceName = '20160915-Weifang-WorldChampionship'



dfResultsTemp = loadCleanCompileResults('Name',raceName,0,'PlatinumAndGold') 

dfResultsTempSilver = loadCleanCompileResults('Name',raceName,len(dfResultsTemp),'Silver')

dfResultsTemp = dfResultsTemp.append(dfResultsTempSilver)



dfResults = mergeResults(dfResults, dfResultsTemp,raceName)

##Load Weifang Results - Women

raceName = '20160915-Weifang-WorldChampionship-Women'



dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)

##Load Weifang Results - Women

raceName = '20160918-SanFrancisco-BattleOfTheBay'



dfResultsTemp = loadCleanCompileResults('Name',raceName) 

dfResults = mergeResults(dfResults, dfResultsTemp,raceName)

def doRating(dfResults):

    

    env = ts.TrueSkill()

    

    #remove people who haven't completed 3 races

    #dfResults = dfResults[dfResults.count(axis=1) > 2]

    

    columns = ['Name','mu_minus_3sigma','NumRaces','Rating']

    dfRatings = pd.DataFrame(columns=columns,index=dfResults.index)

    dfRatings['NumRaces'] = dfResults.count(axis=1)

    dfRatings['Rating'] = pd.Series(np.repeat(env.Rating(),len(dfRatings))).T.values.tolist()



    for raceCol in dfResults:

        competed = dfRatings.index.isin(dfResults.index[dfResults[raceCol].notnull()])

        rating_group = list(zip(dfRatings['Rating'][competed].T.values.tolist()))

        ranking_for_rating_group = dfResults[raceCol][competed].T.values.tolist()

        dfRatings.loc[competed, 'Rating'] = ts.rate(rating_group, ranks=ranking_for_rating_group)



    

    dfRatings = pd.DataFrame(dfRatings) #convert to dataframe



    dfRatings['mu_minus_3sigma'] = pd.Series(np.repeat(0.0,len(dfRatings))) #calculate mu - 3 x sigma: MSFT convention



    for index, row in dfRatings.iterrows():

        dfRatings.loc[dfRatings.index == index,'mu_minus_3sigma'] = float(row['Rating'].mu) - 3 * float(row['Rating'].sigma)



    

    dfRatings['Name'] = dfRatings.index

    dfRatings.index = dfRatings['mu_minus_3sigma'].rank(ascending=False).astype(int) #set index to ranking

    dfRatings.index.names = ['Rank']



 

    

    return dfRatings.sort_values('mu_minus_3sigma',ascending=False) 
#Create Rating

dfRatings = doRating(dfResults)

display(dfRatings)