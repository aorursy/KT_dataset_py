#pandas is a library that allows you to manipulate data in "data frames". 

#Very helpful when dealing with data

import pandas as pd 



import numpy as np



#Trueskill is the function that does the ranking. 

#It's the algorithm Microsoft Research developed to rank XBox players.

#Great summary at: http://www.moserware.com/2010/03/computing-your-skill.html

import trueskill as ts 



#library for displaying tables

from IPython.display import display

pd.set_option('display.max_rows', 300)
#what is are the dates  of the current season

pd.read_csv('../input/season.tsv',sep='\t')[-2:-1]
#pull the completed regattas for this season

df_public_regattas = pd.read_csv('../input/public_regatta.tsv',sep='\t')

#convert the end_date column to date time. When read_csv read it in, it was read as a string. 

df_public_regattas['end_date'] = pd.to_datetime(df_public_regattas['end_date'])

df_public_regattas = df_public_regattas[(df_public_regattas['end_date'] > '2016-08-16') & (df_public_regattas['end_date'] < '2017-01-14')]



#I took a guess about when to include a regatta in the rankings. I assumed the status should be finished or final before a regatta gets included. 

df_public_regattas = df_public_regattas[(df_public_regattas['dt_status'] == 'finished') | (df_public_regattas['dt_status'] == 'final')]
#pull the teams and ranks in each regatta

df_team = pd.read_csv('../input/team.tsv',sep='\t')

df_team = df_team[(df_team['school'].notnull()) | (df_team['name'].notnull())]
#compile all the selected regattas into a results table for ranking



#takes the regattas table as an input. 

#Good programming practices probably would've involved passing  the team table as well



def compile_results(df_public_regattas):

    df_results = pd.DataFrame()

    #loop through the chosen regattas

    for index, row in df_public_regattas.iterrows():

      

        #df_team[['school','name','dt_rank']] selects the school name, team name and the team finishing position from the team table 

        #df_team['regatta'] == row['id'] pulls out only the regattas in the df_public_regattas data frame

        df_results_temp = df_team[['school','name','dt_rank']][df_team['regatta'] == row['id']]

        

        #combine the the school name and team name and make them the index

        df_results_temp.index = df_results_temp['school'] + ' ' + df_results_temp['name']

        

        #delete tose colummns because now that they're combined in the index.

        del(df_results_temp['school'])

        del(df_results_temp['name'])

        

        #make the regatta id the column name

        df_results_temp.columns = [row['id']]

        #name the index 

        df_results_temp.index.names = ['school_name']

    

        #add the new results to the results data frame

        df_results = pd.merge(df_results,df_results_temp,left_index=True,right_index=True,how='outer')

    return df_results
#function to create the ratings table and rank the teams



def doRating(dfResults):

    

    env = ts.TrueSkill()

    

    

    columns = ['Name','Rating','NumRegattas','Rating_Raw']

    #create a ratings data frame with name, rating, number of regattas and the raw trueskill rating

    dfRatings = pd.DataFrame(columns=columns,index=dfResults.index)

    

    #count number of results to calculate number of regattas

    dfRatings['NumRegattas'] = dfResults.count(axis=1)

    

    #initialize the raw ratings column

    dfRatings['Rating_Raw'] = pd.Series(np.repeat(env.Rating(),len(dfRatings))).T.values.tolist()



    #loop through the regattas

    for raceCol in dfResults:

        #a boolean list that tells us which teams competed

        competed = dfRatings.index.isin(dfResults.index[dfResults[raceCol].notnull()])

        #pulls out the ratings for those who competed

        rating_group = list(zip(dfRatings['Rating_Raw'][competed].T.values.tolist()))

        #get the rankings for those who competed

        ranking_for_rating_group = dfResults[raceCol][competed].T.values.tolist()

        #update the rankings

        dfRatings.loc[competed, 'Rating_Raw'] = ts.rate(rating_group, ranks=ranking_for_rating_group)



    

    dfRatings = pd.DataFrame(dfRatings) #convert to dataframe



    dfRatings['Rating'] = pd.Series(np.repeat(0.0,len(dfRatings))) #calculate mu - 3 x sigma: MSFT convention



    for index, row in dfRatings.iterrows():

        #make the actual ranking mu - 3 x sigma as per this recommendation: https://www.kaggle.com/antgoldbloom/d/antgoldbloom/2016-kitefoil-race-results/trueskill-for-kitefoil-rankings-by-race/comments#131047

        dfRatings.loc[dfRatings.index == index,'Rating'] = float(row['Rating_Raw'].mu) - 3 * float(row['Rating_Raw'].sigma)



    

    dfRatings['Name'] = dfRatings.index

    dfRatings = dfRatings.dropna()

    dfRatings.index = dfRatings['Rating'].rank(ascending=False).astype(int) #set index to ranking

    dfRatings.index.names = ['Rank']



 

    

    return dfRatings.sort_values('Rating',ascending=False) 
##we need to aggregate teams into schools. 

#I do this by taking a weight average of each team's ranking to create a school ranking



def create_school_ratings(df_ratings): 

    df_school_ratings = pd.DataFrame()



    #calculated the weight average (numregatta* team rating) to get a school rating

    #I suspec there's a more elegant way to do this using groupby

    df_ratings['SchoolId'] = df_ratings['Name'].str.extract('([A-Z]*)',expand=False)

    for school in df_ratings['SchoolId'].unique():

        num_regattas = df_ratings[df_ratings['SchoolId'] == school]['NumRegattas'].sum()

        rating = (df_ratings[df_ratings['SchoolId'] == school]['Rating'] * df_ratings[df_ratings['SchoolId'] == school]['NumRegattas']).sum()/num_regattas

        df_school_ratings = df_school_ratings.append([[school,rating,num_regattas]],ignore_index=True)



    df_school_ratings.columns =['SchoolId','Rating','NumRegattas']



    ##merge with school name

    df_school_name = pd.read_csv('../input/active_school.tsv',sep='\t')

    df_school_ratings = pd.merge(df_school_ratings,df_school_name[['id','name']],left_on='SchoolId',right_on='id')





    #reorder columns and remove id column

    cols = df_school_ratings.columns.tolist()

    cols = cols[-1:] + cols[1:-2]

    df_school_ratings = df_school_ratings[cols]

    df_school_ratings.columns.values[0] = 'SchoolName'



    #make the school's rank the index

    df_school_ratings.index = df_school_ratings['Rating'].rank(ascending=False).astype(int) #set index to ranking

    df_school_ratings.index.name = 'Rank'



    #order by rating

    df_school_ratings = df_school_ratings.sort_values('Rating',ascending=False)

    return df_school_ratings

    #display the school rankings
df_public_regattas_intersectional = df_public_regattas[(df_public_regattas['dt_num_divisions'] > 1) & (df_public_regattas['type'] == 'intersectional')]

df_public_regattas_intersectional_coed = df_public_regattas_intersectional[df_public_regattas_intersectional['participant'] == 'coed']

df_results_intersectional_coed = compile_results(df_public_regattas_intersectional_coed)

df_ratings_intersectional_coed = doRating(df_results_intersectional_coed)

df_school_ratings_intersectional_coed = create_school_ratings(df_ratings_intersectional_coed)

display(df_school_ratings_intersectional_coed)
df_public_regattas_intersectional = df_public_regattas[(df_public_regattas['dt_num_divisions'] > 1) & (df_public_regattas['type'] == 'intersectional')]

df_public_regattas_intersectional_coed = df_public_regattas_intersectional[df_public_regattas_intersectional['participant'] == 'women']

df_results_intersectional_coed = compile_results(df_public_regattas_intersectional_coed)

df_ratings_intersectional_coed = doRating(df_results_intersectional_coed)

df_school_ratings_intersectional_coed = create_school_ratings(df_ratings_intersectional_coed)

display(df_school_ratings_intersectional_coed)
df_public_regattas_intersectional = df_public_regattas[(df_public_regattas['dt_num_divisions'] > 1) & (df_public_regattas['type'] == 'intersectional')]

df_public_regattas_intersectional_coed = df_public_regattas_intersectional[df_public_regattas_intersectional['participant'] == 'coed']

df_results_intersectional_coed = compile_results(df_public_regattas_intersectional_coed)

df_ratings_intersectional_coed = doRating(df_results_intersectional_coed)

df_school_ratings_intersectional_coed = create_school_ratings(df_ratings_intersectional_coed)

df_school_ratings_intersectional_coed = df_school_ratings_intersectional_coed[df_school_ratings_intersectional_coed['NumRegattas'] > 2]

df_school_ratings_intersectional_coed.index = df_school_ratings_intersectional_coed['Rating'].rank(ascending=False).astype(int)

display(df_school_ratings_intersectional_coed)
df_public_regattas_intersectional = df_public_regattas[(df_public_regattas['dt_num_divisions'] > 1) & (df_public_regattas['type'] == 'intersectional')]

df_public_regattas_intersectional_women = df_public_regattas_intersectional[df_public_regattas_intersectional['participant'] == 'women']

df_results_intersectional_women = compile_results(df_public_regattas_intersectional_women)

df_ratings_intersectional_women = doRating(df_results_intersectional_women)

df_school_ratings_intersectional_women = create_school_ratings(df_ratings_intersectional_women)

df_school_ratings_intersectional_women = df_school_ratings_intersectional_women[df_school_ratings_intersectional_women['NumRegattas'] > 2]

df_school_ratings_intersectional_women.index = df_school_ratings_intersectional_women['Rating'].rank(ascending=False).astype(int)

display(df_school_ratings_intersectional_women)

df_public_regattas_coed = df_public_regattas[df_public_regattas['participant'] == 'coed']

df_results_coed = compile_results(df_public_regattas_coed)

df_ratings_coed = doRating(df_results_coed)

df_school_ratings_coed = create_school_ratings(df_ratings_coed)

#remove schools that don't have 3+ regattas

df_school_ratings_coed = df_school_ratings_coed[df_school_ratings_coed['NumRegattas'] > 2]

df_school_ratings_coed.index = df_school_ratings_coed['Rating'].rank(ascending=False).astype(int)

display(df_school_ratings_coed)
df_public_regattas_women = df_public_regattas[df_public_regattas['participant'] == 'women']

df_results_women = compile_results(df_public_regattas_women)

df_ratings_women = doRating(df_results_women)

df_school_ratings_women = create_school_ratings(df_ratings_women)

#remove schools that don't have 3+ regattas

df_school_ratings_women = df_school_ratings_women[df_school_ratings_women['NumRegattas'] > 2]

df_school_ratings_women.index = df_school_ratings_women['Rating'].rank(ascending=False).astype(int)

display(df_school_ratings_women)