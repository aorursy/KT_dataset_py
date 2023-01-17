import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import trueskill as ts

from IPython.display import display

pd.set_option('display.max_rows', 300)
#what is are the dates  of the current season

pd.read_csv('../input/season.tsv',sep='\t')[-4:-1]
#pull the completed regattas for f15 and s16 seasons

df_public_regattas = pd.read_csv('../input/public_regatta.tsv',sep='\t')

df_public_regattas['end_date'] = pd.to_datetime(df_public_regattas['end_date'])

df_public_regattas = df_public_regattas[(df_public_regattas['end_date'] > '2016-08-16') & (df_public_regattas['end_date'] < '2017-01-14')]

df_public_regattas = df_public_regattas[(df_public_regattas['dt_status'] == 'finished') | (df_public_regattas['dt_status'] == 'final')]

df_public_regattas = df_public_regattas[(df_public_regattas['scoring'] == 'standard') | (df_public_regattas['scoring'] == 'combined')]
df_dt_rp = pd.read_csv('../input/dt_rp.tsv',sep='\t')

df_team_division = pd.read_csv('../input/dt_team_division.tsv',sep='\\t')

df_team = pd.read_csv('../input/team.tsv',sep='\t')

df_active_sailor = pd.read_csv('../input/active_sailor.tsv',sep='\t')

df_active_school = pd.read_csv('../input/active_school.tsv',sep='\t')
def compile_results(df_public_regattas, skipper_or_crew):

    #the ID column is getting read in as a strong, which means causes an issue for joining to dt_rp. Convert to strong

    df_team_division['id'] = pd.to_numeric(df_team_division['id'],errors='coerce')

    df_team_division_results = pd.merge(df_dt_rp,df_team_division,left_on='team_division',right_on='id',how='inner')

    df_team_division_results_with_team = pd.merge(df_team_division_results,df_team,left_on='team',right_on='id',how='inner')

    #only include regattas in public_regattas

    df_team_division_results_with_team = df_team_division_results_with_team[df_team_division_results_with_team['regatta'].isin(df_public_regattas['id'])]

    df_team_division_results_with_team['rank'] = df_team_division_results_with_team['rank_x']

    df_results = pd.DataFrame()

    for index, row in df_public_regattas.iterrows():

        df_results_temp = df_team_division_results_with_team[['sailor', 'rank']][(df_team_division_results_with_team['regatta'] == row['id']) & (df_team_division_results_with_team['boat_role'] == skipper_or_crew)]

        df_results_temp.index = df_results_temp['sailor']

        df_results_temp.index.names = ['sailor']

        del(df_results_temp['sailor'])

        df_results_temp.columns = [row['id']]

        df_results = pd.merge(df_results,df_results_temp,left_index=True,right_index=True,how='outer')

    df_results.drop_duplicates(inplace=True)

    return df_results
#function to create the ratings table and rank the teams



def doRating(dfResults):

    

    env = ts.TrueSkill()

    

    #remove people who haven't completed 3 races

    #dfResults = dfResults[dfResults.count(axis=1) > 2]

    

    columns = ['Name','Rating','NumRegattas','Rating_Raw']

    dfRatings = pd.DataFrame(columns=columns,index=dfResults.index)

    dfRatings['NumRegattas'] = dfResults.count(axis=1)

    dfRatings['Rating_Raw'] = pd.Series(np.repeat(env.Rating(),len(dfRatings))).T.values.tolist()



    for raceCol in dfResults:

        competed = dfRatings.index.isin(dfResults.index[dfResults[raceCol].notnull()])

        rating_group = list(zip(dfRatings['Rating_Raw'][competed].T.values.tolist()))

        ranking_for_rating_group = dfResults[raceCol][competed].T.values.tolist()

        dfRatings.loc[competed, 'Rating_Raw'] = ts.rate(rating_group, ranks=ranking_for_rating_group)



    

    dfRatings = pd.DataFrame(dfRatings) #convert to dataframe



    dfRatings['Rating'] = pd.Series(np.repeat(0.0,len(dfRatings))) #calculate mu - 3 x sigma: MSFT convention



    for index, row in dfRatings.iterrows():

        

        dfRatings.loc[dfRatings.index == index,'Rating'] = float(row['Rating_Raw'].mu) - 3 * float(row['Rating_Raw'].sigma)



    

    dfRatings['Name'] = dfRatings.index

    dfRatings = dfRatings.dropna()

    dfRatings.index = dfRatings['Rating'].rank(ascending=False).astype(int) #set index to ranking

    dfRatings.index.names = ['Rank']



 

    dfRatings.drop_duplicates(['Name','Rating','NumRegattas'],inplace=True)

    return dfRatings.sort_values('Rating',ascending=False) 
def group_by_school(df_ratings):

    df_school_ratings = pd.DataFrame()

    df_ratings = df_ratings[['school','Rating']]

    df_school_ratings = df_ratings.groupby(['school'],as_index=False).mean()

    return df_school_ratings
df_public_regattas_inter_champs_natls = df_public_regattas[((df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship') 

                                                           | (df_public_regattas['type'] == 'conference-championship') 

                                                           | (df_public_regattas['type'] == 'national-championship-semifina')) 

                                                           & (df_public_regattas['dt_num_divisions'] > 1)]

df_public_regattas_inter_champs_natls_coed = df_public_regattas_inter_champs_natls[df_public_regattas_inter_champs_natls['participant'] == 'coed']

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed, 'skipper')

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)



#reformat output

df_ratings_inter_champs_natls_coed = pd.merge(df_ratings_inter_champs_natls_coed, df_active_sailor, left_on='Name',right_on='id',how='outer')

df_school_ratings = group_by_school(df_ratings_inter_champs_natls_coed)

df_school_ratings = pd.merge(df_active_school[['id','name']], df_school_ratings, left_on='id', right_on='school', how='inner')

df_school_ratings.index.name = 'Rank'

df_school_ratings.columns.values[1] = 'SchoolName'

df_school_ratings = df_school_ratings.sort_values('Rating',ascending=False)

df_school_ratings.index = range(1,len(df_school_ratings) + 1)

df_school_ratings = df_school_ratings[np.isfinite(df_school_ratings['Rating'])]

df_school_ratings[['SchoolName','Rating']][:50]
df_public_regattas_inter_champs_natls = df_public_regattas[((df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship') 

                                                           | (df_public_regattas['type'] == 'conference-championship') 

                                                           | (df_public_regattas['type'] == 'national-championship-semifina')) 

                                                           & (df_public_regattas['dt_num_divisions'] > 1)]

df_public_regattas_inter_champs_natls_coed = df_public_regattas_inter_champs_natls[df_public_regattas_inter_champs_natls['participant'] == 'coed']

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed, 'skipper')

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)

#df_school_ratings_inter_champs_natls_coed = create_school_ratings(df_ratings_inter_champs_natls_coed)



#attach sailor name

df_ratings_inter_champs_natls_coed = pd.merge(df_ratings_inter_champs_natls_coed, df_active_sailor, left_on='Name',right_on='id',how='outer')



for index, row in df_ratings_inter_champs_natls_coed.iterrows():

    df_ratings_inter_champs_natls_coed['Name'] = df_ratings_inter_champs_natls_coed['first_name'] + " " + df_ratings_inter_champs_natls_coed['last_name']

df_ratings_inter_champs_natls_coed.index = np.arange(1, len(df_ratings_inter_champs_natls_coed) + 1)

df_ratings_inter_champs_natls_coed.index.names = ['Rank']

df_ratings_inter_champs_natls_coed[['Name', 'school','Rating','NumRegattas']][:50]
df_public_regattas_inter_champs_natls = df_public_regattas[((df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship') 

                                                           | (df_public_regattas['type'] == 'conference-championship') 

                                                           | (df_public_regattas['type'] == 'national-championship-semifina')) 

                                                           & (df_public_regattas['dt_num_divisions'] > 1)]

df_public_regattas_inter_champs_natls_coed = df_public_regattas_inter_champs_natls[df_public_regattas_inter_champs_natls['participant'] == 'women']

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed, 'skipper')

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)



#reformat output

df_ratings_inter_champs_natls_coed = pd.merge(df_ratings_inter_champs_natls_coed, df_active_sailor, left_on='Name',right_on='id',how='outer')

df_school_ratings = group_by_school(df_ratings_inter_champs_natls_coed)

df_school_ratings = pd.merge(df_active_school[['id','nick_name']], df_school_ratings, left_on='id', right_on='school', how='inner')

df_school_ratings.index.name = 'Rank'

df_school_ratings.columns.values[1] = 'SchoolName'

df_school_ratings = df_school_ratings.sort_values('Rating',ascending=False)

df_school_ratings.index = range(1,len(df_school_ratings) + 1)

df_school_ratings = df_school_ratings[np.isfinite(df_school_ratings['Rating'])]

df_school_ratings[['SchoolName','Rating']][:50]
df_public_regattas_inter_champs_natls = df_public_regattas[((df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship') 

                                                           | (df_public_regattas['type'] == 'conference-championship') 

                                                           | (df_public_regattas['type'] == 'national-championship-semifina')) 

                                                           & (df_public_regattas['dt_num_divisions'] > 1)]

df_public_regattas_inter_champs_natls_coed = df_public_regattas_inter_champs_natls[df_public_regattas_inter_champs_natls['participant'] == 'women']

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed, 'skipper')

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)

#df_school_ratings_inter_champs_natls_coed = create_school_ratings(df_ratings_inter_champs_natls_coed)



#attach sailor name

df_ratings_inter_champs_natls_coed = pd.merge(df_ratings_inter_champs_natls_coed, df_active_sailor, left_on='Name',right_on='id',how='outer')

for index, row in df_ratings_inter_champs_natls_coed.iterrows():

    df_ratings_inter_champs_natls_coed['Name'] = df_ratings_inter_champs_natls_coed['first_name'] + " " + df_ratings_inter_champs_natls_coed['last_name']

df_ratings_inter_champs_natls_coed.index = np.arange(1, len(df_ratings_inter_champs_natls_coed) + 1)

df_ratings_inter_champs_natls_coed.index.names = ['Rank']

df_ratings_inter_champs_natls_coed[['Name', 'school','Rating','NumRegattas']][:50]
df_public_regattas_inter_champs_natls = df_public_regattas[((df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship') 

                                                           | (df_public_regattas['type'] == 'conference-championship') 

                                                           | (df_public_regattas['type'] == 'national-championship-semifina')) 

                                                           & (df_public_regattas['dt_num_divisions'] > 1)]

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed, 'crew')

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)

#df_school_ratings_inter_champs_natls_coed = create_school_ratings(df_ratings_inter_champs_natls_coed)



#attach sailor name

df_ratings_inter_champs_natls_coed = pd.merge(df_ratings_inter_champs_natls_coed, df_active_sailor, left_on='Name',right_on='id',how='outer')

for index, row in df_ratings_inter_champs_natls_coed.iterrows():

    df_ratings_inter_champs_natls_coed['Name'] = df_ratings_inter_champs_natls_coed['first_name'] + " " + df_ratings_inter_champs_natls_coed['last_name']

df_ratings_inter_champs_natls_coed.index = np.arange(1, len(df_ratings_inter_champs_natls_coed) + 1)

df_ratings_inter_champs_natls_coed.index.names = ['Rank']

df_ratings_inter_champs_natls_coed[['Name', 'school','Rating','NumRegattas']][:50]