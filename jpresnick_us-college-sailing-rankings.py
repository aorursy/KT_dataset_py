import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import trueskill as ts

from IPython.display import display

pd.set_option('display.max_rows', 300)
#what is are the dates  of the current season

pd.read_csv('../input/season.tsv',sep='\t')[-2:-1]
#pull the completed regattas for this season

df_public_regattas = pd.read_csv('../input/public_regatta.tsv',sep='\t')

df_public_regattas['end_date'] = pd.to_datetime(df_public_regattas['end_date'])

df_public_regattas = df_public_regattas[(df_public_regattas['end_date'] > '2016-08-16') & (df_public_regattas['end_date'] < '2017-01-14')]

df_public_regattas = df_public_regattas[(df_public_regattas['dt_status'] == 'finished') | (df_public_regattas['dt_status'] == 'final')]

df_public_regattas = df_public_regattas[(df_public_regattas['scoring'] == 'standard') | (df_public_regattas['scoring'] == 'combined')]
#pull the teams and ranks in each regatta

df_team = pd.read_csv('../input/team.tsv',sep='\t')

df_team = df_team[(df_team['school'].notnull()) | (df_team['name'].notnull())]


#compile into a results table for ranking

def compile_results(df_public_regattas):

    df_results = pd.DataFrame()

    for index, row in df_public_regattas.iterrows():

      

        df_results_temp = df_team[['school','name','dt_rank']][df_team['regatta'] == row['id']]

        df_results_temp.index = df_results_temp['school'] + ' ' + df_results_temp['name']

        del(df_results_temp['school'])

        del(df_results_temp['name'])

        df_results_temp.columns = [row['id']]

        df_results_temp.index.names = ['school_name']

    

        

        df_results = pd.merge(df_results,df_results_temp,left_index=True,right_index=True,how='outer')

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



 

    

    return dfRatings.sort_values('Rating',ascending=False) 
##extract and group by school

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
df_public_regattas_teamrace = df_public_regattas[df_public_regattas['scoring'] == 'team']

df_results_teamrace = compile_results(df_public_regattas_teamrace)

df_ratings_teamrace = doRating(df_results_teamrace)

df_school_ratings_teamrace = create_school_ratings(df_ratings_teamrace)

display(df_school_ratings_teamrace)
df_public_regattas_inter_champs_natls = df_public_regattas[(df_public_regattas['type'] == 'intersectional') | (df_public_regattas['type'] == 'championship')]

df_public_regattas_inter_champs_natls_coed = df_public_regattas_inter_champs_natls[df_public_regattas_inter_champs_natls['participant'] == 'coed']

display(df_public_regattas_inter_champs_natls_coed)

df_results_inter_champs_natls_coed = compile_results(df_public_regattas_inter_champs_natls_coed)

df_ratings_inter_champs_natls_coed = doRating(df_results_inter_champs_natls_coed)

df_school_ratings_inter_champs_natls_coed = create_school_ratings(df_ratings_inter_champs_natls_coed)

display(df_school_ratings_inter_champs_natls_coed)