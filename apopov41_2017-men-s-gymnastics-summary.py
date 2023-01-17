'''

2017 Artistic Gymnastics World Championships - Men's Results Dataset 

Number of Medals for Each Country



Creating dataframe that holds countries and their corresponding medals of each color.

Then, creating dataframe that counts the medals won by each country

'''



from pandas import DataFrame, Series



def create_dataframe():

    countries = ['China', 'Croatia', 'Great Britain', 'Greece', 'Israel', 'Japan', 'Korea', 'Netherlands', 'Russian Fed.',

                'Ukraine', 'United States']

    gold = [2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0]

    silver = [1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0]

    bronze = [2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    

    medal_counts_df = DataFrame({'country_name': countries, 'gold': gold, 'silver': silver, 'bronze': bronze},

                               columns = ['country_name', 'gold', 'silver', 'bronze'])

    return medal_counts_df

print(create_dataframe())
'''

Get Average Medal Count

'''

import numpy as np

from pandas import DataFrame, Series



def create_average_count():

    countries = ['China', 'Croatia', 'Great Britain', 'Greece', 'Israel', 'Japan', 'Korea', 'Netherlands', 'Russian Fed.',

                'Ukraine', 'United States']

    gold = [2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0]

    silver = [1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0]

    bronze = [2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    

    medal_count = {'country_name': countries,

                  'gold': Series(gold),

                  'silver': Series(silver),

                  'bronze': Series(bronze)}

    medal_count_df = DataFrame(medal_count)

    

    average_medal_count = medal_count_df[['gold', 'silver', 'bronze']].apply(np.mean)

    return average_medal_count

print(create_average_count())
'''

Get Average Bronze Medal Count for Countries who Won At Least One Gold

'''

import numpy as np

from pandas import DataFrame, Series



def create_average_count():

    countries = ['China', 'Croatia', 'Great Britain', 'Greece', 'Israel', 'Japan', 'Korea', 'Netherlands', 'Russian Fed.',

                'Ukraine', 'United States']

    gold = [2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0]

    silver = [1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0]

    bronze = [2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    

    medal_count = {'country_name': countries,

                  'gold': Series(gold),

                  'silver': Series(silver),

                  'bronze': Series(bronze)}

    medal_count_df = DataFrame(medal_count)

    average_bronze_at_least_one_gold = np.mean(medal_count_df.bronze[medal_count_df.gold > 0])

    return average_bronze_at_least_one_gold

print(create_average_count())
'''

Get Average Silver Medal Count for Countries who Won At Least One Gold

'''

import numpy as np

from pandas import DataFrame, Series



def create_average_count():

    countries = ['China', 'Croatia', 'Great Britain', 'Greece', 'Israel', 'Japan', 'Korea', 'Netherlands', 'Russian Fed.',

                'Ukraine', 'United States']

    gold = [2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0]

    silver = [1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0]

    bronze = [2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    

    medal_count = {'country_name': countries,

                  'gold': Series(gold),

                  'silver': Series(silver),

                  'bronze': Series(bronze)}

    medal_count_df = DataFrame(medal_count)

    average_silver_at_least_one_gold = np.mean(medal_count_df.silver[medal_count_df.gold > 0])

    return average_silver_at_least_one_gold

print(create_average_count())
'''

Get Average Gold Medal Count for Countries who Won At Least One Gold

'''

import numpy as np

from pandas import DataFrame, Series



def create_average_count():

    countries = ['China', 'Croatia', 'Great Britain', 'Greece', 'Israel', 'Japan', 'Korea', 'Netherlands', 'Russian Fed.',

                'Ukraine', 'United States']

    gold = [2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0]

    silver = [1, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0]

    bronze = [2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    

    medal_count = {'country_name': countries,

                  'gold': Series(gold),

                  'silver': Series(silver),

                  'bronze': Series(bronze)}

    medal_count_df = DataFrame(medal_count)

    average_gold_at_least_one_gold = np.mean(medal_count_df.gold[medal_count_df.gold > 0])

    return average_gold_at_least_one_gold

print(create_average_count())
'''

Get Number of Placement Points for Each Country.

'''

import numpy as np

from pandas import DataFrame, Series



def create_placement_points_count():

    countries = ['Armenia', 'Brazil', 'Chile', 'China', 'Croatia', 'Cuba', 'France', 'Great Britain', 'Greece', 'Guatemala', 

                 'Israel', 'Japan', 'Korea', 'Netherlands', 'Germany', 'Russian Fed.', 'Ukraine', 'United States', 

                 'Turkey', 'Romania', 'Switzerland', 'Slovenia']

    

    gold = [0, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    silver = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0]

    bronze = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]

    fourth = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]

    fifth = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    sixth = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]

    seventh = [0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]

    eighth = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1]

    

    placement_counts = {'country_name': Series(countries),

                       'gold': Series(gold),

                       'silver': Series(silver),

                       'bronze': Series(bronze),

                       'fourth': Series(fourth),

                       'fifth': Series(fifth),

                       'sixth': Series(sixth),

                       'seventh': Series(seventh),

                       'eighth': Series(eighth)}

    placement_counts_df = DataFrame(placement_counts)

    

    placement_scores = placement_counts_df[['gold', 'silver', 'bronze', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth']]

    '''

    assigning points to each top-eight placing or medal

    '''

    points = np.dot(placement_scores, [10, 8, 6, 5, 4, 3, 2, 1])

    '''

    counting points earned by each country

    '''

    worlds_points = {'country_name': Series(countries),

                    'points': Series(points)}

    '''

    creating the worlds points dataframe that holds the points scored by each country

    '''

    worlds_points_df = DataFrame(worlds_points)

    print(worlds_points_df)

create_placement_points_count()    
# print the first five rows of the dataset

import csv

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline

gymnastics_df = pd.read_csv('../input/World_Champs_Men\'s_All-Around.csv')

gymnastics_df.head(5)
# get the unique apparatus list

apparatus_list = gymnastics_df.Apparatus.unique().tolist()

apparatus_list
# get average score for each apparatus

total_apparatus_score_df = gymnastics_df[['Apparatus', 'Total']].copy()

total_apparatus_score_df.head(10)



total_apparatus_score_df.sort_values('Apparatus')



mean_apparatus_score = total_apparatus_score_df.groupby(['Apparatus'])['Total'].mean()

print("Mean Apparatus Scores")

mean_apparatus_score
mean_apparatus_score.plot(title='Mean Score per Apparatus')

plt.show()
'''

Printing the first 48 rows of the dataset.

'''

name_rank_apparatus_df = gymnastics_df[['Name', 'Apparatus', 'Rank']].copy()

name_rank_apparatus_df.head(48)
diff_vs_exec_df = gymnastics_df[['Diff', 'Exec', 'Apparatus', 'Rank', 'Name']].copy()

diff_vs_exec_df.drop_duplicates()
# get maximum difficulty score for each apparatus

diff_apparatus_score_df = gymnastics_df[['Apparatus', 'Diff']].copy()

diff_apparatus_score_df.head(10)



diff_apparatus_score_df.sort_values('Apparatus')



max_apparatus_diff_score = diff_apparatus_score_df.groupby(['Apparatus'])['Diff'].max()

print("Maximum Apparatus Difficulty Scores")

max_apparatus_diff_score
# get minimum difficulty score for each apparatus

min_apparatus_diff_score = diff_apparatus_score_df.groupby(['Apparatus'])['Diff'].min()

print("Minimum Apparatus Difficulty Scores")

min_apparatus_diff_score
# get mean difficulty score for each apparatus

mean_apparatus_diff_score = diff_apparatus_score_df.groupby(['Apparatus'])[('Diff')].mean()

print("Mean Apparatus Difficulty Scores")

mean_apparatus_diff_score
# get mean execution score for each apparatus

exec_apparatus_score_df = gymnastics_df[['Apparatus', 'Exec']].copy()

exec_apparatus_score_df.head(10)



exec_apparatus_score_df.sort_values('Apparatus')



mean_apparatus_exec_score = exec_apparatus_score_df.groupby(['Apparatus'])['Exec'].mean()

print("Average Apparatus Execution Scores")

mean_apparatus_exec_score
# get maximum execution score for each apparatus

max_apparatus_exec_score = exec_apparatus_score_df.groupby(['Apparatus'])['Exec'].max()

print("Maximum Apparatus Execution Scores")

max_apparatus_exec_score
# get minimum execution score for each apparatus

min_apparatus_exec_score = exec_apparatus_score_df.groupby(['Apparatus'])[('Exec')].min()

print("Minimum Apparatus Execution Scores")

min_apparatus_exec_score
mean_apparatus_score.plot(title='Mean Score per Apparatus')

plt.show()
mean_apparatus_exec_score.plot(title="Execution Score Vs. Difficulty Score")

mean_apparatus_diff_score.plot()

plt.show()
max_apparatus_exec_score.plot()

max_apparatus_diff_score.plot()

plt.show()
min_apparatus_exec_score.plot()

min_apparatus_diff_score.plot()

plt.show()