#library importations.

import pandas as pd

import numpy as np
# Create dataframe

raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 

        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 

        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 

        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],

        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

#df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])

#Alternatively

df = pd.DataFrame(raw_data, columns = raw_data.keys())



df
# Create a groupby variable that groups preTestScores by regiment

groupby_regiment = df['preTestScore'].groupby(df['regiment'])

groupby_regiment
#View a grouping

#Use list() to show what a grouping looks like



list(groupby_regiment)
#Descriptive statistics by group

df['preTestScore'].groupby(df['regiment']).describe()
#Mean of each regiment’s preTestScore

df.groupby(['regiment']).mean()
#Mean preTestScores grouped by regiment and company

df['preTestScore'].groupby([df['regiment'], df['company']]).mean()

#ean preTestScores grouped by regiment and company without heirarchical indexing

df['preTestScore'].groupby([df['regiment'], df['company']]).mean().unstack()
#Group the entire dataframe by regiment and company

df.groupby(['regiment', 'company']).mean()
#Number of observations in each regiment and company

df.groupby(['regiment', 'company']).size()
#Iterate an operations over groups

# Group the dataframe by regiment, and for each regiment,

for name, group in df.groupby('regiment'): 

    # print the name of the regiment

    print(name)

    # print the data of that regiment

    print(group)
#In the dataframe “df”, group by “regiments, take the mean values of the other variables for those groups, then display them

#with the prefix_mean

df.groupby('regiment').mean().add_prefix('mean_')
#Create a function to get the stats of a group

def get_stats(group):

    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
#Create bins and bin up postTestScore by those pins

bins = [0, 25, 50, 75, 100]

group_names = ['Low', 'Okay', 'Good', 'Great']

df['categories'] = pd.cut(df['postTestScore'], bins, labels=group_names)
#Apply the get_stats() function to each postTestScore bin

df['postTestScore'].groupby(df['categories']).apply(get_stats).unstack()