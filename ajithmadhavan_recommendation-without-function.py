import pandas as pd

#import the training data and the recommendation file. Recommendation file contains the list of user ids for which we need to perform recommendations

df=pd.read_csv("../input/train.csv")

rec=pd.read_csv("../input/recommendations.csv",names=['user_id'])

df.head()

rec[:5]
df.loc[df['listen_count']>20]
# do a pivot to categorize the songs corresponding to each user. The column values represents the number of times the user heard that song

table1=df.pivot_table(index='user_id',columns='song_id',values='listen_count')

table1
table1.index
# find the correlation between each songs. This would give a clear understanding of how many songs 

TableB = table1.corr(method='pearson')

TableB.head()
TableB['SOBZZDU12A6310D8A3'].dropna()
table1.loc['0007c0e74728ca9ef0fe4eb7f75732e8026a278b'].dropna()
table1.iloc[0].dropna().index
user_corr=pd.Series()

recomend={}

user_id=0

for song in table1.iloc[user_id].dropna().index:

    corr_list=TableB[song].dropna()*table1.iloc[user_id][song]

    user_corr=user_corr.append(corr_list)

print(user_corr)
user_corr=user_corr.groupby(user_corr.index).sum()

user_corr.sort_values(ascending=False)
sonlistUnHeard=[]

for each in range(len(table1.iloc[user_id].dropna().index)):

    if table1.iloc[user_id].dropna().index[each] in user_corr:

        sonlistUnHeard.append(table1.iloc[user_id].dropna().index[each])

user_corr=user_corr.drop(sonlistUnHeard)
print("The list of songs that user {} has listened".format(table1.index[user_id]))

for songs in table1.iloc[user_id].dropna().index:

    print(songs)
print("The list of songs that can be recommended to user {} ".format(table1.index[0]))

recomendSongs=[]

for songs in user_corr.sort_values(ascending=False).index[:]:

    recomendSongs.append(songs)

recomend['0007c0e74728ca9ef0fe4eb7f75732e8026a278b']=recomendSongs

recomend