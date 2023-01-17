import pandas as pd

df=pd.read_csv("../input/train.csv")

rec=pd.read_csv("../input/recommendations.csv",names=['user_id'])



table1=df.pivot_table(index='user_id',columns='song_id',values='listen_count')



TableB = table1.corr(method='pearson')



user_corr=pd.Series()

recomend={}
user_ids=[x for x in rec['user_id']]

user_ids
table1.loc['2151970107e08d58919003899f952b64af0ee0ec'].dropna()
TableB['SOKOTZG12A6D4F9519'].dropna()
recomend={}

def recomendationDict(user_corr,user):

    recomendSongs=[]

    #print(user_corr)

    #print(user)

    for songs in user_corr.sort_values(ascending=False).index[:10]:

        recomendSongs.append(songs)

    recomend[user]=recomendSongs

    return recomend



for user_id in user_ids:

    user_corr=pd.Series()

    

    for song in table1.loc[user_id].dropna().index:

        corr_list=TableB[song].dropna()*table1.loc[user_id][song]

        user_corr=user_corr.append(corr_list)

    #print(user_corr)



    user_corr=user_corr.groupby(user_corr.index).sum()

    user_corr.sort_values(ascending=False)

    

    songlistUnHeard=[]

    for each in range(len(table1.loc[user_id].dropna().index)):

        if table1.loc[user_id].dropna().index[each] in user_corr:

            songlistUnHeard.append(table1.loc[user_id].dropna().index[each])

    if len(user_corr)>1:

        user_corr=user_corr.drop(songlistUnHeard)

    else:

        pass

    recomendationDict(user_corr,user_id)



    #print("The list of songs that user {} has listened".format(table1.index[user_id]))

    #for songs in table1.iloc[user_id].dropna().index:

        #print(songs)

    

recomend.keys()
recomend['2151970107e08d58919003899f952b64af0ee0ec']
final=pd.DataFrame.from_dict(recomend, orient='index')
final.to_csv("final.csv")
ls
readDF=pd.read_csv("final.csv")
readDF