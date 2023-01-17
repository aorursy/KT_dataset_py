import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



anime = pd.read_csv("../input/anime.csv")

rating = pd.read_csv("../input/rating.csv")

top_num=100 #number of most popular anime in consideration





# Select top popular anime:

anime_top=anime[anime['members']>1000].sort_values(by=['rating','members'],ascending =False)[0:top_num]

anime_top=anime_top.set_index('anime_id')



# Dataframe for top popular anime:

# Dataframe will contain information for every user, if he had watched anime from top list

top_score_columns=['user_id']

for name in anime_top['name'].unique():

    #top_score_columns+=[name]

    top_score_columns+=[name+'_watch']

top_score=pd.DataFrame(columns=top_score_columns)





#Filling dataframe with first info from first 1000000 lines from 

user_id=0

for index, row in rating.sort_values('user_id')[0:100000].iterrows():

    #print(type(row['anime_id']))

    if(row['anime_id'] in anime_top.index):

        if(row['user_id']!=user_id):

            top_score.loc[top_score.shape[0]]=[row['user_id'].astype('int')]+list(np.full([top_score.shape[1]-1], np.nan))

            user_id=row['user_id'].astype('int')

        #print(row['anime_id'])

        anime_id=row['anime_id']

        #top_score.loc[top_score.shape[0]-1].loc[anime_top.loc[row['anime_id']]['name']]=row['rating']

        top_score.loc[top_score.shape[0]-1].loc[anime_top.loc[row['anime_id']]['name']+'_watch']=1

#top_score



#Fill Nan wit zeros, if user hasnt watched this anime

for col in top_score.columns:

    if(col!='user_id'):

        if('_watch' in col):

            top_score[col]=top_score[col].fillna(0)

        else:

            top_score[col]=top_score[col].fillna(top_score[col].mean())

            top_score[col]=top_score[col].replace(-1,top_score[col].mean())

top_score=top_score.fillna(0)





# We wont use info about Steins;Gate Movie, because it may correlate with Steins;Gate anime data

from sklearn.model_selection import train_test_split

X_sg = np.array(top_score.drop(columns=['user_id','Steins;Gate_watch','Steins;Gate Movie: Fuka Ryouiki no Déjà vu_watch']))

y_sg = np.array(top_score['Steins;Gate_watch'])



X_train, X_test, y_train, y_test = train_test_split(

    X_sg, y_sg, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

forest = RandomForestClassifier(n_estimators=50, max_depth=8,

                                  random_state=0)

tree = DecisionTreeClassifier(random_state=0)



print("train with all indexes:")

forest.fit(X_train,y_train)

tree.fit(X_train,y_train)

print("tree score:",tree.score(X_test,y_test))

print("forest score:",forest.score(X_test,y_test))



#Lets see, how good algorithms work for users, who has watched Steins;Gate 

indices=[i for i, x in enumerate(y_test) if x ==1]

tree.score(X_test[indices],y_test[indices])

print("tree score if watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if watched:",forest.score(X_test[indices],y_test[indices]))
np.unique(y_sg, return_counts=True)
#number of samples for y=1 and y=0 is equal

new_ind=np.append(np.where(y_train==1)[0], np.where(y_train==0)[0][0:len(np.where(y_train==1)[0])])



print("\ntrain with special indexes:")

forest.fit(X_train[new_ind],y_train[new_ind])

tree.fit(X_train[new_ind],y_train[new_ind])

print("tree score:",tree.score(X_test,y_test))

print("forest score:",forest.score(X_test,y_test))



indices=[i for i, x in enumerate(y_test) if x ==1]

tree.score(X_test[indices],y_test[indices])

print("tree score if watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if watched:",forest.score(X_test[indices],y_test[indices]))



indices=[i for i, x in enumerate(y_test) if x ==0]

tree.score(X_test[indices],y_test[indices])

print("tree score if not watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if not watched:",forest.score(X_test[indices],y_test[indices]))