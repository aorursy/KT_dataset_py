# Libraries loading

import pandas as pd

import os

from datetime import datetime,timedelta

import warnings

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

from scipy import stats

from scipy.stats import norm, skew #statistics for normality and skewness

import numpy as np

ip = get_ipython()

ibe = ip.configurables[-1]

ibe.figure_formats = { 'pdf', 'png'}

warnings.filterwarnings("ignore")
DATA_DIR='/kaggle/input/atp-and-wta-tennis-data'

df_atp = pd.read_csv(os.path.join(DATA_DIR,"df_atp.csv"),index_col=0)

df_atp["Date"] =df_atp.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

df_atp.head()
print("Total number of matches : "+str(len(df_atp)))
print(list(df_atp.columns))
#verifying the shape of the dataset before droping the 'ATP' column.

print("Shape of the Dataset before droping the 'ATP' Column : {} ".format(df_atp.shape))



#Saving the column (maybe for later use ?)

df_atp_ID = df_atp['ATP']



#Droping the column 

df_atp.drop("ATP", axis = 1, inplace = True)



#verifying the shape of the dataset after droping the 'ATP' column

print("\nShape of the Dataset after droping the 'ATP' Column : {} ".format(df_atp.shape))
df_atp['Winner'].describe()
df_atp['Winner'].value_counts()[0:3]

#Return a Series containing counts of unique values in descending order so that the first element is the most frequently-occurring element.
df_atp['Loser'].describe()
df_atp['Loser'].value_counts()[0:3]
print("'Federer R.'  have won " + str(len(df_atp[df_atp['Winner']=='Federer R.']) )+" and lost " +str(len(df_atp[df_atp['Loser']=='Federer R.' ])))

print("'Nadal R.'    have won " + str(len(df_atp[df_atp['Winner']=='Nadal R.'])) +" and lost " +str(len(df_atp[df_atp['Loser']=='Nadal R.' ])))

print("'Djokovic N.' have won " + str(len(df_atp[df_atp['Winner']=='Djokovic N.'])) +" and lost " +str(len(df_atp[df_atp['Loser']=='Djokovic N.' ])))

print("'Lopez F.'    have won " + str(len(df_atp[df_atp['Winner']=='Lopez F.'])) +" and lost " +str(len(df_atp[df_atp['Loser']=='Lopez F.' ])))

print("'Youzhny M.'  have won " + str(len(df_atp[df_atp['Winner']=='Youzhny M.'])) +" and lost " +str(len(df_atp[df_atp['Loser']=='Youzhny M.' ])))

print("'Verdasco F.' have won " + str(len(df_atp[df_atp['Winner']=='Verdasco F.'])) +" and lost " +str(len(df_atp[df_atp['Loser']=='Verdasco F.' ])))
df_atp['Lsets']= pd.to_numeric(df_atp['Lsets'], errors='coerce')#tranforming str to numeric values and replcing with nan when we can't

N_sets = df_atp['Wsets'][df_atp['Winner']=='Federer R.'].sum() + df_atp['Lsets'][df_atp['Loser']=='Federer R.'].sum()



print('\nPlayer “Federer R.” won a total of : ' + str(N_sets) + ' sets.\n')
beg = datetime(2016,1,1)

end = datetime(2017,1,1)

df_atp_2016 = df_atp[(df_atp['Date']>=beg)&(df_atp['Date']<end)]
df_atp_2016['Wsets'][df_atp_2016['Winner']=='Federer R.'].sum() + df_atp_2016['Wsets'][df_atp_2016['Loser']=='Federer R.'].sum()
beg = datetime(2017,1,1)

end = datetime(2018,1,1)

df_atp_2017 = df_atp[(df_atp['Date']>=beg)&(df_atp['Date']<end)]
df_atp_2017['Wsets'][df_atp_2017['Winner']=='Federer R.'].sum() + df_atp_2017['Wsets'][df_atp_2017['Loser']=='Federer R.'].sum()
beg = datetime(2016,1,1)

end = datetime(2018,1,1)

df_atp_2017 = df_atp[(df_atp['Date']>=beg)&(df_atp['Date']<=end)]

df_atp_2017['Wsets'][df_atp_2017['Winner']=='Federer R.'].sum() + df_atp_2017['Wsets'][df_atp_2017['Loser']=='Federer R.'].sum()
unique_player_index_and_score = {}

#Dictionary containing the player name as a key and the tuple (player_unique_index,x,y)

#x : number_of_matches_won

#y : number_of_matches played

# x and y are intiated 0 in the bigining but as we go through the data set we increment x and y by 1 if the player wins a match

# or we increment only y with 1 if the player loses a matches

i=0

for player in df_atp['Winner'].unique():

    if player not in unique_player_index_and_score.keys():

        unique_player_index_and_score[player] = (i,0,0)

        i+=1

for player in df_atp['Loser'].unique():

    if player not in unique_player_index_and_score.keys():

        unique_player_index_and_score[player] = (i,0,0)

        i+=1

        

print('Number of unqiue player names : ',i)
winner_loser_score_tracking_vector = np.zeros((len(df_atp),2)) 

# two columns one to track the winner percetage and the other for the loser percentage 
#Sorting dataset by date so we can perform our calculation of the player prior win poucentage coorectly by looping one time trough the dataset
df_atp=df_atp.sort_values(by='Date')
for c,row in enumerate(df_atp[['Winner','Loser']].values):

    score_winner = unique_player_index_and_score[row[0]]#Winner up-to date score tracking from the dictionary 

    score_loser = unique_player_index_and_score[row[1]]#Loser up-to date score tracking from the dictionary

    #we consider new player that haven't yet played 5 matches as the have 20% of winning in the past 

    #(kind of a fair approach as they worked hard to get to play in the tournement:))

    if score_winner[2]<5:

        winner_loser_score_tracking_vector[c,0]=0.2

    else:

        winner_loser_score_tracking_vector[c,0] =score_winner[1]/score_winner[2]

    if score_loser[2]<5:

        winner_loser_score_tracking_vector[c,1]=0.2

    else:

        winner_loser_score_tracking_vector[c,1] = score_loser[1]/score_loser[2]

    #updating the dictionary based on the new outcome of the current match

    unique_player_index_and_score[row[0]] = (score_winner[0],score_winner[1]+1,score_winner[2]+1)#Winner

    unique_player_index_and_score[row[1]] = (score_loser[0],score_loser[1],score_loser[2]+1)#loser

    
df_atp['Winner_percentage'] = winner_loser_score_tracking_vector[:,0]

df_atp['Loser_percentage'] = winner_loser_score_tracking_vector[:,1]
df_atp['Winner_percentage'].describe()
df_atp['Loser_percentage'].describe()
sns.distplot(df_atp['Winner_percentage'], label="Winners")

sns.distplot(df_atp['Loser_percentage'], label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Winners and Losers prior win probabilityt')

plt.title('Winners and Losers prior win probability Distributionn')

plt.legend()
sns.distplot(df_atp['Winner_percentage'] , fit=norm);



#Récupèrer les paramètres ajustés utilisés par la fonction

(mu, sigma) = norm.fit(df_atp['Winner_percentage'])

#Tracer la ditribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Winner percentage Distribution')

fig = plt.figure()

res = stats.probplot(df_atp['Winner_percentage'], plot=plt)
sns.distplot(df_atp['Loser_percentage'] , fit=norm);



#Récupèrer les paramètres ajustés utilisés par la fonction

(mu, sigma) = norm.fit(df_atp['Loser_percentage'])

#Tracer la ditribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Loser pourcentage Distribution')

fig = plt.figure()

res = stats.probplot(df_atp['Loser_percentage'], plot=plt)
train_na = (df_atp.isnull().sum() / len(df_atp)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Pourcentage of missing values' :train_na})

missing_data
#With a visiualisation:

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=train_na.index, y=train_na)

plt.xlabel('Columns', fontsize=15)

plt.ylabel('Pourcentage of missing values', fontsize=15)

plt.title('Pourcentage of missing values by variables', fontsize=15)
#Drop the columns with missing values and that we won't be using:

for column in train_na.index[:26]:

    df_atp.drop(column, axis = 1, inplace = True)
#With a visiualisation:

train_na = (df_atp.isnull().sum() / len(df_atp)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Pourcentage of missing values' :train_na})

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=train_na.index, y=train_na)

plt.xlabel('Columns', fontsize=15)

plt.ylabel('Pourcentage of missing values', fontsize=15)

plt.title('Pourcentage of missing values by variables', fontsize=15)
df_atp.drop('W1', axis = 1, inplace = True)

df_atp.drop('L1', axis = 1, inplace = True)

df_atp.drop('W2', axis = 1, inplace = True)

df_atp.drop('L2', axis = 1, inplace = True)
sns.distplot(df_atp['WPts'].dropna(), label="Winners")

sns.distplot(df_atp['LPts'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
sns.distplot(df_atp['PSW'].dropna(), label="Winners")

sns.distplot(df_atp['PSL'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
df_atp['EXW']= pd.to_numeric(df_atp['EXW'], errors='coerce')
df_atp['EXW']= pd.to_numeric(df_atp['EXW'], errors='coerce')

sns.distplot(df_atp['EXW'].dropna(), label="Winners")

sns.distplot(df_atp['EXL'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
df_atp['B365W']= pd.to_numeric(df_atp['B365W'], errors='coerce')

sns.distplot(df_atp['B365W'].dropna(), label="Winners")

sns.distplot(df_atp['B365L'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
df_atp['Wsets']=pd.to_numeric(df_atp['Wsets'],errors='coerce' )

#df_atp['Lsets']=pd.to_numeric(df_atp['Lsets'],errors='coerce')

#df_atp['Wsets'].replace('scott', np.nan, inplace=True)

sns.distplot(df_atp['Wsets'].dropna(), label="Winners",kde=False)

sns.distplot(df_atp['Lsets'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
df_atp['LRank']=pd.to_numeric(df_atp['LRank'],errors='coerce' )

df_atp['WRank']=pd.to_numeric(df_atp['WRank'],errors='coerce' )

sns.distplot(df_atp['LRank'].dropna(), label="Winners")

sns.distplot(df_atp['WRank'].dropna(), label="Losers")

plt.ylabel('Frequency')

plt.xlabel('Pourcentage of victory in the past')

plt.title('Winners and Losers pourcentage Distribution')

plt.legend()
columns=['WPts','LPts','PSW','PSL','EXW','EXL','B365W','B365L','Lsets','Wsets','LRank','WRank']

for column in columns:

    df_atp[column]=df_atp[column].fillna(float(df_atp[column].mode()[0]))
train_na = (df_atp.isnull().sum() / len(df_atp)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Pourcentage of missing values' :train_na})

missing_data

#No more missing values
df_atp.columns
df_atp.drop("Comment", axis = 1, inplace = True)
df_atp.columns
df_atp.Tournament.describe()
df_atp.Location.describe()
len(df_atp)
df_atp.describe()
#winner prior sets winns pourcentage column

unique_player_index_and_score = {}

#Dictionary containing the player name as a key and the tuple (player_unique_index,x,y)

#x : number_of_set_won

#y : number_of_sets_played

# x and y are intiated 0 in the bigining but as we go through the data set we increment x Wsets(or Lsets) witch are the number of

# won by matches winner(orloser) and we increment y by Wsets+Lsets wich is the number of stes played in that match

i=0

for player in df_atp['Winner'].unique():

    if player not in unique_player_index_and_score.keys():

        unique_player_index_and_score[player] = (i,0,0)

        i+=1

for player in df_atp['Loser'].unique():

    if player not in unique_player_index_and_score.keys():

        unique_player_index_and_score[player] = (i,0,0)

        i+=1

        

print('Number of unqiue player names : ',i)

winner_loser_score_tracking_vector = np.zeros((len(df_atp),2)) 

# two columns one to track the winner percetage and the other for the loser percentage 

df_atp=df_atp.sort_values(by='Date')

for i in range(len(df_atp)):

    row=[df_atp.Winner[i],df_atp.Loser[i]]

    score_winner = unique_player_index_and_score[row[0]]#Winner up-to date set win score tracking from the dictionary 

    score_loser = unique_player_index_and_score[row[1]]#Loser up-to date  set win score tracking from the dictionary

    #we consider new player that haven't yet had 15 sets yet as they had a 20% of winning in the past 

    #(kind of a fair optimist approach as the worked hard to get to play in the tournement:))

    if int(score_winner[2])<15:

        winner_loser_score_tracking_vector[i,0]=0.2

    else:

        winner_loser_score_tracking_vector[i,0] =score_winner[1]/score_winner[2]

    if score_loser[2]<15:

        winner_loser_score_tracking_vector[i,1]=0.2

    else:

        winner_loser_score_tracking_vector[i,1] = score_loser[1]/score_loser[2]

    #updating the dictionary based on the new outcome of the current match

    unique_player_index_and_score[row[0]] = (score_winner[0],score_winner[1]+float(df_atp.Wsets[i]),score_winner[2]+float(df_atp.Wsets[i]+df_atp.Lsets[i]))#Winner

    unique_player_index_and_score[row[1]] = (score_loser[0],score_loser[1]+float(df_atp.Lsets[i]),score_loser[2]+float(df_atp.Wsets[i]+df_atp.Lsets[i]))#loser

    

df_atp['Winner_set_percentage'] = winner_loser_score_tracking_vector[:,0]

df_atp['Loser_set_percentage'] = winner_loser_score_tracking_vector[:,1]

df_atp['Winner_set_percentage'].describe()
sns.distplot(df_atp['Winner_set_percentage'])
df_atp['Loser_set_percentage'].describe()
sns.distplot(df_atp['Loser_set_percentage'])
#Not mine, I took from the internet But i got a full understanding of it :)

def compute_elo_rankings(data):

    """

    Given the list on matches in chronological order, for each match, computes 

    the elo ranking of the 2 players at the beginning of the match

    

    """

    print("Elo rankings computing...")

    players=list(pd.Series(list(data.Winner)+list(data.Loser)).value_counts().index)

    elo=pd.Series(np.ones(len(players))*1500,index=players)

    ranking_elo=[(1500,1500)]

    for i in range(1,len(data)):

        w=data.iloc[i-1,:].Winner

        l=data.iloc[i-1,:].Loser

        elow=elo[w]

        elol=elo[l]

        pwin=1 / (1 + 10 ** ((elol - elow) / 400))    

        K_win=32

        K_los=32

        new_elow=elow+K_win*(1-pwin)

        new_elol=elol-K_los*(1-pwin)

        elo[w]=new_elow

        elo[l]=new_elol

        ranking_elo.append((elo[data.iloc[i,:].Winner],elo[data.iloc[i,:].Loser])) 

        if i%5000==0:

            print(str(i)+" matches computed...")

    ranking_elo=pd.DataFrame(ranking_elo,columns=["elo_winner","elo_loser"])    

    ranking_elo["proba_elo"]=1 / (1 + 10 ** ((ranking_elo["elo_loser"] - ranking_elo["elo_winner"]) / 400))   

    return ranking_elo
Elo =  compute_elo_rankings(df_atp)
df_atp["Elo_Winner"] = Elo["elo_winner"]

df_atp["Elo_Loser"] = Elo["elo_loser"]

df_atp["Proba_Elo"]= Elo["proba_elo"]
sns.distplot(df_atp["Elo_Winner"], label="Winners")

sns.distplot(df_atp["Elo_Loser"], label="Losers")

plt.legend()
sns.distplot(df_atp["Proba_Elo"],fit=norm)
df_atp.drop(['Wsets','Lsets'], axis = 1, inplace = True)
target_1 = np.ones(len(df_atp))

target_2 = np.zeros(len(df_atp))

target_1 = pd.DataFrame(target_1,columns=['label'])

target_2 = pd.DataFrame(target_2,columns=['label'])
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce

print(df_atp.columns)
features_categorical = df_atp[["Series","Court","Surface","Round","Best of","Tournament"]].copy()

features_onehot = pd.get_dummies(features_categorical)

#tournaments_encoded = features_tournaments_encoding(df_atp)

#features_binary = pd.concat([features_categorical_encoded,tournaments_encoded],1)



## For the moment we have one row per match. 

## We "duplicate" each row to have one row for each outcome of each match. 

## Of course it isn't a simple duplication of  each row, we need to "invert" some features



# Elo data

elo_rankings = df_atp[["Elo_Winner","Elo_Loser","Proba_Elo"]]

elo_1 = elo_rankings

elo_2 = elo_1[["Elo_Loser","Elo_Winner","Proba_Elo"]]

elo_2.columns = ["Elo_Winner","Elo_Loser","Proba_Elo"]

elo_2.Proba_Elo = 1-elo_2.Proba_Elo

# Player prior win probability

win_pourcentage = df_atp[['Winner_percentage', 'Loser_percentage']]

win_1 = win_pourcentage

win_2 = win_1[['Loser_percentage','Winner_percentage']]

win_2.columns = ['Winner_percentage', 'Loser_percentage']

# Player prior win set probability

set_win_pourcentage = df_atp[['Winner_set_percentage','Loser_set_percentage']]

set_1 = set_win_pourcentage

set_2 = set_1[['Loser_set_percentage','Winner_set_percentage']]

set_2.columns = ['Winner_set_percentage','Loser_set_percentage']

# Player entry points

Pts = df_atp[['WPts','LPts']]

Pts_1 = Pts

Pts_2 = Pts_1[['LPts','WPts']]

Pts_2.columns = ['WPts','LPts']

# Player Entry Ranking

Rank = df_atp[['WRank','LRank']]

Rank_1 = Rank

Rank_2 = Rank_1[['LRank','WRank']]

Rank_2.columns = ['LRank','WRank']

#Player Odds for winning

Odds = df_atp[['EXW','EXL','PSW','PSL','B365W','B365L']]

Odds_1 = Odds

Odds_2 = Odds_1[['EXL','EXW','PSL','PSW','B365L','B365W']]

Odds_2.columns = ['EXW','EXL','PSW','PSL','B365W','B365L']

#Date 

Date_1 = df_atp.Date

Date_2 = df_atp.Date

elo_1.index = range(0,2*len(elo_1),2)

elo_2.index = range(1,2*len(elo_1),2)

win_1.index = range(0,2*len(win_1),2)

win_2.index = range(1,2*len(win_1),2)

set_1.index = range(0,2*len(set_1),2)

set_2.index = range(1,2*len(set_1),2)

Pts_1.index = range(0,2*len(Pts_1),2)

Pts_2.index = range(1,2*len(Pts_1),2)

Rank_1.index = range(0,2*len(Rank_1),2)

Rank_2.index = range(1,2*len(Rank_1),2)

Odds_1.index = range(0,2*len(Odds_1),2)

Odds_2.index = range(1,2*len(Odds_1),2)

Date_1.index = range(0,2*len(Date_1),2)

Date_2.index = range(1,2*len(Date_1),2)

target_1.index = range(0,2*len(target_1),2)

target_2.index = range(1,2*len(target_1),2)

features_elo_ranking = pd.concat([elo_1,elo_2]).sort_index(kind='merge')

features_win_pourcentage = pd.concat([win_1,win_2]).sort_index(kind='merge')

features_set_pourcentage = pd.concat([set_1,set_2]).sort_index(kind='merge')

features_Pts = pd.concat([Pts_1,Pts_2]).sort_index(kind='merge')

features_Rank =  pd.concat([Rank_1,Rank_2]).sort_index(kind='merge')

features_Odds = pd.concat([Odds_1,Odds_2]).sort_index(kind='merge')

target = pd.concat([target_1,target_2]).sort_index(kind='merge')

Date = pd.concat([Date_1,Date_2]).sort_index(kind='merge').to_frame()

'''

features_Odds.reset_index(drop=True, inplace=True)

features_elo_ranking.reset_index(drop=True, inplace=True)

#features_onehot.reset_index(drop=True, inplace=True)

features_win_pourcentage.reset_index(drop=True, inplace=True)

features_set_pourcentage.reset_index(drop=True, inplace=True)

features_set_pourcentage.reset_index(drop=True, inplace=True)

features_Pts.reset_index(drop=True, inplace=True)

features_Rank.reset_index(drop=True, inplace=True)

features_Odds.reset_index(drop=True, inplace=True)

target.reset_index(drop=True, inplace=True)

'''

features_onehot = pd.DataFrame(np.repeat(features_onehot.values,2, axis=0),columns=features_onehot.columns)

features_onehot.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_Odds.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_elo_ranking.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_win_pourcentage.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_set_pourcentage.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_Pts.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_Rank.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

features_Odds.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

target.set_index(pd.Series(range(0,2*len(df_atp))), inplace=True)

Date.set_index(pd.Series(range(0,2*len(df_atp))),inplace=True)

### Building of the pre final dataset 

# We can remove some features to see the effect on our model

features = pd.concat([features_win_pourcentage,

                  features_set_pourcentage,

                  features_elo_ranking,

                  features_Pts,

                  features_Rank,

                  features_Odds,

                  features_onehot,

                  Date,

                  target],1)



#Setting the 2019 matches as the test dataset.

#beg = datetime(2016,1,1)

end_train = datetime(2019,1,1)

beg_test = datetime(2019,1,1)

end_test = datetime(2020,1,1)

train = features[features['Date']<end_train]

test = features[(features['Date']>=beg_test)&(features['Date']<end_test)]
#For saving the features

#features.to_csv("df_atp_features.csv",index=False)
#loading after saveing

#features = pd.read_csv('df_atp_features.csv')
print(len(train))

print(len(test))
from sklearn.ensemble import RandomForestClassifier

df = features.drop(columns=['Date','label'])

feat_forest = RandomForestClassifier(n_jobs=-1)

feat_forest.fit(X=df, y=features['label'])



plt.figure(figsize=(10, 10))

feat_imp = feat_forest.feature_importances_



feat_imp, cols = zip(*sorted(zip(feat_imp, df.columns)))

feat_imp = np.array(feat_imp)[-30:]

cols = np.array(cols)[-30:]

d = {'feat_name': cols

    ,'feat_imp': feat_imp }

importance =  pd.DataFrame(data=d)

sns.barplot( x=  importance['feat_imp'],y = importance['feat_name']

           );

plt.yticks(range(len(cols[-30:])), cols[-30:])

plt.title("Features Relevance for Classification")

plt.xlabel("Relevance Percentage")
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

# We will be using the accuracy, precision,recall and the f1  as scores to asses our model performence

#Importing most important alogorithms 

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC#we will not be using SVM due tot he huge training time required on our dataset.

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis



from sklearn import model_selection #Cross-validation multiple scoring function



#features.drop(Odds_1.columns,axis=1,inplace=True)

X = train.drop(columns=['Date','label','EXW', 'EXL', 'PSW', 'PSL', 'B365W', 'B365L'])

Y = train['label']

# prepare configuration for cross validation test harness

seed = 42

# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA',QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(5, n_jobs=-1)))

models.append(('CART', DecisionTreeClassifier(max_depth=10)))

models.append(('NB', GaussianNB()))

#models.append(('SVM_linear', SVC(kernel="linear", C=0.025)))

#models.append(('SVM_',SVC(gamma=2, C=1)))

models.append(('RandomForest',RandomForestClassifier( n_estimators=100, n_jobs=-1)))

models.append(('MLP',MLPClassifier(alpha=0.0001)))

models.append(('ADABoost',AdaBoostClassifier()))



# evaluate each model in turn



results = []

scoring = {'accuracy': make_scorer(accuracy_score),

          'precision_score': make_scorer(precision_score),

          'recall_score' : make_scorer(recall_score),

          'f1_score' : make_scorer(f1_score)}

names = []

for name, model in models:

    stratifiedKFold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_validate(model, X, Y, cv=stratifiedKFold, scoring=scoring) 

    results.append(cv_results)

    names.append(name)

    msg ='-------------------------------------------------------------------------------------------------------------\n'

    msg = "Model : %s \n" % (name)

    msg = msg +'\n'

    msg =  msg + "Accuracy :  %f (%f)\n" % (cv_results['test_accuracy'].mean(),cv_results['test_accuracy'].std())

    msg =  msg + "Precision score :  %f (%f)\n" % (cv_results['test_precision_score'].mean(),cv_results['test_precision_score'].std())

    msg =  msg + "Recall score :  %f (%f)\n" % (cv_results['test_recall_score'].mean(),cv_results['test_recall_score'].std())

    msg =  msg + "F1 score :  %f (%f)\n" % (cv_results['test_f1_score'].mean(),cv_results['test_f1_score'].std())

    msg = msg + '------------------------------------------------------------------------------------------------------------\n'

    print(msg)
Accuracy = []

Precision = []

Recall = []

F1 = []

for idx,scores in enumerate(results):

    Accuracy.append(scores['test_accuracy'])

    Precision.append(scores['test_precision_score'])

    Recall.append(scores['test_recall_score'])

    F1.append(scores['test_f1_score'])

    
fig = plt.figure(figsize=(14,12))

fig.suptitle('Algorithms Comparison')

ax = fig.add_subplot(221)

plt.boxplot(Accuracy)

plt.title('Accuracy score')

ax.set_xticklabels(names)

ax = fig.add_subplot(222)

plt.boxplot(Precision)

plt.title('Precision Score')

ax.set_xticklabels(names)

ax = fig.add_subplot(223)

plt.boxplot(Recall)

ax.set_xticklabels(names)

plt.title('Recall score')

ax = fig.add_subplot(224)

plt.title('F1 score')

plt.boxplot(F1)

ax.set_xticklabels(names)



plt.show()



#now to test

from time import time



X_test = test.drop(columns=['Date','label','EXW', 'EXL', 'PSW', 'PSL', 'B365W', 'B365L'])

Y_test = test['label']



y_pred = []

train_time = []



for name, model in models:

    tic = time()

    model.fit(X, Y)

    toc = time()

    

    y_pred.append(model.predict(X_test))

    train_time.append(toc - tic)

    

    print("Classifier : {} ===> Training duration : {} sec".format(name, train_time[-1]))

    



    
reports = []

metrics = ["Classifier", "Accuracy", "Precision", "Recall", "F1-Score",'Training Duration (seconds)']

for idx, y_clf in enumerate(y_pred):

    acc = accuracy_score(Y_test, y_clf)

    pre = precision_score(Y_test, y_clf)

    rec = recall_score(Y_test, y_clf)

    f1s = f1_score(Y_test, y_clf)

    report = (models[idx][0], acc, pre, rec, f1s,train_time[idx])

    reports.append(report)       

display(pd.DataFrame.from_records(reports, columns=metrics))
reports = pd.DataFrame.from_records(reports, columns=metrics)

plt.figure(figsize=(10,10))

plt.plot(reports['Classifier'].values, reports['Accuracy'].values,

             label='Accuracy' )

plt.plot(reports['Classifier'], reports['Precision'], lw=1, alpha=0.6,

             label='Precision' )

plt.plot(reports['Classifier'], reports['Recall'], lw=1, alpha=0.6,

             label='Recall' )

plt.plot(reports['Classifier'], reports['F1-Score'], lw=1, alpha=0.6,

             label='F1-Score' )





plt.xlabel('Algorithm')

plt.ylabel('score')

plt.title('Algorithms comparison on test set')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, auc

from scipy import interp



y_prob = []



for name, model in models:

    y_prob.append(model.predict_proba(X_test)[:,1])

    

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)



i = 0

plt.figure(figsize=(10,10))

for idx, y_clf in enumerate(y_prob):

    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(Y_test, y_clf)

    tprs.append(interp(mean_fpr, fpr, tpr))

    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.6,

             label='ROC  Model %s (AUC = %0.2f)' % (models[idx][0], roc_auc))



    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

         label='Chance', alpha=.7)

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
betting_columns = ['EXL','EXW','PSL','PSW','B365L','B365W']

#Columns containg the Odds

Betting_Odds =  test[betting_columns]



#Our Capital will be 1500Euros for each strategy and for each betting site for a single model. 

budget_1 = 1500


import random





def rollDice():

    roll = random.randint(1,100)



    if roll == 100:

        return False

    elif roll <= 50:

        return False

    elif 100 > roll >= 50:

        return True





'''

Simple bettor, betting the same amount each time. This will be our baselane.

'''

def simple_bettor(data,y_true,budget):

    #return on investement for each betting site

    ROI_1 = budget

    ROI_2 = budget

    ROI_3 = budget

    wager = 10



    currentWager = 0



    for i in range(len(data)):

        if rollDice() and y_true.values[i]==1:

            ROI_1 += wager*(data['EXW'].values[i]-1)

            ROI_2 += wager*(data['PSW'].values[i]-1)

            ROI_3 += wager*(data['B365W'].values[i]-1)

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        elif rollDice() and y_true.values[i]==0:

            ROI_1 -= wager

            ROI_2 -= wager

            ROI_3 -= wager

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        elif not rollDice() and y_true.values[i]==0:

            ROI_1 += wager*(data['EXL'].values[i]-1)

            ROI_2 += wager*(data['PSL'].values[i]-1)

            ROI_3 += wager*(data['B365L'].values[i]-1)

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        else :

            ROI_1 -= wager

            ROI_2 -= wager

            ROI_3 -= wager

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



    if ROI_1<0:

        ROI_1 = 0

    if ROI_2<0:

        ROI_2 = 0

    if ROI_3<0:

        ROI_3 = 0

    return [(ROI_1-budget)/budget,(ROI_2-budget)/budget,(ROI_3-budget)/budget]







#If our model predict that a player is going to win, we'll invest 10Euros on that match for that player winning 

# and compare it with the real value to see if we won or lost



def strategy_1(data,y_pred,y_true,budget):

    '''

    

       If our model predict that a player is going to win, we'll invest 10Euros on that match for that player winning 

       and compare it with the real value to see if we won or lost

       

    '''

    #Retrun on investement for each betting site

    ROI_1 = budget

    ROI_2 = budget

    ROI_3 = budget

    for i in range(0,len(test)):

        if y_pred[i]==1 and y_true.values[i]==1.0:

            ROI_1 += 10*(data['EXW'].values[i]-1)

            ROI_2 += 10*(data['PSW'].values[i]-1)

            ROI_3 += 10*(data['B365W'].values[i]-1)

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        elif y_pred[i]==1 and y_true.values[i]==0.0:

            ROI_1 += -10

            ROI_2 += -10

            ROI_3 += -10

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        elif y_pred[i]==0 and y_true.values[i] == 1.0:

            #checking if we are already broke

            ROI_1 += -10

            ROI_2 += -10

            ROI_3 += -10

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



        else :

            ROI_1 += 10*(data['EXL'].values[i]-1)

            ROI_2 += 10*(data['PSL'].values[i]-1)

            ROI_3 += 10*(data['B365L'].values[i]-1)

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                

            

    if ROI_1<0:

        ROI_1 = 0

    if ROI_2<0:

        ROI_2 = 0

    if ROI_3<0:

        ROI_3 = 0

    return [(ROI_1-budget)/budget,(ROI_2-budget)/budget,(ROI_3-budget)/budget]



def strategy_2(data,y_proba,y_true,budget):

    '''

    

      In each match we'll invest 10(probability_player_win)Euros for the player winning, and 10(probability_player_lose)Euros

      for the player losing



    

    '''

    ROI_1 = budget

    ROI_2 = budget

    ROI_3 = budget

    for i in range(0,len(test)):

        if y_true.values[i]==1.0:

            ROI_1 += y_proba[i]*10*(data['EXW'].values[i]-1) -(1- y_proba[i])*10

            ROI_2 += y_proba[i]*10*(data['PSW'].values[i]-1) - (1-y_proba[i])*10

            ROI_3 += y_proba[i]*10*(data['B365W'].values[i]-1) - (1-y_proba[i])*10

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                

        else :

            ROI_1 += (1-y_proba[i])*10*(data['EXL'].values[i]-1) - y_proba[i]*10

            ROI_2 += (1-y_proba[i])*10*(data['PSL'].values[i]-1) - y_proba[i]*10

            ROI_3 += (1-y_proba[i])*10*(data['B365L'].values[i]-1) - y_proba[i]*10

            #checking if we are already broke

            if ROI_1<=0:

                ROI_1 = -100000000000000000000

            if ROI_2<=0:

                ROI_2 = -100000000000000000000

            if ROI_3<=0:

                ROI_3 = -100000000000000000000                



    if ROI_1<0:

        ROI_1 = 0

    if ROI_2<0:

        ROI_2 = 0

    if ROI_3<0:

        ROI_3 = 0

    return [(ROI_1-budget)/budget,(ROI_2-budget)/budget,(ROI_3-budget)/budget]





#P.S: Seing how we contructed the dataset (Each row repeated one time). We'll be actualy investing 20Euros of our capital in each match instead of 10



        
#Our Capital will be 1500Euros for each strategy and for each betting site for a single model. 

reports = []

metrics = ["Classifier",  "Strat 1 EX", "Strat 2 EX", "Strat 1 PS", "Strat 2 PS", "Strat 1 B365", "Strat 2 B365" ,'Random EX', 'Random PS','Random B365']

for idx, y_clf in enumerate(y_pred):

    Random = simple_bettor(Betting_Odds ,Y_test,budget_1)

    strat_1 = strategy_1(Betting_Odds,y_clf,Y_test,budget_1)

    strat_2 = strategy_2(Betting_Odds,y_prob[idx],Y_test,budget_1)

    report = (models[idx][0],strat_1[1],strat_2[1],strat_1[1],strat_2[1],strat_1[2],strat_2[2],Random[0],Random[1],Random[2])

    reports.append(report)       

display(pd.DataFrame.from_records(reports, columns=metrics))
reports = pd.DataFrame.from_records(reports, columns=metrics)

plt.figure(figsize=(10,10))

plt.plot(reports['Classifier'].values, reports['Strat 1 EX'].values,

             label='EX : Strategy 1' )

plt.plot(reports['Classifier'], reports['Strat 2 EX'], lw=1, alpha=0.6,

             label='EX : Strategy 2' )

plt.plot(reports['Classifier'], reports['Strat 1 PS'], lw=1, alpha=0.6,

             label='PS : Strategy 1' )

plt.plot(reports['Classifier'], reports['Strat 2 PS'], lw=1, alpha=0.6,

             label='PS : Strategy 2' )

plt.plot(reports['Classifier'], reports['Strat 1 B365'], lw=1, alpha=0.6,

             label='B365 : Strategy 1' )

plt.plot(reports['Classifier'], reports['Strat 2 B365'], lw=1, alpha=0.6,

             label='B365 : Strategy 2' )





plt.xlabel('Algorithm')

plt.ylabel('Return on investement')

plt.title('Algorithms ROI on Test set')

plt.legend(loc="lower right")

plt.show()