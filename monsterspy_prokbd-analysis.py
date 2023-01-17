import os

import pandas as pd

import numpy as np

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns
Df_team = pd.read_csv(f"/kaggle/input/prokabadi/ProKabadi_match_level_details/DS_team.csv")

Df_match = pd.read_csv(f"//kaggle/input/prokabadi/ProKabadi_match_level_details/DS_match.csv")

Df_Players = pd.read_csv(f"/kaggle/input/prokabadi/ProKabadi_match_level_details/DS_players.csv")
display(Df_match.shape)



Df_match_clean =Df_match[Df_match.series_name.str.contains("Pro Kabaddi")]

Df_match_clean.isna().sum(axis=0)
Df_match_clean =Df_match[Df_match.series_name.str.contains("Pro Kabaddi")]

Df_match_clean = Df_match_clean[Df_match_clean.result.notnull()]

Df_match_clean = Df_match_clean.drop(["home_team_id","home_team_name","player_name_of_the_match","status","match_number"],axis=1)



# fill the player of the id for Null cases



for i  in Df_match_clean[Df_match_clean.player_id_of_the_match.isna()].match_id:

    _temp=Df_Players[Df_Players.match_id ==i].sort_values(by="player_total_points",ascending =False).head(1)

    Df_match_clean.loc[Df_match_clean.match_id ==i,"player_id_of_the_match"] = int(_temp.player_id)    

    



Df_match_clean[(Df_match_clean.series_id ==30) | (Df_match_clean.series_id ==32)] 



# drop the 

Df_match_clean=Df_match_clean.drop([8,657])



#on Each day you will have two matches. based on that creating categorical column is it 1st or second match

_dic ={"19:30":"1","20:00":"1","20:30":"2","21:00":"2","21:15":"2"}

Df_match_clean["match_of_the_day"]=Df_match_clean.start_time.map(_dic)



# Deleting not needed columns

Df_match_clean=Df_match_clean.drop(["start_time","Unnamed: 0"],axis=1)



# Creating match level inputs

_info_cols = ["id","name"]

Df_match_clean["Team_A_id"] = 0

Df_match_clean["Team_B_id"] = 0

Df_match_clean["Team_A_name"] = ""

Df_match_clean["Team_B_name"] = ""

Df_match_clean["Team_A_score"] = 0

Df_match_clean["Team_B_score"] = 0



for i in Df_match_clean.match_id:

    _temp =Df_team[Df_team.match_id ==i]

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_A_id"] = _temp.iloc[0,1]

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_A_name"] = _temp.iloc[0,3]

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_A_score"] = _temp.iloc[0,4]

    

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_B_id"] = _temp.iloc[1,1]

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_B_name"] = _temp.iloc[1,3]

    Df_match_clean.loc[Df_match_clean.match_id ==i,"Team_B_score"] = _temp.iloc[1,4]

    



Df_match_clean["temp"] =  Df_match_clean.result.str.split("beat")

Df_match_clean["winner"] =[i[0] for i in Df_match_clean.temp]

Df_match_clean=Df_match_clean.drop("temp",axis=1)



#Deleting Temp column

Df_match_clean=Df_match_clean.drop("result",axis=1)



#replace Sraw Text

Df_match_clean.winner=[ "Draw" if "Match" in i else i for i in Df_match_clean.winner]
Df_match_clean.groupby("winner")["winner"].count().plot(kind="bar")

plt.show()
# Design Target Variable 

Df_match_clean["Target"] = [1 if i== True else 2 for i in (Df_match_clean.winner.str.strip() == Df_match_clean.Team_A_name.str.strip())]

#Df_match_clean["Target"] = [0 if i=="Draw" else i for i in Df_match_clean.winner ]

Df_match_clean.loc[(Df_match_clean.winner == "Draw"),"Target"] =0





Df_match_clean["Team_A_points"] = Df_match_clean.Target.map({0:0.5,1:1,2:0})

Df_match_clean["Team_B_points"] = Df_match_clean.Target.map({0:0.5,1:0,2:1})





pd.pivot_table(Df_match_clean,index=["series_name"],columns=["Target"],values=["winner"],aggfunc="count").plot(kind="bar",title="Seasion Wise Winners Team A vs Team B")
# Identified Team ID is Unque so we can drive based on team ID

display(Df_match_clean[["Team_B_id","Team_B_name"]].drop_duplicates().sort_values(by="Team_B_id"))

Df_match_clean[["Team_A_id","Team_A_name"]].drop_duplicates().sort_values(by="Team_A_id")

# Update Overall Performence

_temp1 =Df_match_clean[["Team_A_id","Team_A_points"]]

_temp1.columns =["id","points"]

_temp2 =Df_match_clean[["Team_B_id","Team_B_points"]]

_temp2.columns =["id","points"]

_temp1=pd.concat([_temp1,_temp2],axis=0,sort=False)

_temp1["dd"] =1

_temp =pd.pivot_table(data=_temp1,index="id",columns="points",values="dd",aggfunc="count")

_temp=_temp.reset_index()

_temp.name =""

_temp.columns =["id","Lost","Draw","Winning"]

Team_level_stats =_temp



Df_match_clean=pd.merge(Df_match_clean,Team_level_stats, left_on=['Team_A_id'],right_on=['id'])

Df_match_clean=Df_match_clean.drop("id",axis=1)

Df_match_clean=Df_match_clean.rename(columns={"Lost": "Team_A_tot_lost", "Draw": "Team_A_tot_Draw","Winning":"Team_A_tot_Won"})



Df_match_clean=pd.merge(Df_match_clean,Team_level_stats, left_on=['Team_B_id'],right_on=['id'])

Df_match_clean=Df_match_clean.drop("id",axis=1)

Df_match_clean=Df_match_clean.rename(columns={"Lost": "Team_B_tot_lost", "Draw": "Team_B_tot_Draw","Winning":"Team_B_tot_Won"})

Df_match_clean.head()




# Update Team last 1,2,3,4,5,10 performences 

#for i in range(Df_match_clean.shape[0]-1,0,-1):

#    print(i)



#Df_match_clean.iloc[601]



def get_last_10_math_performence(team_id,match_id):

    _temp1 =Df_match_clean[["Team_A_id","Team_A_points","match_id"]]

    _temp1.columns =["id","points","match_id"]

    _temp2 =Df_match_clean[["Team_B_id","Team_B_points","match_id"]]

    _temp2.columns =["id","points","match_id"]

    _temp1=pd.concat([_temp1,_temp2],axis=0,sort=False)

    

    _temp1=_temp1.loc[(_temp1.match_id < match_id) &(_temp1.id == team_id)].sort_values(by="match_id",ascending=False)

   

    _Last_10_matches = _temp1[:10].points.sum()

    _Last_5_matches = _temp1[:5].points.sum()

    _Last_4_matches =_temp1[:4].points.sum()

    _Last_3_matches = _temp1[:3].points.sum()

    _Last_2_matches = _temp1[:2].points.sum()

    _Last_1_matches = _temp1[:1].points.sum()

    

    _dic ={"_Last_10_matches":_Last_10_matches,"_Last_5_matches":_Last_5_matches,"_Last_4_matches":_Last_4_matches,"_Last_3_matches":_Last_3_matches,"_Last_2_matches":_Last_2_matches,

        "_Last_1_matches":_Last_1_matches}

    #print(_dic)

    

    return _dic



Df_match_clean["Team_A_last10_matches"]=0

Df_match_clean["Team_A_last5_matches"]=0

Df_match_clean["Team_A_last4_matches"]=0

Df_match_clean["Team_A_last3_matches"]=0

Df_match_clean["Team_A_last2_matches"]=0

Df_match_clean["Team_A_last1_matches"]=0





Df_match_clean["Team_B_last10_matches"]=0

Df_match_clean["Team_B_last5_matches"]=0

Df_match_clean["Team_B_last4_matches"]=0

Df_match_clean["Team_B_last3_matches"]=0

Df_match_clean["Team_B_last2_matches"]=0

Df_match_clean["Team_B_last1_matches"]=0



    

for i in range(Df_match_clean.shape[0]-1,0,-1):

    

    #update Team A Stats 

    _team_A_id = Df_match_clean.loc[i].Team_A_id

    _team_B_id = Df_match_clean.loc[i].Team_B_id

    _match_id = Df_match_clean.loc[i].match_id

    

    _dic = get_last_10_math_performence(_team_A_id,_match_id)

    Df_match_clean.loc[i,"Team_A_last10_matches"] = _dic["_Last_10_matches"]

    Df_match_clean.loc[i,"Team_A_last5_matches"] = _dic["_Last_5_matches"]

    Df_match_clean.loc[i,"Team_A_last4_matches"] = _dic["_Last_4_matches"]

    Df_match_clean.loc[i,"Team_A_last3_matches"] = _dic["_Last_3_matches"]

    Df_match_clean.loc[i,"Team_A_last2_matches"] = _dic["_Last_2_matches"]

    Df_match_clean.loc[i,"Team_A_last1_matches"] = _dic["_Last_1_matches"]

    

    

    _dic = get_last_10_math_performence(_team_B_id,_match_id)

    Df_match_clean.loc[i,"Team_B_last10_matches"] = _dic["_Last_10_matches"]

    Df_match_clean.loc[i,"Team_B_last5_matches"] = _dic["_Last_5_matches"]

    Df_match_clean.loc[i,"Team_B_last4_matches"] = _dic["_Last_4_matches"]

    Df_match_clean.loc[i,"Team_B_last3_matches"] = _dic["_Last_3_matches"]

    Df_match_clean.loc[i,"Team_B_last2_matches"] = _dic["_Last_2_matches"]

    Df_match_clean.loc[i,"Team_B_last1_matches"] = _dic["_Last_1_matches"]

    

def Get_head_to_Head(team_id,opposition,match_id):

    _temp1 =Df_match_clean[["Team_A_id","Team_A_points","match_id","Team_B_id"]]

    _temp1.columns =["id","points","match_id","opposition"]

    _temp2 =Df_match_clean[["Team_B_id","Team_B_points","match_id","Team_A_id"]]

    _temp2.columns =["id","points","match_id","opposition"]

    _temp1=pd.concat([_temp1,_temp2],axis=0,sort=False)

    

    _temp1=_temp1.loc[(_temp1.match_id < int(match_id)) &(_temp1.id == team_id) & (_temp1.opposition ==  opposition)].sort_values(by="match_id",ascending=False)

    #display(_temp1)

    _Last_5_matches = _temp1[:5].points.sum()

    _Last_3_matches = _temp1[:3].points.sum()

    _Last_2_matches = _temp1[:2].points.sum()

    _Last_1_matches =_temp1[:1].points.sum()

    

    _dic ={"_Last_5_matches":_Last_5_matches,"_Last_3_matches":_Last_3_matches,"_Last_2_matches":_Last_2_matches,"_Last_1_matches":_Last_1_matches}

    #print(_dic)

    

    return _dic



Df_match_clean["Team_A_head2Head_with_teamB_Last5_matches"]=0

Df_match_clean["Team_A_head2Head_with_teamB_Last3_matches"]=0

Df_match_clean["Team_A_head2Head_with_teamB_Last2_matches"]=0

Df_match_clean["Team_A_head2Head_with_teamB_Last1_matches"]=0



    

#Get_head_to_Head(2,3,500)



for i in range(Df_match_clean.shape[0]):

    

        #update Team A Stats 

    _team_A_id = Df_match_clean.loc[i].Team_A_id

    _team_B_id = Df_match_clean.loc[i].Team_B_id

    _match_id = Df_match_clean.loc[i].match_id

        #print(_team_A_id,_team_B_id,_match_id)



    _dic = Get_head_to_Head(_team_A_id,_team_B_id,_match_id)

    Df_match_clean.loc[i,"Team_A_head2Head_with_teamB_Last5_matches"] = int(_dic["_Last_5_matches"])

    Df_match_clean.loc[i,"Team_A_head2Head_with_teamB_Last3_matches"] = _dic["_Last_3_matches"]

    Df_match_clean.loc[i,"Team_A_head2Head_with_teamB_Last2_matches"] = _dic["_Last_2_matches"]

    Df_match_clean.loc[i,"Team_A_head2Head_with_teamB_Last1_matches"] = _dic["_Last_1_matches"]





    
#Df_match_clean.groupby(["Target","Team_A_head2Head_with_teamB_Last3_matches"])["match_id"].count()

#pd.pivot_table(Df_match_clean,index=["series_name"],columns=["Target"],values=["winner"],aggfunc="count").iplot(kind="bar",title="Seasion Wise Winners Team A vs Team B")

pd.pivot_table(data=Df_match_clean,index="Target",columns="Team_A_last10_matches",values="match_id",aggfunc="count")
Df_match_clean["Team_A_Total_matches"] =  Df_match_clean[["Team_A_tot_lost","Team_A_tot_Won","Team_A_tot_Draw"]].sum(axis=1)

Df_match_clean["Team_B_Total_matches"] =  Df_match_clean[["Team_B_tot_lost","Team_B_tot_Won","Team_B_tot_Draw"]].sum(axis=1)

Df_match_clean["Team_A_winning_prob"]=Df_match_clean.Team_A_tot_Won / Df_match_clean.Team_A_Total_matches

Df_match_clean["Team_B_winning_prob"]=Df_match_clean.Team_B_tot_Won / Df_match_clean.Team_B_Total_matches

Df_match_clean["Team_A_Odd_ratio"] =Df_match_clean["Team_A_winning_prob"] / (1- Df_match_clean["Team_A_winning_prob"])

Df_match_clean["Team_B_Odd_ratio"] =Df_match_clean["Team_B_winning_prob"] / (1- Df_match_clean["Team_B_winning_prob"])



Df_match_clean["Team_A_H2H_5MATCHES_PROB"] = Df_match_clean.Team_A_head2Head_with_teamB_Last5_matches /5

Df_match_clean["Team_A_H2H_3MATCHES_PROB"] = Df_match_clean.Team_A_head2Head_with_teamB_Last3_matches /3

Df_match_clean["Team_A_H2H_2MATCHES_PROB"] = Df_match_clean.Team_A_head2Head_with_teamB_Last3_matches /2

Df_match_clean["Team_A_H2H_1MATCHES_PROB"] = Df_match_clean.Team_A_head2Head_with_teamB_Last1_matches /1
Df_match_clean["is_Team_A_won_toss"] = [1 if i else 0 for i in Df_match_clean.toss_winner== Df_match_clean.Team_A_id]
_cols =["match_of_the_day",'Team_A_tot_lost', 'Team_A_tot_Draw', 'Team_A_tot_Won', 'Team_B_tot_lost', 'Team_B_tot_Draw', 'Team_B_tot_Won', 'Team_A_last10_matches', 'Team_A_last5_matches', 'Team_A_last4_matches', 'Team_A_last3_matches', 'Team_A_last2_matches', 'Team_A_last1_matches', 'Team_B_last10_matches', 'Team_B_last5_matches', 'Team_B_last4_matches', 'Team_B_last3_matches', 'Team_B_last2_matches', 'Team_B_last1_matches', 'Team_A_head2Head_with_teamB_Last5_matches', 'Team_A_head2Head_with_teamB_Last3_matches', 'Team_A_head2Head_with_teamB_Last2_matches', 'Team_A_head2Head_with_teamB_Last1_matches', 'Team_A_winning_prob', 'Team_A_Odd_ratio', 'Team_B_winning_prob', 'Team_B_Odd_ratio', 'Team_A_Total_matches',

       'Team_B_Total_matches', 'Team_A_H2H_5MATCHES_PROB', 'Team_A_H2H_3MATCHES_PROB', 'Team_A_H2H_2MATCHES_PROB', 'Team_A_H2H_1MATCHES_PROB', 'is_Team_A_won_toss']

X_train = Df_match_clean[_cols]

y_train = [1 if i ==1 else 0 for i in Df_match_clean.Target]



from sklearn.preprocessing import StandardScaler

scal = StandardScaler()



X_train=pd.DataFrame(scal.fit_transform(X_train))

X_train.columns = _cols

X_train.head()
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score



class EstimatorSelectionHelper:



    def __init__(self, models, params):

        

        if not set(models.keys()).issubset(set(params.keys())):

            missing_params = list(set(models.keys()) - set(params.keys()))

            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        self.models = models

        self.params = params

        self.keys = models.keys()

        self.grid_searches = {}



    def fit(self, X, y, cv=5, n_jobs=3, verbose=1, scoring=None, refit=False):

        for key in self.keys:

            print("Running GridSearchCV for %s." % key)

            model = self.models[key]

            params = self.params[key]

            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,

                              verbose=verbose, scoring=scoring, refit=refit,

                              return_train_score=True)

            gs.fit(X,y)

            self.grid_searches[key] = gs    



    def score_summary(self, sort_by='mean_score'):

        def row(key, scores, params):

            d = {

                 'estimator': key,

                 'min_score': min(scores),

                 'max_score': max(scores),

                 'mean_score': np.mean(scores),

                 'std_score': np.std(scores),

            }

            return pd.Series({**params,**d})



        rows = []

        for k in self.grid_searches:

            print(k)

            params = self.grid_searches[k].cv_results_['params']

            scores = []

            for i in range(self.grid_searches[k].cv):

                key = "split{}_test_score".format(i)

                r = self.grid_searches[k].cv_results_[key]        

                scores.append(r.reshape(len(params),1))



            all_scores = np.hstack(scores)

            for p, s in zip(params,all_scores):

                rows.append((row(k, s, p)))



        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)



        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']

        columns = columns + [c for c in df.columns if c not in columns]



        return df[columns]
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.metrics import precision_score

from sklearn.svm import NuSVC











models1 = {

    #'ExtraTreesClassifier': ExtraTreesClassifier(),

    'RandomForestClassifier': RandomForestClassifier(),

    #'AdaBoostClassifier': AdaBoostClassifier(),

    'GradientBoostingClassifier': GradientBoostingClassifier(),

    'SVC': SVC(),

    "NuSVC":NuSVC()

    

}



params1 = {

    #'ExtraTreesClassifier': { 'n_estimators': [5] },

    'RandomForestClassifier': { 'n_estimators': [32] },

    #'AdaBoostClassifier':  { 'n_estimators': [23,50,100] },

    'GradientBoostingClassifier': { 'n_estimators': [100], 'learning_rate': [0.04,0.03,0.01,0.05] },    

    "NuSVC":{'gamma':[0.1,0.5,0.9]},

    'SVC': [

        {'kernel': ['rbf'], 'C': [1,3,4,2,5], 'gamma': [0.001,0.01]},

    ]

}



helper1 = EstimatorSelectionHelper(models1, params1)

#helper1.fit(X_train, y_train,scoring='roc_auc', n_jobs=2)

helper1.fit(X_train, y_train,scoring='f1', n_jobs=2)
helper1.score_summary()
display(Df_Players.shape)

Df_Players.isna().sum(axis=0)