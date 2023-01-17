import numpy as np 
import pandas as pd 

#dataframe
df = pd.read_csv("ATP.csv")

null_percent = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': null_percent})
print(missing_value_df.reset_index().drop(columns=['index']))
df = df.drop(columns=['score','tourney_name','winner_name','tourney_date','loser_name','minutes'])
df = df[df['surface'].notna()]
df = df[df.surface != 'None']
df = df.rename(columns={"loser_age": "P1_age",\
                        "loser_entry": "P1_entry",\
                        "loser_hand": "P1_hand",\
                        "loser_ht": "P1_ht",\
                        "loser_id": "P1_id",\
                        "loser_ioc": "P1_ioc",\
                        "loser_rank": "P1_rank",\
                        "loser_rank_points": "P1_rank_points",\
                        "loser_seed": "P1_seed",\
                        'l_1stIn': 'P1_1stIn',\
                        'l_1stWon':'P1_1stWon',\
                        'l_2ndWon': 'P1_2ndWon',\
                        'l_SvGms': 'P1_SvGms',\
                        'l_ace': 'P1_ace',\
                        'l_bpFaced': 'P1_bpFaced',\
                        'l_bpSaved': 'P1_bpSaved',\
                        'l_df': 'P1_df',\
                        'l_svpt': 'P1_svpt',\
                        "winner_age": "P2_age",\
                        "winner_entry": "P2_entry",\
                        "winner_hand": "P2_hand",\
                        "winner_ht": "P2_ht",\
                        "winner_id": "P2_id",\
                        "winner_ioc": "P2_ioc",\
                        "winner_rank": "P2_rank",\
                        "winner_rank_points": "P2_rank_points",\
                        "winner_seed": "P2_seed",\
                        'w_1stIn': 'P2_1stIn',\
                        'w_1stWon':'P2_1stWon',\
                        'w_2ndWon': 'P2_2ndWon',\
                        'w_SvGms': 'P2_SvGms',\
                        'w_ace': 'P2_ace',\
                        'w_bpFaced': 'P2_bpFaced',\
                        'w_bpSaved': 'P2_bpSaved',\
                        'w_df': 'P2_df',\
                        'w_svpt': 'P2_svpt'},)
df_mirror = df.copy()
df_mirror[[ 'P1_age','P1_entry','P1_hand','P1_ht','P1_id','P1_ioc','P1_rank','P1_rank_points','P1_seed',\
            'P1_1stIn','P1_1stWon','P1_2ndWon','P1_SvGms','P1_ace','P1_bpFaced','P1_bpSaved','P1_df','P1_svpt',\
            'P2_age','P2_entry','P2_hand','P2_ht','P2_id','P2_ioc','P2_rank','P2_rank_points','P2_seed',\
            'P2_1stIn','P2_1stWon','P2_2ndWon','P2_SvGms','P2_ace','P2_bpFaced','P2_bpSaved','P2_df','P2_svpt']]\
=df_mirror[['P2_age','P2_entry','P2_hand','P2_ht','P2_id','P2_ioc','P2_rank','P2_rank_points','P2_seed',\
            'P2_1stIn','P2_1stWon','P2_2ndWon','P2_SvGms','P2_ace','P2_bpFaced','P2_bpSaved','P2_df','P2_svpt',\
            'P1_age','P1_entry','P1_hand','P1_ht','P1_id','P1_ioc','P1_rank','P1_rank_points','P1_seed',\
            'P1_1stIn','P1_1stWon','P1_2ndWon','P1_SvGms','P1_ace','P1_bpFaced','P1_bpSaved','P1_df','P1_svpt']]


#Ajout de la colonne winner player qui correspond à notre target.
winner_player2 = [1 for a in range(df.shape[0])]
df.insert(df.shape[1], "Winner_player", winner_player2, True)

winner_player1 = [-1 for a in range(df_mirror.shape[0])]
df_mirror.insert(df_mirror.shape[1], "Winner_player", winner_player1, True)

df = df.append(df_mirror)
df = df.reset_index().drop(columns=['index'])
df
df.info()
from sklearn.preprocessing import LabelEncoder
df['P2_entry'] = LabelEncoder().fit_transform(df['P2_entry'].astype(str))
df['P2_hand'] = LabelEncoder().fit_transform(df['P2_hand'].astype(str))
df['P2_ioc'] = LabelEncoder().fit_transform(df['P2_ioc'].astype(str))
df['round'] = LabelEncoder().fit_transform(df['round'].astype(str))
df['surface'] = LabelEncoder().fit_transform(df['surface'].astype(str))
df['tourney_level'] = LabelEncoder().fit_transform(df['tourney_level'].astype(str))
df['tourney_id'] = LabelEncoder().fit_transform(df['tourney_id'].astype(str))
df['P1_entry'] = LabelEncoder().fit_transform(df['P1_entry'].astype(str))
df['P1_hand'] = LabelEncoder().fit_transform(df['P1_hand'].astype(str))
df['P1_ioc'] = LabelEncoder().fit_transform(df['P1_ioc'].astype(str))
df.info()
from sklearn.impute import SimpleImputer
df2 = pd.DataFrame(SimpleImputer().fit_transform(df))
df2.columns = df.columns
df2.index = df.index
df_imputed = df2.copy()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30,30))
sns.heatmap(df_imputed.corr(), annot= True, linewidth=0.1, cmap= 'Blues')
import numpy as np
count, division = np.histogram(df_imputed)

df_imputed.hist(figsize=(30,30))
df_imputed_deleted_columns = df_imputed.copy()
df_imputed_deleted_columns = df_imputed_deleted_columns.drop(columns=['P2_1stIn','P2_1stWon','P2_2ndWon','P2_ace','P2_bpFaced','P2_bpSaved','P2_df','P2_svpt',\
            'P1_1stIn','P1_1stWon','P1_2ndWon','P1_ace','P1_bpFaced','P1_bpSaved','P1_df','P1_svpt'])

df_final = df_imputed_deleted_columns.copy()
plt.figure(figsize=(30,30))
sns.heatmap(df_final.corr(), annot= True, linewidth=0.1, cmap= 'Blues')
df_final = df_final.drop(columns=['P1_SvGms','P2_SvGms'])
from sklearn.preprocessing import LabelBinarizer

jobs_encoder = LabelBinarizer()
jobs_encoder.fit(df_final['surface'])
transformed = jobs_encoder.transform(df_final['surface'])
transform_df = pd.DataFrame(transformed)
df_final.tail()
df_final = pd.concat([df_final, transform_df], axis=1).drop(['surface'], axis=1)


df_final.sample(frac=1).reset_index(drop=True)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_final.drop(columns=["Winner_player"]), df_final["Winner_player"], test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import model_selection
XGB_model = XGBClassifier()
XGB_model.fit(x_train,y_train)
score = XGB_model.score(x_test,y_test)

print("XGBoost score: ", score)

score = XGB_model.score(x_train,y_train)

print("XGBoost score: ", score)
#évaluation en validation croisée : 10 cross-validation
succes = model_selection.cross_val_score(XGB_model,df_final.drop(columns=["Winner_player"]), df_final["Winner_player"],cv=10,scoring='accuracy')
#moyenne des taux de succès
print(succes.mean())
KNN_model = KNeighborsClassifier()
KNN_model.fit(x_train,y_train)
score = KNN_model.score(x_test,y_test)


print("KNN score: ", score)
#évaluation en validation croisée :
succes = model_selection.cross_val_score(KNN_model,df_final.drop(columns=["Winner_player"]), df_final["Winner_player"],cv=10,scoring='accuracy')
#moyenne des taux de succès
print(succes.mean())
RandomForest_model = RandomForestClassifier(n_estimators=100)
RandomForest_model.fit(x_train,y_train)
score = RandomForest_model.score(x_test,y_test)


print("RandomForest score: ", score)
#évaluation en validation croisée :
succes = model_selection.cross_val_score(RandomForest_model,df_final.drop(columns=["Winner_player"]), df_final["Winner_player"],cv=10,scoring='accuracy')
#moyenne des taux de succès
print(succes.mean())