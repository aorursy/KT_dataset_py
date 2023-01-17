import gc
import os
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
##############################################
%matplotlib inline
DATA_PATH = "/kaggle/input/innovationcup/"
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))
ID_Data_train = pd.read_csv(DATA_PATH+"ID_Data_train.csv")
ID_Data_test = pd.read_csv(DATA_PATH+"ID_Data_test.csv")
ID_Time_train = pd.read_csv(DATA_PATH+"ID_Time_train.csv")
sample_submission_kaggle = pd.read_csv(DATA_PATH+"sample_sub_kaggle.csv")
sample_submission_coachs = pd.read_csv(DATA_PATH+"sample_sub_coachs.csv")
print(ID_Data_train.shape)
print(ID_Data_train.columns)
print(ID_Data_train['id'].nunique())
ID_Data_train.sample(3)


print(ID_Time_train.shape)
print(ID_Time_train.columns)
print(ID_Time_train['id'].nunique())
ID_Time_train.sample(3)
def calc_diff_angle(data):
    data.loc[:,'Diff_angle'] = data.loc[:,'direction_vent'] - data.loc[:,'cap']
    return data
# ID_Time_trainV1.head()
def creer_features(input_data, time_id):
    """
    input_data : DF comprenant les courses des bateaux (séries temporelles des variables considérées)
    time_id : DF lié à input_data qui comprend l'ID, le temps, la course, et le rang 
    """
    X_model = pd.DataFrame()
    IDs = time_id['id'].values
    data = input_data.copy()
    for i in tqdm_notebook(IDs): 
        data_id = data[data['id']==i]
        data_id = calc_diff_angle(data_id)
        X_model.loc[i, 'lat_mean'] = data_id['latitude'].mean()
        X_model.loc[i, 'long_std'] = data_id['longitude'].std()
        
       
    X_model = X_model.fillna(0)
    return X_model
X_train = creer_features(ID_Data_train, ID_Time_train)
y_train = X_train.merge(ID_Time_train, left_index=True, right_on='id', how='left')['temps']
print(len(y_train), X_train.shape)

X_test = creer_features(ID_Data_test, sample_submission_kaggle)
print(X_test.shape)
def ecart_classement(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def calc_rank_from_time(data_pred, id_time_train): 
    """ 
    Calcul le rang à partir d'un fichier de prédiction de temps 
    Il faut que la prédiction soit un dataframe avec en indice les id bateau hash, en colonne le temps prédit
    et une autre colonne avec la course.
    Cela permet de reconstruire le classement à partir des données prédites et des données présentes 
    dans le jeu d'entrainement
    """
    IDs = data_pred['id'].tolist()
    id_time_all = pd.concat([data_pred, id_time_train], axis=0)
    id_time_all['rang'] = id_time_all.groupby('id_race')['temps'].rank(ascending=True)
    data_pred_rank = id_time_all[id_time_all['id'].isin(IDs)]
    return data_pred_rank 

def score_innovation_cup(sub_true, sub_pred):
    """
    Il faut fournir une soumission sous format dataframe avec en index les id_bateau_hash, 
    une première colonne Time, et une seconde colonne rang, calculée par l'étudiant
    Il est conseillé d'inclure la course en colonne également mais ce n'est pas obligatoire
    """
    true_time, true_rank = sub_true['temps'], sub_true['rang']
    pred_time, pred_rank = sub_pred['temps'], sub_pred['rang']
    
    score = ecart_classement(true_rank, pred_rank) + (rmse(true_time, pred_time) / np.sqrt(np.mean(true_time)))
    return score


LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
submission_final_coachs = sample_submission_coachs.copy()
submission_final_kaggle= sample_submission_kaggle.copy()
submission_final_kaggle['temps'] = y_pred
submission_final_coachs['temps'] = y_pred
submission_final_coachs = calc_rank_from_time(submission_final_coachs, ID_Time_train)
submission_final_coachs.head(7)
submission_final_coachs.to_csv("soumission_finale_TEAMNAME#3.csv", index=False)
submission_final_kaggle.to_csv("soumission_finale_TEAMNAME.csv", index=False)