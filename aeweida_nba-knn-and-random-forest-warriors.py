import re
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_csv('NBA_2017_2018_Data_Without_warriors.csv')
data.columns = ['TEAM', 'DATE', 'HOMEADV', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PlusMinus']
data.WL.replace("(W)", 1, regex=True, inplace=True)
data.WL.replace("(L)", 0, regex=True, inplace=True)

data.HOMEADV.replace("(@)", 0, regex=True, inplace=True)
data.HOMEADV.replace("(vs)", 1, regex=True, inplace=True)
features_train = data[['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']].values
target_train = data['WL'].values
data = pd.read_csv('NBA_Warriors.csv')
data.columns = ['TEAM', 'DATE', 'HOMEADV', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PlusMinus']
data.WL.replace("(W)", 1, regex=True, inplace=True)
data.WL.replace("(L)", 0, regex=True, inplace=True)

data.HOMEADV.replace("(@)", 0, regex=True, inplace=True)
data.HOMEADV.replace("(vs)", 1, regex=True, inplace=True)

features_test = data[['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']].values
target_test = data['WL'].values
model = RandomForestClassifier(random_state=0)
model = RandomForestClassifier(random_state=0).fit(features_train,target_train)
predictions =  model.predict(features_test)
result=confusion_matrix(target_test, predictions)
result
Sensitivity= result[0][0]/(result[0][0]+ result[0][1])
Sensitivity
Specificity = result[1][1]/(result[1][0]+ result[1][1])
Specificity
Precision= result[0][0]/(result[0][0]+ result[1][0])
Precision
F1_Score= 2*result[0][0]/(2*result[0][0]+ result[1][0]+result[0][1])
F1_Score
r2_score(target_test, predictions, sample_weight=None)
