# Import packages

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

%matplotlib inline



from pandas import DataFrame

from subprocess import check_output

from ast import literal_eval

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
# Data info

df_columns = pd.read_csv('../input/leagueoflegends/_columns.csv',sep=',')

df_original = pd.read_csv('../input/leagueoflegends/LeagueofLegends.csv',sep=',')



df_original[['bResult','goldblue','bKills','bTowers', 'bInhibs', 'bDragons', 'bBarons', 'bHeralds']].head(3)
#Look the information of the dataframe

df_original.info()

df = df_original.copy(deep=True)
# Transform all the columns containing pseudo lists to real lists



df['goldblue'] = df['goldblue'].apply(literal_eval)

df['bKills'] = df['bKills'].apply(literal_eval)

df['bTowers'] = df['bTowers'].apply(literal_eval)

df['bInhibs'] = df['bInhibs'].apply(literal_eval)

df['bDragons'] = df['bDragons'].apply(literal_eval)

df['bBarons'] = df['bBarons'].apply(literal_eval)

df['bHeralds'] = df['bHeralds'].apply(literal_eval)



df['goldred'] = df['goldred'].apply(literal_eval)

df['rKills'] = df['rKills'].apply(literal_eval)

df['rTowers'] = df['rTowers'].apply(literal_eval)

df['rInhibs'] = df['rInhibs'].apply(literal_eval)

df['rDragons'] = df['rDragons'].apply(literal_eval)

df['rBarons'] = df['rBarons'].apply(literal_eval)

df['rHeralds'] = df['rHeralds'].apply(literal_eval)
# Capturing only the information that interests us from the data lists



data = pd.DataFrame()



data['blue_tag'] = df['blueTeamTag']

data['blue_result'] = df['bResult']

data['blue_end_gold'] = df['goldblue'].apply(max)

data['blue_kills'] = df['bKills'].apply(len)

data['blue_towers'] = df['bTowers'].apply(len)

data['blue_inhibs'] = df['bInhibs'].apply(len)

data['blue_dragons'] = df['bDragons'].apply(len)

data['blue_barons'] = df['bBarons'].apply(len)

data['blue_heralds'] = df['bHeralds'].apply(len)



data['red_tag'] = df['redTeamTag']

data['red_result'] = df['rResult']

data['red_end_gold'] = df['goldred'].apply(max)

data['red_kills'] = df['rKills'].apply(len)

data['red_towers'] = df['rTowers'].apply(len)

data['red_inhibs'] = df['rInhibs'].apply(len)

data['red_dragons'] = df['rDragons'].apply(len)

data['red_barons'] = df['rBarons'].apply(len)

data['red_heralds'] = df['rHeralds'].apply(len)



data = data[(data['blue_tag'] == 'C9') | (data['red_tag'] == 'C9')]

data = data.reset_index(drop=True)



data['winner'] = np.where(data['blue_result'] == 1, 1, 2)





data
fig = plt.figure(figsize=(12,12))



sns.set_style('darkgrid')

sns.heatmap(data[['blue_end_gold','blue_kills', 'blue_towers', 'blue_inhibs', 'blue_dragons', 'blue_barons', 'blue_heralds',

                  'red_end_gold','red_kills','red_towers','red_inhibs', 'red_dragons', 'red_barons', 'red_heralds', 'winner']].corr(), annot=True, square=True, cmap='coolwarm')
X = data[['blue_end_gold','blue_kills', 'blue_towers', 'blue_inhibs', 'blue_dragons', 'blue_barons', 'blue_heralds',

          'red_end_gold','red_kills','red_towers','red_inhibs', 'red_dragons', 'red_barons', 'red_heralds']]

y = data['winner']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report



logmodel = LogisticRegression(max_iter=1000)

logmodel.fit(X_train, y_train)



predictions = logmodel.predict(X_test)
cr = classification_report(y_test, predictions)

print('Classification Report : \n', cr)



acc = round(logmodel.score(X_test, y_test) * 100, 2)

print("Accuracy of Logistic Regression: " + str(acc) + "%")



cm = confusion_matrix(y_test,predictions)

sns.heatmap(cm, annot=True, fmt="d", xticklabels=['2 win', '1 win'], yticklabels=['2 win', '1 win'],);
# Predict the TSM x C9 match

x1 = [[62729, 16, 9, 2, 1, 0, 0,

       56672, 9, 4, 0, 3, 1, 0]]



pred = logmodel.predict_proba(x1).reshape(-1,1)



win = round(logmodel.predict(x1)[0], 2)

print("Winner is :", win)



fir_prob = round(pred[0][0] * 100, 2)

sec_prob = round(pred[1][0] * 100, 2) 

print("First team (blue) win probability is: " + str(fir_prob) + "%")

print("Second team (red) win probability is: " + str(sec_prob) + "%")