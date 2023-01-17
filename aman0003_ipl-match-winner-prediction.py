# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder()
# Input data files are available in the "../input/" directory.1

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/IPL Match Data.csv")
df.head()
df[df["result"] == 'tie']
df.describe()
df.shape
df.dtypes
df.isna().sum()
for col in df.columns:
    print(col, "Length : ",  len(pd.Series(df[col]).value_counts()) )
    print(col, "\n",pd.Series(df[col]).value_counts())
df.drop(["umpire3"], axis = 1, inplace = True)

df['city'].dropna(inplace = True)
df['city'].isna().sum()
df.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14},
          'city': {'Hyderabad': 1, 'Pune': 2, 'Rajkot': 3, 'Indore': 4, 'Bangalore': 5, 'Mumbai': 6,
       'Kolkata': 7, 'Delhi': 8, 'Chandigarh': 9, 'Kanpur': 10, 'Jaipur': 11, 'Chennai': 12,
       'Cape Town': 13, 'Port Elizabeth': 14, 'Durban': 15, 'Centurion': 16,
       'East London': 17, 'Johannesburg': 18, 'Kimberley': 19, 'Bloemfontein': 20,
       'Ahmedabad': 21, 'Cuttack': 22, 'Nagpur': 23, 'Dharamsala': 24, 'Kochi': 25,
       'Visakhapatnam': 26, 'Raipur': 27, 'Ranchi': 28, 'Abu Dhabi': 29, 'Sharjah': 30}}
df.replace(encode, inplace=True)
df.head()
from sklearn import preprocessing 
lbe = preprocessing.LabelEncoder()
lbe.fit(df['toss_decision'])
df['toss_decision'] = lbe.transform(df['toss_decision'])

lbe.fit(df['result'])
df['result'] = lbe.transform(df['result'])

lbe.fit(df['venue'])
df['venue'] = lbe.transform(df['venue'])

df.head()
df = df[df["winner"].isna() == False]
df.head()
list(df.columns.values)
df.isna().sum()
df.dropna(inplace = True)
features = df.drop(['id','date'
 ,'season', 'winner', 'result',
 'dl_applied',
 'win_by_runs',
 'win_by_wickets',
 'player_of_match',
 'umpire1',
 'umpire2'], axis = 1)

target = df['winner']
features.columns
features['city'].isna().sum()
target = target.astype(int)
features.dtypes
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score as score

x_train, x_test, y_train, y_test = tts(features, target, random_state= 0)
from sklearn.model_selection import GridSearchCV

def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param = { 'criterion' : ['gini', 'entropy'], 'max_depth': range(1,20)}
    #use gridsearch to test all values
    gscv = GridSearchCV(DecisionTreeClassifier(), param, cv=nfolds)
    #fit model to data
    gscv.fit(X, y)
    return gscv.best_params_
dtree_grid_search(x_train, y_train, 5)
clf_d = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,random_state=0)
clf_d.fit(x_train, y_train)
predictions = clf_d.predict(x_test)
print("Test:" , clf_d.score(x_test, y_test))
print("Train:",  clf_d.score(x_train, y_train))
from sklearn.metrics import confusion_matrix
result = confusion_matrix(y_test, predictions)
result
from sklearn.metrics import classification_report
report = classification_report(digits = 3, y_true = y_test, y_pred = predictions)
print(report)
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus

#dot_data = StringIO() 
dot_data = tree.export_graphviz(clf_d, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 

Image(graph.create_png())  # must access graph's first element

from sklearn.ensemble import RandomForestClassifier
clf_r = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf= 2, random_state = 0)
clf_r.fit(x_train, y_train)
predictions_r = clf_r.predict(x_test)
print("Test:" , clf_r.score(x_test, y_test))
print("Train:",  clf_r.score(x_train, y_train))
