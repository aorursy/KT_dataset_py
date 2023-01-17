# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
df = pd.read_csv('/kaggle/input/christano-ronaldo/data.csv')
df
df.info()
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df
df['team_name'].value_counts()
df['match_id'].value_counts()
#Drop irrelevant columns and columns having more than 50% missing values
dropcols = ["Unnamed: 0", "match_event_id", "team_name", "match_id", "team_id", "date_of_game","type_of_shot", "type_of_combined_shot", "remaining_min.1", "power_of_shot.1", "knockout_match.1", "remaining_sec.1", "distance_of_shot.1"  ]
df = df.drop(dropcols, axis=1)
#Removing Null values from 'home/away'
df['home/away'] = df['home/away'].fillna(method='ffill')
#Home:1; Away:0

def find(word):
    if ('vs') in word:
        return 0
    else:
        return 1
df['home/away'] = df['home/away'].apply(lambda x :find(x))
df = pd.get_dummies(df, columns=["home/away"])
df['shot_basics'].value_counts()
df['shot_basics'] = df['shot_basics'].fillna('Not Known')
def process_shot_basics(x):
    if(x in ["Mid Range", "Goal Area", "Penalty Spot", "Goal Line"]):
        return x
    return "Others"

df['shot_basics'] = df['shot_basics'].apply(lambda x : process_shot_basics(x))
plt.figure(figsize=(15,8))
sns.countplot(df['shot_basics'])
df = pd.get_dummies(df, columns=["shot_basics"])
df['area_of_shot'].value_counts()
chart = sns.countplot(df['area_of_shot'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

df = pd.get_dummies(df, columns=["area_of_shot"])
df['game_season'].value_counts()
df['game_season'] = df['game_season'].fillna(method= 'ffill')
def process_game_season(x):
    if(x in ["2015-16", "1997-98", "1998-99", "2014-15","1996-97", "2013-14"]):
        return "others"
    return x

df['game_season'] = df['game_season'].apply(lambda x : process_game_season(x))
plt.figure(figsize=(15,8))
chart = sns.countplot(df['game_season'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

df = pd.get_dummies(df, columns=["game_season"])
df['range_of_shot'].value_counts()
plt.figure(figsize=(8,5))
sns.countplot(df['range_of_shot'])
df = pd.get_dummies(df, columns=["range_of_shot"])
df['lat/lng'].value_counts()
df['lat/lng'] = df['lat/lng'].fillna(method= 'ffill')
def process_lat_lng(x):
    if(x in ['28.549237, -81.372780','39.993941, -75.143458', "42.379455, -83.115635","40.361408, -86.186052", 
             "43.717098, -79.395917", "40.708999, -73.872430", "41.484971, -81.671552" ,"38.919619, -77.015211",
             "35.492151, -97.519011", "41.845137, -87.660450", "33.768092, -84.393817", "25.790710, -80.207819" , 
             "30.018061, -90.022651" , "43.062206, -87.944754", "40.643505, -73.939507", "35.262047, -80.865746", 
             "30.028164, -89.997933", "49.250068, -123.114646", "30.055498, -89.960838" , "35.205878, -80.841194",
             "40.623199, -73.951223", "33.513157, -112.082793", "40.324211, -111.674849"]):
        return "others"
    return x

df['lat/lng'] = df['lat/lng'].apply(lambda x : process_lat_lng(x))
plt.figure(figsize=(15,8))
chart = sns.countplot(df['lat/lng'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
df = pd.get_dummies(df, columns=['lat/lng'])
df['knockout_match'].value_counts()
df['knockout_match'] = df['knockout_match'].fillna(df['knockout_match'].mode()[0])
df = pd.get_dummies(df, columns=["knockout_match"])
df['power_of_shot'].value_counts()
df['power_of_shot'] = df['power_of_shot'].fillna(method= 'ffill')
def process_power_of_shot(x):
    if(x in [5.0, 6.0, 7.0]):
        return "others"
    return x

df['power_of_shot'] = df['power_of_shot'].apply(lambda x : process_power_of_shot(x))
df = pd.get_dummies(df, columns=["power_of_shot"])
df.isnull().sum()
df['Pos-x'] = np.cos(df['location_x']) * np.cos(df['location_y'])
df['Pos-y'] = np.cos(df['location_x']) * np.sin(df['location_y'])
df['Pos-z'] = np.sin(df['location_x'])

df['Pos-x'] = df['Pos-x'].fillna('-1')
df['Pos-y'] = df['Pos-y'].fillna('-1')
df['Pos-z'] = df['Pos-z'].fillna('-1')

del df['location_x']
del df['location_y']
#Filling NaN values in "shot_id_number"
for i in range(1, (len(df['shot_id_number'])-1)):
    if (math.isnan(df['shot_id_number'][i])):
        df['shot_id_number'][i] = ((df['shot_id_number'][i-1]) + 1 )
        
df.head(10)
#Filling NaN Values

df['remaining_min'] = df['remaining_min'].fillna(df['remaining_min'].mean())                                                       
df['distance_of_shot'] = df['distance_of_shot'].fillna(df['distance_of_shot'].mean())    
df['remaining_sec'] = df['remaining_sec'].fillna(df['remaining_sec'].mean()) 


# Correlation Matrix

corr_matrix = df.corr()
corr_matrix
cols = list(df.columns.values)
cols
#Rearrange column order

df = df[['remaining_min',
 'remaining_sec',
 'distance_of_shot',
 'shot_id_number',
 'home/away_0',
 'home/away_1',
 'shot_basics_Goal Area',
 'shot_basics_Goal Line',
 'shot_basics_Mid Range',
 'shot_basics_Others',
 'shot_basics_Penalty Spot',
 'area_of_shot_Center(C)',
 'area_of_shot_Left Side Center(LC)',
 'area_of_shot_Left Side(L)',
 'area_of_shot_Mid Ground(MG)',
 'area_of_shot_Right Side Center(RC)',
 'area_of_shot_Right Side(R)',
 'game_season_1999-00',
 'game_season_2000-01',
 'game_season_2001-02',
 'game_season_2002-03',
 'game_season_2003-04',
 'game_season_2004-05',
 'game_season_2005-06',
 'game_season_2006-07',
 'game_season_2007-08',
 'game_season_2008-09',
 'game_season_2009-10',
 'game_season_2010-11',
 'game_season_2011-12',
 'game_season_2012-13',
 'game_season_others',
 'range_of_shot_16-24 ft.',
 'range_of_shot_24+ ft.',
 'range_of_shot_8-16 ft.',
 'range_of_shot_Back Court Shot',
 'range_of_shot_Less Than 8 ft.',
 'lat/lng_29.444994, -98.524120',
 'lat/lng_29.740325, -95.365762',
 'lat/lng_32.757824, -96.786653',
 'lat/lng_33.552026, -112.071667',
 'lat/lng_34.189593, -118.471724',
 'lat/lng_35.103812, -89.964007',
 'lat/lng_37.754130, -122.437947',
 'lat/lng_38.567296, -121.456638',
 'lat/lng_39.739968, -104.954013',
 'lat/lng_40.774891, -111.930790',
 'lat/lng_42.330507, -71.074655',
 'lat/lng_42.982923, -71.446094',
 'lat/lng_45.539131, -122.651648',
 'lat/lng_46.667324, -94.419250',
 'lat/lng_47.633181, -122.308343',
 'lat/lng_others',
 'knockout_match_0.0',
 'knockout_match_1.0',
 'power_of_shot_1.0',
 'power_of_shot_2.0',
 'power_of_shot_3.0',
 'power_of_shot_4.0',
 'power_of_shot_others',
 'Pos-x',
 'Pos-y',
 'Pos-z',
 'is_goal',
]]
df_input = df.dropna(subset=['is_goal'])
df_output = df[df['is_goal'].isnull()]

df_output = df_output.drop("is_goal", axis=1)

df_output
sns.boxplot(df_input["remaining_min"])
sns.boxplot(df_input["remaining_sec"])
sns.boxplot(df_input["distance_of_shot"])
#Removing Outliers

df_input = df_input[(df_input['distance_of_shot']<65) ]

df.isnull().sum()
df_output.shape
y = df_input.is_goal.values
X = df_input.drop(["is_goal"], axis = 1)
# Initialise the Scaler 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
X = scaler.fit_transform(X)
# Splitting the Data set in training and test data


from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, random_state=42)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
def score(mae):
    return(1/(1+mae))    
# Fitting Logistic regression to the training set 
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)
y_pred_proba = lr_model.predict_proba(X_test)[:,1]
mae1 = (mean_absolute_error(y_test, y_pred_proba))
score1 = score(mae1)

print("Score: {}".format(score1))
from sklearn import tree

dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_proba = dt_model.predict_proba(X_test)[:,1]
mae2 = (mean_absolute_error(y_test, y_pred_proba))
score2 = score(mae2)

print("Score: {}".format(score2))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_proba = rf_model.predict_proba(X_test)[:,1]
mae3 = (mean_absolute_error(y_test, y_pred_proba))
score3 = score(mae3)

print("Score: {}".format(score3))
from sklearn.ensemble import AdaBoostClassifier

abc_model = AdaBoostClassifier(learning_rate=1)
abc_model.fit(X_train, y_train)
y_pred_proba = abc_model.predict_proba(X_test)[:,1]
mae4 = (mean_absolute_error(y_test, y_pred_proba))
score4 = score(mae4)

print("Score: {}".format(score4))
from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier(n_estimators = 100, learning_rate=1, max_depth=1)
gbc_model.fit(X_train, y_train)
y_pred_proba = gbc_model.predict_proba(X_test)[:,1]
mae5 = (mean_absolute_error(y_test, y_pred_proba))
score5 = score(mae5)

print("Score: {}".format(score5))
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_proba = xgb_model.predict_proba(X_test)[:,1]
mae6 = (mean_absolute_error(y_test, y_pred_proba))
score6 = score(mae6)

print("Score: {}".format(score6))
methods = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost"]
score = [score1, score2, score3, score4, score5, score6 ]
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(8,5))
plt.ylabel("Score %")
plt.xlabel("Algorithms")
chart = sns.barplot(x=methods, y=score, palette=colors)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
df_output
shot_id = df_output[['shot_id_number']]
new_index = np.arange(0,6268)

shot_id['index_values'] = new_index
shot_id.set_index('index_values', inplace= True)
shot_id
df_output = scaler.transform(df_output)
#Predicting Output
y_output_prediction =  xgb_model.predict_proba(df_output)[:,1]
print(y_output_prediction)
result = pd.DataFrame(data=y_output_prediction)
result
#Column addition

result['shot_id_number']= shot_id['shot_id_number']
result
result.columns = ['is_goal', 'shot_id_number']
result.shot_id_number = result.shot_id_number.astype(int)
result = result[['shot_id_number', 'is_goal']]
result['is_goal'] = round((result['is_goal'].astype(float)),4) 
result
result.to_csv("prediction.csv", index=False)
from IPython.display import HTML

def download_link(title="Download CSV", filename='data.csv'):
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title, filename=filename)
    return HTML(html)

download_link(filename='prediction.csv')

