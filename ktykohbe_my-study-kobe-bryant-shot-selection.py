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
#STEP2-1 data read

########################################################

import numpy as np

import pandas as pd

import datetime

from sklearn.preprocessing import LabelEncoder



BE_data = pd.read_csv('../input/kobe-bryant-shot-selection/data.csv.zip')

BE_sample = pd.read_csv('../input/kobe-bryant-shot-selection/sample_submission.csv.zip')
#STEP2-2 Drop unimportant data

########################################################

#Wk4 (Wk 19.Oct.2020) 'matchup'残す

delete_columns = ['shot_distance',

                 'game_date']



BE_data.drop(delete_columns, axis=1, inplace=True)



#Obfect to Label by LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

BE_data["action_type"] = le.fit_transform(BE_data["action_type"])

BE_data["season"] = le.fit_transform(BE_data["season"])

BE_data["combined_shot_type"] = le.fit_transform(BE_data["combined_shot_type"])

BE_data["shot_type"] = le.fit_transform(BE_data["shot_type"])

BE_data["shot_zone_area"] = le.fit_transform(BE_data["shot_zone_area"])

BE_data["shot_zone_basic"] = le.fit_transform(BE_data["shot_zone_basic"])

BE_data["shot_zone_range"] = le.fit_transform(BE_data["shot_zone_range"])

BE_data["team_name"] = le.fit_transform(BE_data["team_name"])

BE_data["opponent"] = le.fit_transform(BE_data["opponent"])
#STEP2-3 Focus on Distance Between LAL Arena and Opponent Arena

########################################################

BE_data["matchup"]=BE_data["matchup"].map({ "LAL @ ATL": 3111.71,  "LAL @ BKN" : 3940.98, "LAL @ BOS" : 4171.64,

                                            "LAL @ CHA" : 3406.51, "LAL @ CHH" : 3406.51, "LAL @ CHI" : 2802.76,

                                            "LAL @ CLE" : 3293.8,  "LAL @ DAL" : 1992.7,  "LAL @ DEN" : 1336.99,

                                            "LAL @ DET" : 3187.13, "LAL @ GSW" : 556.05,  "LAL @ HOU" : 2209.31,

                                            "LAL @ IND" : 2908.74, "LAL @ LAC" : 0,       "LAL @ MEM" : 2577.17,

                                            "LAL @ MIA" : 3760.32, "LAL @ MIL" : 2804.07, "LAL @ MIN" : 2450.08,

                                            "LAL @ NJN" : 3941,    "LAL @ NOH" : 2687.88, "LAL @ NOP" : 2687.88,

                                            "LAL @ NYK" : 3938.9,  "LAL @ OKC" : 1898.97, "LAL @ ORL" : 3538.33,

                                            "LAL @ PHI" : 3845.69, "LAL @ PHX" : 576.62,  "LAL @ POR" : 1331.11,

                                            "LAL @ SAC" : 581.69,  "LAL @ SAS" : 1940.71, "LAL @ SEA" : 1898.99,

                                            "LAL @ TOR" : 3497.01, "LAL @ UTA" : 935.07,  "LAL @ VAN" : 3695.84,

                                            "LAL @ WAS" : 3695.84

                                          })

BE_test = BE_data

BE_train = BE_data



#Step3

########################################################

BE_test = BE_test[BE_test["shot_made_flag"].isnull()]

BE_train.dropna(subset=['shot_made_flag'], inplace=True)



X_train = BE_train.drop('shot_made_flag', axis=1)
y_train = BE_train['shot_made_flag']

X_test = BE_test.drop('shot_made_flag', axis=1)

Y_train = BE_train['shot_made_flag']
from xgboost import XGBClassifier



model = XGBClassifier(n_estimators=20, random_state=71)

model.fit(X_train, Y_train)

pred = model.predict_proba(X_test)[:, 1]





sub_data = pd.DataFrame({'shot_id':X_test['shot_id'], 'shot_made_flag':pred})
#Step4

########################################################

# Submit File

submission = sub_data

submission.to_csv('submission_first.csv', index=False)