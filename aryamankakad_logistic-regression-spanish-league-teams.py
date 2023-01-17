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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")

# Extract the player's name and some of the continous variables.
df = df[['club', 'overall', 'potential', 'value_eur', 'wage_eur', 'international_reputation', 'weak_foot',
       'skill_moves', 'release_clause_eur', 'pace', 'shooting',
       'passing', 'dribbling', 'defending', 'physic', 'gk_diving',
       'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
       'gk_positioning', 'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle', 'goalkeeping_diving',
       'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes']]

# Replacing null values with mean
df = df.fillna(df.mean())
# Selecting the Top 4, Mid 4, and Bottom 4 clubs

df_clubs = df[(df.club =='FC Barcelona') | (df.club =='Atlético Madrid') | (df.club =='Real Madrid') | (df.club == 'Valencia CF') |
              (df.club =='Real Sociedad') | (df.club =='Real Betis') | (df.club =='Deportivo Alavés') | (df.club == 'SD Eibar') |
              (df.club == 'RC Celta') |(df.club == 'Girona FC') | (df.club == 'SD Huesca') | (df.club == 'Rayo Vallecano')]

# Categorizing the Top 4, Mid 4, Bottom 4 clubs

df_clubs.club = df_clubs.club.replace({'FC Barcelona':'top 4', 'Atlético Madrid':'top 4', 'Real Madrid':'top 4', 'Valencia CF':'top 4',
                                       'Real Sociedad':'mid 4', 'Real Betis':'mid 4', 'Deportivo Alavés':'mid 4', 'SD Eibar':'mid 4',
                                       'RC Celta':'bottom 4', 'Girona FC':'bottom 4', 'SD Huesca':'bottom 4', 'Rayo Vallecano':'bottom 4'})

df_clubs.club.value_counts()
# X - all features except the club name , y - Club's name
X = df_clubs.iloc[:, 1:].values
y = df_clubs.club.values

# 80/20 train & test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)

# Standardizing the Data on the training set only
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scaling both the training and the testing set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

# Intializing the Logistic Regression Model
lr = LogisticRegression(random_state = 1, max_iter = 500)

# Train the Model
lr.fit(X_train, y_train)

# Getting the prediction
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print("-----------Confusion matrix-----------")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print("--------------------------------------")

print("Test set accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("=======================================")

# 10 fold CV
acc = cross_val_score(lr, X_test, y_test, cv=10)
print("\n10-fold CV accuracy for each fold\n {}".format(acc))
print("\n--------------------------------------")
print("10-fold CV Average Accuracy: {:.2f}".format(acc.mean()))
df = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")

# Extracting player's name and few of the continous variables
df = df[['club', 'overall', 'potential', 'value_eur', 'wage_eur', 'international_reputation']]

# Replacing null values with the mean
df = df.fillna(df.mean())
# Selecting the Top 4, Mid 4, and Bottom 4 clubs

df_clubs = df[(df.club =='FC Barcelona') | (df.club =='Atlético Madrid') | (df.club =='Real Madrid') | (df.club == 'Valencia CF') |
              (df.club =='Real Sociedad') | (df.club =='Real Betis') | (df.club =='Deportivo Alavés') | (df.club == 'SD Eibar') |
              (df.club == 'RC Celta') |(df.club == 'Girona FC') | (df.club == 'SD Huesca') | (df.club == 'Rayo Vallecano')]

# Categorizing the Top 4, Mid 4, Bottom 4 clubs

df_clubs.club = df_clubs.club.replace({'FC Barcelona':'top 4', 'Atlético Madrid':'top 4', 'Real Madrid':'top 4', 'Valencia CF':'top 4',
                                       'Real Sociedad':'mid 4', 'Real Betis':'mid 4', 'Deportivo Alavés':'mid 4', 'SD Eibar':'mid 4',
                                       'RC Celta':'bottom 4', 'Girona FC':'bottom 4', 'SD Huesca':'bottom 4', 'Rayo Vallecano':'bottom 4'})

df_clubs.club.value_counts()
# X - All features except the club's name , y - Club's name
X = df_clubs.iloc[:, 1:].values
y = df_clubs.club.values

# Training and Testing split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)

# Standardizing using the training set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scaling both the training and testing set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

# Intializing the Logisitic Regression Model
lr = LogisticRegression(random_state = 1, max_iter = 500)

# Training the Model
lr.fit(X_train, y_train)

# Get Predicition
y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print("-----------Confusion matrix-----------")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print("--------------------------------------")

print("\nTest set accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("=======================================")

# 10 fold CV
acc = cross_val_score(lr, X_test, y_test, cv=10)
print("\n10-fold CV accuracy for each fold\n {}".format(acc))
print("\n--------------------------------------")
print("10-fold CV Average Accuracy: {:.2f}".format(acc.mean()))