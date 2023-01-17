#import the library

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from termcolor import colored

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/sc2-matches-history.csv')

df.head()
#data process

X = df[['player_1', 'player_2', 'player_1_race', 'player_2_race', 'addon', 'tournament_type']]

y = df['player_1_match_status']

# Encoding the categorical data

labelencoder_X_1 = LabelEncoder()

X['player_1'] = labelencoder_X_1.fit_transform(X['player_1'])

labelencoder_X_2 = LabelEncoder()

X['player_2'] = labelencoder_X_2.fit_transform(X['player_2'])

labelencoder_X_3 = LabelEncoder()

X['player_1_race'] = labelencoder_X_3.fit_transform(X['player_1_race'])

labelencoder_X_4 = LabelEncoder()

X['player_2_race'] = labelencoder_X_4.fit_transform(X['player_2_race'])

labelencoder_X_5 = LabelEncoder()

X['addon'] = labelencoder_X_5.fit_transform(X['addon'])

labelencoder_X_6 = LabelEncoder()

X['tournament_type'] = labelencoder_X_6.fit_transform(X['tournament_type'])

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

y=pd.Series(y)

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())
#Build the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=400)
#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
# Input the data you want to predict

print("please input the folowing information:player_1_name")

player_1_name = input("Player_1_name:")

print("please input the folowing information:player_2_name")

player_2_name = input("Player_2_name:")

print("please input the folowing information:player_1_race")

player_1_race = input("Player_1_race:")

print("please input the folowing information:player_2_race")

player_2_race = input("Player_2_race:")

print("please input the folowing information:addon")

addon = input("Addon:")

print("please input the folowing information:tournament_type")

tournament_type = input("Tournament_type:")

#  Encoding categorical data

player_1_name = labelencoder_X_1.transform(np.array([[player_1_name]]))

player_2_name = labelencoder_X_2.transform(np.array([[player_2_name]]))

player_1_race = labelencoder_X_3.transform(np.array([[player_1_race]]))

player_2_race = labelencoder_X_4.transform(np.array([[player_2_race]]))

addon = labelencoder_X_5.transform(np.array([[addon]]))

tournament_type = labelencoder_X_6.transform(np.array([[tournament_type]]))
# Make prediction

new_prediction = model.predict(xgb.DMatrix([[int(player_1_name), int(player_2_name), int(player_1_race), int(player_2_race), int(addon) , int(tournament_type)]]))

if(new_prediction > 0.5):

    print(labelencoder_X_1.inverse_transform(player_1_name), " should be winner")

else:

    print(labelencoder_X_2.inverse_transform(player_2_name), " should be winner")