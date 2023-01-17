import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

matplotlib.style.use('fivethirtyeight')



from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re

from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
shots = pd.read_csv('../input/shot_logs.csv')

player_name = 'stephen curry'

player_df = shots[shots.player_name == player_name]

print(np.true_divide(len(player_df[player_df.FGM == 1]),len(player_df)))
player_df['LOCATION'][player_df.LOCATION == 'H'] = 1

player_df['LOCATION'][player_df.LOCATION == 'A'] = 0



player_df = player_df.drop(['GAME_ID','MATCHUP','W','FINAL_MARGIN','SHOT_RESULT', 'CLOSEST_DEFENDER_PLAYER_ID','GAME_CLOCK','player_name','player_id','PTS','CLOSEST_DEFENDER'], axis = 1)

player_df = player_df[~np.isnan(player_df.SHOT_CLOCK) == True ]



X_train, X_test, y_train, y_test = train_test_split(

     player_df.drop(['FGM'], axis = 1), player_df.FGM, test_size=0.33, random_state=42)



decision_tree = tree.DecisionTreeClassifier(max_depth = 3,min_samples_leaf = 5)

decision_tree.fit(X_train, y_train)





y_pred = decision_tree.predict(X_test)



diff = y_pred - y_test

print(np.true_divide(len(diff[diff == 0]),len(y_test)))





with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
plt.bar(range(len(X_test.columns.values)), decision_tree.feature_importances_)

plt.xticks(range(len(X_test.columns.values)),X_test.columns.values, rotation= 45)

plt.title('Feature Importance')
train_score = []

test_score = []

for depth in np.arange(1,20):

    decision_tree = tree.DecisionTreeClassifier(max_depth = depth,min_samples_leaf = 5)

    decision_tree.fit(X_train, y_train)

    train_score.append(decision_tree.score(X_train,y_train))

    test_score.append(decision_tree.score(X_test,y_test))



plt.plot(np.arange(1,20),train_score)

plt.plot(np.arange(1,20),test_score)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.xlabel('Depth')

plt.ylabel('Accuracy')
player_name = 'stephen curry'

player_df = shots[shots.player_name == player_name]



player_df['LOCATION'][player_df.LOCATION == 'H'] = 1

player_df['LOCATION'][player_df.LOCATION == 'A'] = 0



player_df = player_df.drop(['GAME_ID','MATCHUP','W','FINAL_MARGIN','SHOT_RESULT', 'CLOSEST_DEFENDER_PLAYER_ID','GAME_CLOCK','player_name','player_id','PTS','CLOSEST_DEFENDER'], axis = 1)

player_df = player_df[~np.isnan(player_df.SHOT_CLOCK) == True ]



X_train, X_test, y_train, y_test = train_test_split(

     player_df.drop(['FGM'], axis = 1), player_df.FGM, test_size=0.33, random_state=42)



decision_tree = tree.DecisionTreeClassifier(max_depth = 5,min_samples_leaf = 5)

decision_tree.fit(X_train, y_train)





y_pred = decision_tree.predict(X_test)



diff = y_pred - y_test

print(np.true_divide(len(diff[diff == 0]),len(y_test)))



plt.bar(range(len(X_test.columns.values)), decision_tree.feature_importances_)

plt.xticks(range(len(X_test.columns.values)),X_test.columns.values, rotation= 45)

plt.title('Feature Importance')