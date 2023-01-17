#import packages and datasets

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn import preprocessing, neighbors, svm

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC 

from sklearn import metrics



player=pd.read_csv('../input/nhl-game-data/game_skater_stats.csv')

plays=pd.read_csv('../input/nhl-game-data/game_plays.csv')

player_info=pd.read_csv('../input/nhl-game-data/player_info.csv')

skater_stats=pd.read_csv('../input/nhl-game-data/game_skater_stats.csv')

team_stats = pd.read_csv("../input/nhl-game-data/game_teams_stats.csv")

team_info = pd.read_csv("../input/nhl-game-data/team_info.csv")

teams = team_info.merge(team_stats, on='team_id')

players = player_info.merge(skater_stats, on='player_id')

game=pd.read_csv("../input/nhl-game-data/game.csv")

nullplays=plays.isna().sum()

# Percentage of null values

nullplays/len(plays)
plays.event.unique()
#Plays that are shots

shots=plays[(plays.event == 'Shot') | (plays.event == 'Missed Shot') | (plays.event == 'Blocked Shot') | (plays.event == 'Shot')]

shots.head()
#goals by shot type

Goals=plays[plays['event']=='Goal']

totalgoals=Goals.groupby('secondaryType').count()

# Descriptive Statistics of Shots

shots.describe()
# Types of Shots

shots.secondaryType.unique()
# maybe tried to add different color based on result.

snap_sample = shots.sample(n=1000, replace=False)

snapgoal_samp=Goals.sample(n=1000, replace=False)

x_coords = snap_sample['x']

y_coords = snap_sample['y']

x_coords2 = snap_sample['x']

y_coords2 = snap_sample['y']

event=shots['event']

img = plt.imread("../input/rinkpic/rink.jpg")

fig, ax = plt.subplots()

ax.imshow(img, extent=[-100, 100, -43, 43])

plt.scatter(x_coords, y_coords,s=3)

plt.title('Snap shots')

plt.show()



wrist=shots[shots['secondaryType']=='Wrist Shot']

wrist_sample = wrist.sample(n=1000, replace=False)

x_coords = wrist_sample['x']

y_coords = wrist_sample['y']

#image

img = plt.imread("../input/rinkpic/rink.jpg")

fig, ax = plt.subplots()

ax.imshow(img, extent=[-100, 100, -43, 43])

plt.scatter(x_coords, y_coords,s=5)

plt.title('Wrist Shots')

plt.show()



slap=shots[shots['secondaryType']=='Slap Shot']

slap_sample = slap.sample(n=1000, replace=False)

x_coords = slap_sample['x']

y_coords = slap_sample['y']

#image

img = plt.imread("../input/rinkpic/rink.jpg")

fig, ax = plt.subplots()

ax.imshow(img, extent=[-100, 100, -43, 43])

plt.scatter(x_coords, y_coords,s=5)

plt.title('Slap Shots')

plt.show()



Backhand=shots[shots['secondaryType']=='Backhand']

Backhand_sample = Backhand.sample(n=1000, replace=False)

x_coords = Backhand_sample['x']

y_coords = Backhand_sample['y']

#image

img = plt.imread("../input/rinkpic/rink.jpg")

fig, ax = plt.subplots()

ax.imshow(img, extent=[-95, 95, -43, 43])

plt.scatter(x_coords, y_coords,s=5)

plt.title('Backhand shots')

plt.show()



wrap=shots[shots['secondaryType']=='Wrap-around']

wrap_sample = wrap.sample(n=1000, replace=False)

x_coords = wrap_sample['x']

y_coords = wrap_sample['y']

#image

img = plt.imread("../input/rinkpic/rink.jpg")

fig, ax = plt.subplots()

ax.imshow(img, extent=[-100, 100, -43, 43])

plt.scatter(x_coords, y_coords,s=5)

plt.title('Wrap-around shots')

plt.show()
sns.jointplot(x="x", y="y", data=snap_sample,kind="kde");

sns.jointplot(x="x", y="y", data=wrist_sample,kind="kde");

sns.jointplot(x="x", y="y", data=slap_sample,kind="kde");

sns.jointplot(x="x", y="y", data=Backhand_sample,kind="kde");

sns.jointplot(x="x", y="y", data=wrap_sample,kind="kde");

plot=sns.countplot(x = 'secondaryType',

              data = Goals,

              order = Goals['secondaryType'].value_counts().index)

plot.set_xticklabels(plot.get_xticklabels(),rotation=30)
totalshots=shots.groupby('secondaryType').count()

plot2=sns.countplot(x = 'secondaryType',

              data = shots,

              order = shots['secondaryType'].value_counts().index)

plot2.set_xticklabels(plot2.get_xticklabels(),rotation=30)
#Shot Accuracy by type

shotaccuracy=totalgoals/totalshots

shotaccuracy.iloc[:,1].loc[["Deflected", "Tip-In", "Backhand","Snap Shot","Wrist Shot","Wrap-around","Slap Shot"]].plot(kind='bar',title='Shot Accuracy by Shot Type',color=['brown', 'purple', 'red', 'g','b','pink','orange'],rot=30).grid(axis='y')
# Join player and player info on player_id

player = pd.merge(player, player_info,on='player_id', how='left')

player = pd.merge(player,team_info,on='team_id',how='left')

player.head()
dist=sns.distplot(player.timeOnIce/60,color='lightblue')

dist.set_title('Time on Ice Distribuition by Minutes')
#goals per second in a game

player['GoalsPerMin']=(player.goals*60)/ player.timeOnIce

player['PointsPerMin']=(player.assists + player.goals)*60/ player.timeOnIce

player.head()
#highest goals per 15 minutes on ice (Min 5 minutes playing time).



playeractive=player[player['timeOnIce'] >= 300]

top10=playeractive.nlargest(10,'GoalsPerMin')

top10 = top10[['firstName','lastName','GoalsPerMin','goals']]

top10
# All time leader in points

player['Points']=player.goals + player.assists

positions=player.groupby('primaryPosition').Points.mean().loc[["C", "RW", "LW","D"]].plot(kind='bar',color=['r', 'g', 'b', 'orange'],title='Points by Position').grid(axis='y')

nat=player.groupby(['nationality'])['Points'].agg('sum').nlargest(6)

nat.sort_values(ascending=True).plot(kind='barh', figsize=(15, 8),color=['orange', 'g', 'darkred', 'gold','darkblue','red'], title='Points Scored by Nationality').grid(axis='x')

teamhits=player.groupby(['abbreviation'])['hits'].agg('sum').nlargest(10)

teamhits.sort_values(ascending=True).plot(kind='barh', figsize=(15, 8), title='Hits by Team').grid(axis='x')
hits=player.groupby('hits').agg('sum')

hits.head()
teams.dtypes
teams.faceOffWinPercentage = teams.faceOffWinPercentage.astype(int)

teams.won = teams.won.astype(int)

teams['TakeawayDiff']=(teams.takeaways-teams.giveaways)

teams.head()
plt.figure(figsize=(12,10))

cor = teams.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
teams.columns
X = teams[['goals','shots','hits','pim','faceOffWinPercentage','TakeawayDiff','takeaways','giveaways','powerPlayGoals','powerPlayOpportunities']]  #independent columns

y = teams[['won']]   #target column i.e price range

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(9).plot(kind='barh')

plt.show()
X = teams[['goals','shots','hits','faceOffWinPercentage','pim']]  #independent columns

y = teams[['won']]   #dependent column predicting wins



# train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

  

# training KNN  

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

  

# accuracy

score = knn.score(X_test, y_test) 

print(score)

  

# confusion matrix 

knn_predictions = knn.predict(X_test)  
# creating a confusion matrix 

cm = metrics.confusion_matrix(y_test, knn_predictions)

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

title = 'Accuracy Score: {0}'.format(score)

plt.title(title, size = 15)
X = teams[['goals','shots','hits','faceOffWinPercentage','pim']]  

y = teams[['won']]  



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)



logmodel=LogisticRegression()



logmodel.fit(X_train,y_train)



log_predictions=logmodel.predict(X_test)



score2=accuracy_score(y_test,log_predictions)

score2
cm2 = metrics.confusion_matrix(y_test, log_predictions)

sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');

plt.ylabel('Actual label');

plt.xlabel('Predicted label')

title = 'Accuracy Score: {0}'.format(score2)

plt.title(title, size = 15)
X = teams[['goals','shots','hits','faceOffWinPercentage','pim']]  #independent columns

y = teams[['won']]   #target column i.e price range

# dividing X, y into train and test data 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

  

# training a linear SVM classifier 

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 

  

# model accuracy for X_test   

score3=accuracy_score(y_test,svm_predictions)

score3
cm3 = metrics.confusion_matrix(y_test, svm_predictions)

sns.heatmap(cm3, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

title = 'Accuracy Score: {0}'.format(score3)

plt.title(title, size = 15);