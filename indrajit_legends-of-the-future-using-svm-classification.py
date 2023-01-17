import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import svm

from sklearn.svm import SVC

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/FullData.csv')

Names = pd.read_csv('../input/PlayerNames.csv')

df.assign(Index=np.nan)

df['Index'] = [v.split('/')[2] for v in Names['url']]



#weigts

a = 0.5

b = 1

c= 2

d = 3
#Attackers Index calculated using given feature space

df['at_wing'] = (c*df.Weak_foot + c*df.Ball_Control + c*df.Dribbling + c*df.Speed + d*df.Acceleration + b*df.Vision + c*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + b*df.Aggression + b*df.Agility + a*df.Curve + c*df.Long_Shots + b*df.Freekick_Accuracy + d*df.Finishing)/(a + 6*b + 6*c + 2*d)

df['at_striker'] = (b*df.Weak_foot + b*df.Ball_Control + a*df.Vision + b*df.Aggression + b*df.Agility + a*df.Curve + a*df.Long_Shots + d*df.Balance + d*df.Finishing + d*df.Heading + c*df.Jumping + c*df.Dribbling)/(3*a + 4*b + 2*c + 3*d)



X = df[['Name', 'Club_Position', 'Rating', 'Age', 'at_wing', 'at_striker']]



# Filtering out only strikers and wingers

X = X[(X['Club_Position'] == 'ST') | (X['Club_Position'] == 'RW') | (X['Club_Position'] == 'LW')]



# Normalizing the Features: We did MinMax Scaling. Same can be used using MinMaxScalers via Sklearn Module.

X['at_wing'] = (X['at_wing']-X.at_wing.min())/(X.at_wing.max() - X.at_wing.min())

X['at_striker'] = (X['at_striker']-X.at_striker.min())/(X.at_striker.max() - X.at_striker.min())



# Labeling the Legends: Score is a component of overall rating which indicates to the capability of the player

X['Score'] = ((X['at_wing']+X['at_striker'])/2)*100/X['Rating']

X['Status'] = np.where(((X['Score'] > 0.8) & (X['Club_Position'] == 'RW')), 'W', np.where(((X['Score'] > 0.8) & (X['Club_Position'] == 'ST')), 'ST', np.where(((X['Score'] > 0.8) & (X['Club_Position'] == 'LW')), 'W', 0)))



X.head()
#Making Train and Test Data

X_train = X[X['Age'] > 22]

y_train = X_train.Status

X_train = X_train[['at_wing', 'at_striker']]



test = X[X['Age'] < 23]

y_test = test.Status

X_test = test[['at_wing', 'at_striker']]



#Running SVC

clf = svm.SVC(kernel='rbf', gamma=1, C=100).fit(X_train, y_train)
result = clf.predict(X_test)

test['Status'] = result

Data = test.sort_values(['Score', 'Age'], ascending=[False, True])[:10]

sns.barplot( 'Score', 'Name', data= Data, palette = sns.color_palette("Blues_d"))
#Defending Indices

df['df_Hulk'] = (c*df.Marking + c*df.Sliding_Tackle + c*df.Standing_Tackle + c*df.Aggression + a*df.Reactions + b*df.Interceptions + c*df.Strength)/(5*c + b + a)

df['df_Aerial_Beast'] = (c*df.Marking + b*df.Sliding_Tackle + b*df.Standing_Tackle + c*df.Sliding_Tackle + c*df.Reactions + c*df.Interceptions + b*df.Balance + c*df.Jumping + b*df.Agility + d*df.Heading)/(5*c + 4*b + d)

df['df_Sweeper'] = (c*df.Ball_Control + b*df.Reactions + b*df.Interceptions + d*df.Vision + b*df.Composure + b*df.Short_Pass + b*df.Long_Pass)/(5*b + c + d)

df['df_wb_Wing_Back_Eff'] = (b*df.Ball_Control + a*df.Dribbling + a*df.Marking + c*df.Sliding_Tackle + b*df.Standing_Tackle + c*df.Attacking_Position + d*df.Vision + c*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + d*df.Acceleration +d*df.Speed + c*df.Stamina + a*df.Finishing)/(3*a + 4*b + 4*c + 3*d)



X = df[['Name', 'Club_Position', 'Rating', 'Age', 'df_Hulk', 'df_Aerial_Beast', 'df_Sweeper', 'df_wb_Wing_Back_Eff']]



# Filtering out only strikers and wingers

X = X[(X['Club_Position'] == 'CB') | (X['Club_Position'] == 'RB') | (X['Club_Position'] == 'LB') | (X['Club_Position'] == 'LCB') | (X['Club_Position'] == 'RCB') | (X['Club_Position'] == 'LWB') | (X['Club_Position'] == 'RWB')]



# Normalizing the Features: We did MinMax Scaling. Same can be used using MinMaxScalers via Sklearn Module.

X['df_Hulk'] = (X['df_Hulk']-X.df_Hulk.min())/(X.df_Hulk.max() - X.df_Hulk.min())

X['df_Aerial_Beast'] = (X['df_Aerial_Beast']-X.df_Aerial_Beast.min())/(X.df_Aerial_Beast.max() - X.df_Aerial_Beast.min())

X['df_Sweeper'] = (X['df_Sweeper']-X.df_Sweeper.min())/(X.df_Sweeper.max() - X.df_Sweeper.min())

X['df_wb_Wing_Back_Eff'] = (X['df_wb_Wing_Back_Eff']-X.df_wb_Wing_Back_Eff.min())/(X.df_wb_Wing_Back_Eff.max() - X.df_wb_Wing_Back_Eff.min())



# Labeling the Legends: Score is a component of overall rating which indicates to the capability of the player

X['Score'] = ((X['df_Hulk']+X['df_Aerial_Beast']+X['df_Sweeper']+X['df_wb_Wing_Back_Eff'])/4)*100/X['Rating']

X['Status'] = np.where(((X['Score'] > 0.8) & ((X['Club_Position'] == 'CB') | (X['Club_Position'] == 'LB') | (X['Club_Position'] == 'LCB') | (X['Club_Position'] == 'RB') | (X['Club_Position'] == 'RCB'))), 'B', np.where(((X['Score'] > 0.8) & ((X['Club_Position'] == 'LWB') | (X['Club_Position'] == 'RWB'))), 'WB', 0))



X.head()
#Making Train and Test Data

X_train = X[X['Age'] > 21]

y_train = X_train.Status

X_train = X_train[['df_Hulk', 'df_Aerial_Beast', 'df_Sweeper', 'df_wb_Wing_Back_Eff']]



test = X[X['Age'] < 22]

y_test = test.Status

X_test = test[['df_Hulk', 'df_Aerial_Beast', 'df_Sweeper', 'df_wb_Wing_Back_Eff']]



#Running SVC

clf = svm.SVC(kernel='rbf', gamma=1, C=100).fit(X_train, y_train)
result = clf.predict(X_test)

test['Status'] = result

Data = test[test['Status'] != '0'].sort_values(['Age', 'Score'], ascending=[True, False])[:10]

sns.barplot( 'Score', 'Name', data= Data, palette = sns.color_palette("Set1", n_colors=10, desat=.5))
#Midfielding Indices

df['mf_controller'] = (a*df.Weak_foot + c*df.Ball_Control + a*df.Dribbling + a*df.Marking + a*df.Reactions + d*df.Vision + c*df.Composure + d*df.Short_Pass + d*df.Long_Pass)/(2*c + 3*d + 4*a)

df['mf_beast'] = (b*df.Agility + b*df.Balance + b*df.Jumping + c*df.Strength + c*df.Stamina + b*df.Speed + a*df.Acceleration + b*df.Short_Pass + d*df.Aggression + d*df.Reactions + d*df.Marking + c*df.Standing_Tackle + c*df.Sliding_Tackle + d*df.Interceptions)/(1*a + 5*b + 4*c + 4*d)

df['mf_playmaker'] = (b*df.Ball_Control + a*df.Dribbling + a*df.Marking + b*df.Reactions + d*df.Vision + c*df.Crossing + c*df.Short_Pass + c*df.Long_Pass + a*df.Curve + a*df.Long_Shots + c*df.Freekick_Accuracy)/(4*a + 2*b + 4*c + d)

df['mf_attacker'] = (b*df.Ball_Control + c*df.Dribbling + b*df.Vision + b*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + c*df.Agility + a*df.Curve + c*df.Long_Shots + b*df.Freekick_Accuracy + d*df.Finishing)/(a + 6*b + 3*c + d)

X = df[['Name', 'Club_Position', 'Rating', 'Age', 'mf_controller', 'mf_beast', 'mf_playmaker', 'mf_attacker']]



# Filtering out only strikers and wingers

X = X[(X['Club_Position'] == 'RAM') | (X['Club_Position'] == 'RDM') | (X['Club_Position'] == 'RCM') | (X['Club_Position'] == 'LCM') | (X['Club_Position'] == 'LDM') | (X['Club_Position'] == 'LAM') | (X['Club_Position'] == 'CDM') | (X['Club_Position'] == 'CAM') | (X['Club_Position'] == 'LM') | (X['Club_Position'] == 'RM')]



# Normalizing the Features: We did MinMax Scaling. Same can be used using MinMaxScalers via Sklearn Module.

X['mf_controller'] = (X['mf_controller']-X.mf_controller.min())/(X.mf_controller.max() - X.mf_controller.min())

X['mf_beast'] = (X['mf_beast']-X.mf_beast.min())/(X.mf_beast.max() - X.mf_beast.min())

X['mf_playmaker'] = (X['mf_playmaker']-X.mf_playmaker.min())/(X.mf_playmaker.max() - X.mf_playmaker.min())

X['mf_attacker'] = (X['mf_attacker']-X.mf_attacker.min())/(X.mf_attacker.max() - X.mf_attacker.min())



# Labeling the Legends: Score is a component of overall rating which indicates to the capability of the player

X['Score'] = ((X['mf_controller']+X['mf_beast']+X['mf_playmaker']+X['mf_attacker'])/4)*100/X['Rating']

X['Status'] = np.where((X['Score'] > 0.8) & (X['Club_Position'] == 'RAM') | (X['Club_Position'] == 'RDM') | (X['Club_Position'] == 'RCM') | (X['Club_Position'] == 'LCM') | (X['Club_Position'] == 'LDM') | (X['Club_Position'] == 'LAM') | (X['Club_Position'] == 'CDM') | (X['Club_Position'] == 'CAM') | (X['Club_Position'] == 'LM') | (X['Club_Position'] == 'RM'), 'M', 0)



X.head()
#Making Train and Test Data

X_train = X[X['Age'] > 21]

y_train = X_train.Status

X_train = X_train[['mf_controller', 'mf_beast', 'mf_playmaker', 'mf_attacker']]



test = X[X['Age'] < 22]

y_test = test.Status

X_test = test[['mf_controller', 'mf_beast', 'mf_playmaker', 'mf_attacker']]



#Running SVC

clf = svm.SVC(kernel='rbf', gamma=1, C=100).fit(X_train, y_train)
result = clf.predict(X_test)

test['Status'] = result

Data = test[test['Status'] != '0'].sort_values(['Age', 'Score'], ascending=[True, False])[:10]

sns.barplot( 'Score', 'Name', data= Data, palette = sns.color_palette("Set1", n_colors=10, desat=.5))