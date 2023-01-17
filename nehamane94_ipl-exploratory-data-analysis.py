#import packages

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns

from scipy.stats import norm

import sys

from sklearn.multiclass import OneVsRestClassifier

from pandas import DataFrame

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

#reading files

iplmatches = pd.read_csv('C:/Users/HP/Desktop/DSP/project/matches.csv')

ipldelivery = pd.read_csv('C:/Users/HP/Desktop/DSP/project/deliveries.csv')
#reading file data

iplmatches.head(5)
## BATSMEN DATA GROUPED BY MATCH

# Here the data is grouped to provide deeper depth of statistics and later for the team classificaiton



batsman_grp = ipldelivery.groupby(["match_id", "inning", "batting_team", "batsman"])

batsmen = batsman_grp["batsman_runs"].sum().reset_index()

#

# Ignore the wide balls.

balls_faced = ipldelivery[ipldelivery["wide_runs"] == 0]

balls_faced = balls_faced.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

balls_faced.columns = ["match_id", "inning", "batsman", "balls_faced"]

batsmen = batsmen.merge(balls_faced, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")



fours = ipldelivery[ ipldelivery["batsman_runs"] == 4]

sixes = ipldelivery[ ipldelivery["batsman_runs"] == 6]



fours_per_batsman = fours.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

sixes_per_batsman = sixes.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()



fours_per_batsman.columns = ["match_id", "inning", "batsman", "4s"]

sixes_per_batsman.columns = ["match_id", "inning", "batsman", "6s"]



batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen['SR'] = np.round(batsmen['batsman_runs'] / batsmen['balls_faced'] * 100, 2)



for col in ["batsman_runs", "4s", "6s", "balls_faced", "SR"]:

    batsmen[col] = batsmen[col].fillna(0)



dismissals = ipldelivery[ pd.notnull(ipldelivery["player_dismissed"])]

dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]

dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)

batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 

                        right_on=["match_id", "inning", "batsman"], how="left")



batsmen = iplmatches[['id','season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

batsmen.head(10)

ipldelivery.head(5)

#balls_faced = ipldelivery[ipldelivery["wide_runs"] == 0]

#balls_face
## Bowlers grouped by sets of data

# Data is grouped for bowlers to provide greater depth of information. Very important for the regression analysis.



bowler_grp = ipldelivery.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])

bowlers = bowler_grp["total_runs", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"].sum().reset_index()



bowlers["runs"] = bowlers["total_runs"] - (bowlers["bye_runs"] + bowlers["legbye_runs"])

bowlers["extras"] = bowlers["wide_runs"] + bowlers["noball_runs"]



del( bowlers["bye_runs"])

del( bowlers["legbye_runs"])

del( bowlers["total_runs"])



dismissal_kinds_for_bowler = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]

dismissals = ipldelivery[ipldelivery["dismissal_kind"].isin(dismissal_kinds_for_bowler)]

dismissals = dismissals.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])["dismissal_kind"].count().reset_index()

dismissals.rename(columns={"dismissal_kind": "wickets"}, inplace=True)



bowlers = bowlers.merge(dismissals, left_on=["match_id", "inning", "bowling_team", "bowler", "over"], 

                        right_on=["match_id", "inning", "bowling_team", "bowler", "over"], how="left")

bowlers["wickets"] = bowlers["wickets"].fillna(0)



bowlers_over = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler'])['over'].count().reset_index()

bowlers = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).sum().reset_index().drop('over', 1)

bowlers = bowlers_over.merge(bowlers, on=["match_id", "inning", "bowling_team", "bowler"], how = 'left')

bowlers['Econ'] = np.round(bowlers['runs'] / bowlers['over'] , 2)

bowlers = iplmatches[['id','season']].merge(bowlers, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)



bowlers.head(10)
#shows the matches count of each year

sns.countplot(x = 'season', data = iplmatches)

plt.show()
#toss winner count according to each team

sns.countplot( x = 'toss_winner', data = iplmatches)

plt.xticks(rotation='vertical')
#combined the toss_winner and ipl winner 

#true and false are total combinations of the condition

winneroft = iplmatches['toss_winner'] == iplmatches['winner']

winneroft.groupby(winneroft).size()

sns.countplot(winneroft)
#seasonwise count of wins by team who won toss and won matches

winneroftoss = iplmatches[(iplmatches['toss_winner']) == (iplmatches['winner'])]

wot = sns.countplot( x = 'winner', hue='season', data=winneroftoss)

sns.set(rc={'figure.figsize':(8,6)})

plt.xticks(rotation = 'vertical')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel("Teams")

plt.ylabel("Number of Wins")

plt.title("Number of Teams who won, given they win the toss, every season")

plt.show(wot)
#bar plot of top player of matches winners

top_players = iplmatches.player_of_match.value_counts()[:10]

#sns.barplot(x="day", y="total_bill", data=tips)

fig, ax = plt.subplots()

ax.set_ylim([0,20])

ax.set_ylabel("Number of Awards")

ax.set_xlabel("Name of Players")

ax.set_title("Top player of the match Winners")

#top_players.plot.bar()

sns.barplot(x = top_players.index, y = top_players, orient='v', palette="RdBu");

plt.xticks(rotation = 'vertical')

plt.show()
## Question regarding top bastsmen and top bowlers in history of IPL.



batsman_runsperseason = batsmen.groupby(['season', 'batting_team', 'batsman'])['batsman_runs'].sum().reset_index()

batsman_runsperseason = batsman_runsperseason.groupby(['season', 'batsman'])['batsman_runs'].sum().unstack().T

batsman_runsperseason['Total'] = batsman_runsperseason.sum(axis=1) #add total column to find batsman with the highest runs

batsman_runsperseason = batsman_runsperseason.sort_values(by = 'Total', ascending = False).drop('Total', 1)

ax.set_ylabel('Number of Runs')

ax = batsman_runsperseason[:8].T.plot()
#bowler performances

bowler_wicketsperseason = bowlers.groupby(['season', 'bowling_team', 'bowler'])['wickets'].sum().reset_index()

bowler_wicketsperseason = bowler_wicketsperseason.groupby(['season', 'bowler'])['wickets'].sum().unstack().T

bowler_wicketsperseason ['Total'] = bowler_wicketsperseason .sum(axis=1) #add total column to find bowler with the highest number of wickets

bowler_wicketsperseason  = bowler_wicketsperseason .sort_values(by = 'Total', ascending = False).drop('Total', 1)

ax = bowler_wicketsperseason [:8].T.plot()

#.ylabel('Number of Wickets')
#total runs by batsman in all the season

runs_scored=batsmen.groupby(['batsman'])['batsman_runs'].sum()

runs_scored=runs_scored.sort_values(ascending=False)

top10runs = runs_scored.head(8)

top10runs.plot('barh')
## Barplot of Runs



#sns.barplot(x="day", y="total_bill", data=tips)

fig, ax = plt.subplots()

#fig.figsize = [16,10]

#ax.set_ylim([0,20])

ax.set_xlabel("Runs")

ax.set_title("Winning by Runs - Team Performance")

#top_players.plot.bar()

sns.boxplot(y = 'winner', x = 'win_by_runs', data=iplmatches[iplmatches['win_by_runs']>0], orient = 'h'); #palette="Blues");

plt.show()
## Barplot of Wickets Win



#sns.barplot(x="day", y="total_bill", data=tips)

fig, ax = plt.subplots()

#fig.figsize = [16,10]

#ax.set_ylim([0,20])

ax.set_title("Winning by Wickets - Team Performance")

#top_players.plot.bar()

sns.boxplot(y = 'winner', x = 'win_by_wickets', data=iplmatches[iplmatches['win_by_wickets']>0], orient = 'h'); #palette="Blues");

plt.show()
# Import the new Dataset.

# Now for this dataset, I removed some features that I felt were unnecessary from the original IPL

# Dataset. However, you can use that or use the one below with a screenshot of the headings.

matches = pd.read_csv('C:/Users/HP/Desktop/DSP/project/matches1234.csv')

matches.head(3)

# Make a copy of the dataset that you imported or used before

copy_data = matches.copy()

copy_data['city'].fillna('Dubai',inplace=True)

copy_data['umpire1'].fillna('Aleem Dar',inplace=True)

# Firstly, we should have a look whether the data is completed or not.

# Because the missing value will have an adverse impact on the building of regression model.



null_values_col = copy_data.isnull().sum()

null_values_col = null_values_col[null_values_col != 0].sort_values(ascending = False).reset_index()

null_values_col.columns = ["variable", "number of missing"]

null_values_col.head()

#print(copy_data.columns)

#Create now a dataframe copy of the data and all its rows and named columns.

df = DataFrame(copy_data,columns=['team1', 'team2', 'toss_decision','toss_winner','city', 'venue', 'season', 'win_by_runs', 'win_by_wickets', 'umpire1', 'winner'])
# Now what we have done is replace the name values with numbers. Regression can only be run with 

# numbers and not anything else. 

df['winner'].fillna('Draw', inplace=True)

df.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)



encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}

df.replace(encode, inplace=True)
dicVal = encode['winner']

print(dicVal['MI']) #key value

print(list(dicVal.keys())[list(dicVal.values()).index(1)]) 

from sklearn.preprocessing import LabelEncoder

var_mod = ['toss_decision', 'city', 'venue', 'umpire1']

le = LabelEncoder()

for i in var_mod:

    df[i] = le.fit_transform(df[i])



df['winner'].astype(str).astype(int)

df.dtypes
# Now we are going to split the training and test models in a typical 60:20:20 set.

x = df[['team1', 'team2', 'toss_decision','toss_winner','city', 'venue', 'season', 'win_by_runs', 'win_by_wickets', 'umpire1']]

y = df[['winner']]



x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_model, y_model, test_size=0.2, random_state=1)
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



decision_tree_model = DecisionTreeClassifier()

y_train=y_train.astype('int')

decision_tree_model.fit(x_train,y_train)

plt.bar(range(len(x_train.columns.values)), decision_tree_model.feature_importances_)

plt.xticks(range(len(x_train.columns.values)),x_train.columns.values, rotation= 45)

plt.title('Figure 1.7 Importance of each Feature')
train_score = []

val_score = []

for depth in np.arange(1,11):

    decision_tree = tree.DecisionTreeClassifier(max_depth = depth,min_samples_leaf = 5)

    y_train=y_train.astype('int')

    decision_tree.fit(x_train, y_train)

    train_score.append(decision_tree.score(x_train,y_train))

    y_val=y_val.astype('int')

    val_score.append(decision_tree.score(x_val, y_val))



plt.plot(np.arange(1,11),train_score)

plt.plot(np.arange(1,11),val_score)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.title('Decision Tree Tuning')

plt.xlabel('Depth')

plt.ylabel('Accuracy')
train_score = []

val_score = []

for depth in np.arange(1,15):

    decision_tree = tree.DecisionTreeClassifier(max_depth = depth,min_samples_leaf = 5)

    decision_tree.fit(x_train, y_train)

    train_score.append(decision_tree.score(x_train, y_train))

    val_score.append(decision_tree.score(x_val, y_val))



plt.plot(np.arange(1,15),train_score)

plt.plot(np.arange(1,15),val_score)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.title('Decision Tree Tuning')

plt.xlabel('Depth')

plt.ylabel('Accuracy')
train_score = []

val_score = []

for leaf in np.arange(1,20):

    decision_tree = tree.DecisionTreeClassifier(max_depth = 9, min_samples_leaf = leaf)

    decision_tree.fit(x_train, y_train)

    train_score.append(decision_tree.score(x_train, y_train))

    val_score.append(decision_tree.score(x_val, y_val))



plt.plot(np.arange(1,20),train_score)

plt.plot(np.arange(1,20),val_score)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.title('Decision Tree Tuning')

plt.xlabel('Minimum Samples Leaf')

plt.ylabel('Accuracy')
my_decision_tree_model = DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 3)

my_decision_tree_model.fit(x_train, y_train)

print(my_decision_tree_model.score(x_train,y_train))

print(my_decision_tree_model.score(x_val,y_val))

y_test=y_test.astype('int')

print(my_decision_tree_model.score(x_test,y_test))
y_predict_decision = my_decision_tree_model.predict(x_test)

cm = confusion_matrix(y_test, y_predict_decision) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'], 

                     columns = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW' ])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Decision Tree \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_decision)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Curse of Dimensionality



d_train = []

d_val = []



for i in range(1,11):

    

    x_train_index = x_train.iloc[: , 0:i]

    x_val_index = x_val.iloc[: , 0:i]

    

    classifier = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 6)

    y_test=y_test.astype('int')

    dt_model = classifier.fit(x_train_index, y_train)



    d_train.append(dt_model.score(x_train_index, y_train))

    d_val.append(dt_model.score(x_val_index, y_val))

plt.title('Decision Tree Curse of Dimensionality')

plt.plot(range(1,11),d_val,label="Validation")

plt.plot(range(1,11),d_train,label="Train")

plt.xlabel('Number of Features')

plt.ylabel('Score (Accuracy)')

plt.legend()

plt.xticks(range(1,11))

plt.show()
#logistic regression

from sklearn.linear_model import LogisticRegression

import warnings

#warnings.filterwarnings("ignore", category=FutureWarning)

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

logistic_model = LogisticRegression()

y_train=y_train.astype('int')

logistic_model.fit(x_train,y_train.values.ravel())

print(logistic_model.score(x_train,y_train))

print(logistic_model.score(x_val,y_val))
train_score = []

val_score=[]



for i in np.arange(1,80):

    

    logistic_model = LogisticRegression(penalty = 'l2', C = i,random_state = 0)

    y_train=y_train.astype('int')

    logistic_model.fit(x_train,y_train.values.ravel()) 

    

    train_score.append(logistic_model.score(x_train, y_train))

    val_score.append(logistic_model.score(x_val,y_val))



    

plt.plot(np.arange(1,80),train_score)

plt.plot(np.arange(1,80),val_score)

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.title('Logistic Regression Tuning')

plt.xlabel('C')

plt.ylabel('Accuracy')
my_logistic_regression_model = LogisticRegression(penalty = 'l2', C = 48, random_state = 0)

my_logistic_regression_model.fit(x_train, y_train.values.ravel())

print(my_logistic_regression_model.score(x_train,y_train))

print(my_logistic_regression_model.score(x_val,y_val))

print(my_logistic_regression_model.score(x_test,y_test))

y_predict_logit = my_logistic_regression_model.predict(x_test)

cm = confusion_matrix(y_test, y_predict_logit) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'], 

                     columns = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW' ])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Logistic Regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_logit)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Curse of Dimensionality



d_train = []

d_val = []

for i in range(1,11):

    

    x_train_index = x_train.iloc[: , 0:i]

    x_val_index = x_val.iloc[: , 0:i]

    

    classifier = LogisticRegression(penalty = 'l2', C = 48, random_state = 0)

    y_train=y_train.astype('int')

    lr_model = classifier.fit(x_train_index, y_train.values.ravel())



    d_train.append(lr_model.score(x_train_index, y_train))

    d_val.append(lr_model.score(x_val_index, y_val))

    plt.title('Logistic Regression Curse of Dimensionality')

plt.plot(range(1,11),d_val,label="Validation")

plt.plot(range(1,11),d_train,label="Train")

plt.xlabel('Number of Features')

plt.ylabel('Score (Accuracy)')

plt.legend()

plt.xticks(range(1,11))

plt.show()
#k-nn classifier

# Model Tuning



# 5-fold cross validation



from sklearn.model_selection import KFold, cross_val_score



def rmse_cv(model):

    kf = KFold(5, shuffle=True, random_state= 42).get_n_splits(x_model.values)

    predictions = model.predict(x_test)

    rmse= np.sqrt(-cross_val_score(model, x_model.values, y_model, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_model, y_model, test_size=0.2, random_state=1)
# How to find K?



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold



train_scores = []

validation_scores = []



x_model_values = x_model.values

y_model_values = y_model.values



# 5-fold cross validation



kfold = KFold(5, shuffle=True, random_state=42)



for i in range(1,20):

    knn = KNeighborsClassifier(i)

    

    tr_scores = []

    va_scores = []

    

    for a, b in kfold.split(x_model_values):



        x_train_fold, y_train_fold = x_model_values[a], y_model_values[a]

        x_val_fold, y_val_fold = x_model_values[b], y_model_values[b]

        y_train_fold=y_train_fold.astype('int');

        knn.fit(x_train_fold, y_train_fold.ravel())

        y_val_fold=y_val_fold.astype('int');

        va_scores.append(knn.score(x_val_fold, y_val_fold))

        tr_scores.append(knn.score(x_train_fold, y_train_fold))

        

    validation_scores.append(np.mean(va_scores))

    train_scores.append(np.mean(tr_scores))
plt.title('k-NN Varying number of neighbours')

plt.plot(range(1,20),validation_scores,label="Validation")

plt.plot(range(1,20),train_scores,label="Train")

plt.legend()

plt.xticks(range(1,20))

plt.show()
# The best result is captured at k = 5 hence it is used for the final model.



#Setup a knn classifier with k neighbors



kfold = KFold(5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(5)



for m,n in kfold.split(x_model_values):

        

        x_train_fold, y_train_fold = x_model_values[m], y_model_values[m]

        y_train_fold=y_train_fold.astype('int')

        Knn = knn.fit(x_train_fold, y_train_fold.ravel())

y_test=y_test.astype('int')

print('When k=5, the testing score(accuracy) is: ')

print(Knn.score(x_test,y_test))
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



y_predict_knn = knn.predict(x_test)

cm = confusion_matrix(y_test, y_predict_knn) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'], 

                     columns = ['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW' ])

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('KNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_knn)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
print("Logistic Regression \nAccuracy:{0:.4f}".format(accuracy_score(y_test, y_predict_logit)))

print("Decision Tree \nAccuracy:{0:.4f}".format(accuracy_score(y_test, y_predict_decision)))

print("KNN Accuracy \nAccuracy:{0:.4f}".format(accuracy_score(y_test, y_predict_knn)))

# Import the dataset

Bowlers = pd.read_csv('C:/Users/HP/Desktop/DSP/project/Bowlers.csv')

# Make a copy

copy_data = Bowlers.copy()

#Ensure there are no missing values

null_values_col = copy_data.isnull().sum()

null_values_col = null_values_col[null_values_col != 0].sort_values(ascending = False).reset_index()

null_values_col.columns = ["variable", "number of missing"]

null_values_col.head()

copy_data.head(3)

#This shows us the top correlated variables with respect to one another

df = DataFrame(copy_data,columns=['over', 'wide_runs', 'noball_runs', 'runs', 'extras', 'wickets', 'Econ'])



'''

pandas.DataFrame.corr

method : {‘pearson’, ‘kendall’, ‘spearman’}

pearson : standard correlation coefficient

kendall : Kendall Tau correlation coefficient

spearman : Spearman rank correlation



min_periods : int, optional

Minimum number of observations required per pair of columns to have a valid result. Currently only available for pearson and spearman correlation

'''



corrmat = df.corr(method='pearson', min_periods=1)

r_square = corrmat ** 2



## Top 8 correlated variables

k = 9 #number of variables for heatmap

cols = r_square.nlargest(k, 'runs')['runs'].index

cm = df[cols].corr()

cm_square = cm ** 2

f, ax = plt.subplots(figsize=(10, 10))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm_square, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()



from sklearn.metrics import mean_squared_error



# RMSE for testing data



def rmse_model(model, x_test, y_test):

    predictions = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(predictions, y_test))

    return(rmse)
from sklearn.model_selection import train_test_split



x = df[['over', 'wide_runs', 'noball_runs','extras', 'wickets', 'Econ']]

y = df[['runs']]



x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_model, y_model, test_size=0.2, random_state=1)
print("the number of data for training:")

print(y_train.count())

print("the number of data for validation:")

print(y_val.count())

print("the number of data for testing:")

print(y_test.count())
#Basic Linear Regression

from sklearn.linear_model import LinearRegression



linear_regression = LinearRegression()

linear_regression.fit(x_train, y_train)



print(rmse_model(linear_regression, x_test, y_test))

print(linear_regression.coef_)

print(linear_regression.intercept_)