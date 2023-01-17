%matplotlib inline

# Essentials: Data Cleansing and ETL
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerLine2D

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc # good for evaluation of binary classification problems
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/NFL Play by Play 2009-2017 (v4).csv')
df.head(10)
df.info()
print("Rows: ",len(df))
# take the dataframe for field goals above and shorten the scope of the columns to just FG information
plays = ['Date','GameID','qtr','time','yrdline100','PlayType','FieldGoalResult','FieldGoalDistance','posteam','DefensiveTeam','PosTeamScore','DefTeamScore','Season']
plays = df[plays]

# Filter out any random results of NA for the Play Type
plays = plays[plays.PlayType != 'NA']
# take the dataframe for field goals above and shorten the scope of the columns to just FG information
play_attr = ['GameID','qtr','TimeSecs','yrdline100','ydstogo','Drive','down','PlayType','PassAttempt','RushAttempt','Yards.Gained','posteam','DefensiveTeam','PosTeamScore','DefTeamScore','Season']
plays = df[play_attr]

plays = plays[plays.PlayType.notna() & (plays.PlayType != 'No Play') & (plays.PlayType != 'Kickoff') & (plays.PlayType != 'Extra Point')]
plays=plays.rename(columns = {'posteam':'Team'})
plays.head(5)
# group by qtr: count
regulation_plays = plays[plays.qtr != 5]
ax = regulation_plays.groupby(['qtr'])['PassAttempt','RushAttempt'].sum().plot.bar(figsize=(20,9),color=['saddlebrown','orange'],rot=0,fontsize=16)
ax.set_title("Amount of Plays each Quarter - by Rush or Pass", fontsize=24)
ax.set_xlabel("Quarter", fontsize=18)
ax.set_ylabel("# of Plays", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.04, i.get_height()-1500, str(round((i.get_height()), 2)), fontsize=16, color='black',rotation=0)

# group by qtr: count
plays_down = plays[plays.down <= 3]
ax = plays_down.groupby(['down'])['PassAttempt','RushAttempt'].sum().plot.bar(figsize=(20,9),color=['saddlebrown','orange'],rot=0,fontsize=16)
ax.set_title("Play Calling by Down - by Rush or Pass", fontsize=24)
ax.set_xlabel("Down", fontsize=18)
ax.set_ylabel("# of Plays", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.06, i.get_height()-2400, str(round((i.get_height()), 2)), fontsize=16, color='black',rotation=0)
plays_down = plays[(plays.down <= 3) & (plays.qtr < 5) & (plays.Team == 'CLE')]
ax = plays_down.groupby(['Season','GameID'])['PassAttempt','RushAttempt'].sum().plot.line(color=['saddlebrown','orange'],figsize=(20,9),rot=0,fontsize=16)
ax.set_title("Season Play Calling - Cleveland Browns", fontsize=24)
ax.set_ylabel("# Plays", fontsize=14)
ax.set_alpha(0.8)
# Get average results for offensive plays by game for model
# to preserve the dataframe's shape (with GameID being unique), I'm going to use a split-apply-merge strategy

# Split - from origional DF: Get 2 DF's for plays that are labeled Run or Pass
r_off_agg = df[(df.PlayType == 'Run')]
p_off_agg = df[(df.PlayType == 'Pass')|(df.PlayType == 'Sack')]

# Apply - groupby aggregation to find the Median yards by game, team, PlayType, and qtr
r_off_agg = r_off_agg.groupby(['GameID','qtr','posteam'])['Yards.Gained'].mean().reset_index()
p_off_agg = p_off_agg.groupby(['GameID','qtr','posteam'])['Yards.Gained'].mean().reset_index()

r_off_agg = r_off_agg.rename(columns={'Yards.Gained':'RushingMean'}) # Rename the columns for clarity
p_off_agg = p_off_agg.rename(columns={'Yards.Gained':'PassingMean'})

# Merge - Combine the Away and Home averages into one dataframe
off_agg = pd.merge(r_off_agg,
                 p_off_agg,
                 left_on=['GameID','qtr','posteam'],
                 right_on=['GameID','qtr','posteam'],
                 how='outer')

off_agg.head(8)
off_tendencies = df[df.PlayType.notna()&
              (df.PlayType != 'No Play')&
              (df.PlayType != 'Kickoff')&
              (df.PlayType != 'Extra Point')&
              (df.PlayType != 'End of Game')&
              (df.PlayType != 'Quarter End')&
              (df.PlayType != 'Half End')&
              (df.PlayType != 'Two Minute Warning')&
              (df.PlayType != 'Field Goal')&
              (df.PlayType != 'Punt') &
              (df.PlayAttempted == 1)]

# Moving average by team, quarter, and season. This is a rolling average to consider recent decisions to compensate for coaching changes
off_tendencies = off_tendencies.groupby(['GameID','posteam','Season','qtr'])['PassAttempt','RushAttempt'].sum().reset_index()
off_tendencies['PassingWA']=off_tendencies.groupby(['posteam','qtr','Season']).PassAttempt.apply(lambda x: x.shift().rolling(8,min_periods=1).mean().fillna(x))
off_tendencies['RushingWA']=off_tendencies.groupby(['posteam','qtr','Season']).RushAttempt.apply(lambda x: x.shift().rolling(8,min_periods=1).mean().fillna(x))
off_tendencies = off_tendencies.drop(columns=['PassAttempt', 'RushAttempt'])
off_tendencies[(off_tendencies.posteam == 'CLE')&(off_tendencies.qtr == 1)].head(20)
# to limit the data size, lets look at one team to begin
team = 'CLE'
# take the dataframe for plays above and define particular columns we want
play_attr = ['PlayAttempted','GameID','qtr','TimeSecs','yrdline100','ydstogo','Drive','down','PlayType','GoalToGo',
             'TimeUnder','PlayTimeDiff','PassAttempt','RushAttempt','posteam','DefensiveTeam','PosTeamScore',
             'DefTeamScore','Season','HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre','No_Score_Prob',
             'Opp_Field_Goal_Prob','Opp_Safety_Prob','Win_Prob','HomeTeam','ExpPts']
plays = df[play_attr]


# filter out the records that we wont use to predict run or pass
plays = plays[plays.PlayType.notna()&
              (plays.PlayType != 'No Play')&
              (plays.PlayType != 'Kickoff')&
              (plays.PlayType != 'Extra Point')&
              (plays.PlayType != 'End of Game')&
              (plays.PlayType != 'Quarter End')&
              (plays.PlayType != 'Half End')&
              (plays.PlayType != 'Two Minute Warning')&
              (plays.PlayType != 'Field Goal')&
              (plays.PlayType != 'Punt')]

# assure that there was a play attempted to filter out penalties before the play occured.
plays = plays[plays.PlayAttempted == 1]

# add data regarding offensive stats
plays = pd.merge(plays,
                off_agg,
                left_on=['GameID','qtr','posteam'],
                right_on=['GameID','qtr','posteam'],
                how='left')

# merge data for moving average play calling tendencies
plays = pd.merge(plays,
                off_tendencies,
                left_on=['GameID','qtr','posteam','Season'],
                right_on=['GameID','qtr','posteam','Season'],
                how='left')

plays=plays.rename(columns = {'posteam':'Team'})

# filter on just possessions by the cleveland browns (woof woof)
plays = plays[(plays['Team'] == team)]
plays.head(5)
# get score difference for each play cleveland is in possession of the ball
plays['ScoreDiff'] = plays['PosTeamScore'] - plays['DefTeamScore']

# add column to show boolean indicator for whether the Browns are winning or losing (I expect a lot of 0's)
plays['CurrentScoreBool'] = plays.apply(lambda x: 1 if x.ScoreDiff > 0 else 0, axis=1)

# add column to show if the Brownies are playing at home
plays['Home'] = plays.apply(lambda x: 1 if x.HomeTeam == team else 0, axis=1)

# changing the timeouts attributes to reflect the posteam: CLE and the defensive teams remaining timeouts
plays['PosTO_PreSnap'] = plays.apply(lambda x: x.HomeTimeouts_Remaining_Pre if x.HomeTimeouts_Remaining_Pre == team else x.AwayTimeouts_Remaining_Pre, axis=1)
plays['DefTO_PreSnap'] = plays.apply(lambda x: x.HomeTimeouts_Remaining_Pre if x.HomeTimeouts_Remaining_Pre != team else x.AwayTimeouts_Remaining_Pre, axis=1)

# indicator for 2-minute situations
plays['TwoMinuteDrill'] = plays.apply(lambda x: 1 if (
    (((x.TimeSecs <= 0)&(x.TimeSecs >= 120))|((x.TimeSecs <= 1920)&(x.TimeSecs >= 1800)))&
    (x.CurrentScoreBool == 0)) else 0, axis=1)
                                      
plays.info()
# need to clean float data and transfer to integer
plays.TimeSecs = plays.TimeSecs.fillna(0).astype(int)
plays.yrdline100 = plays.yrdline100.fillna(0).astype(int)
plays.down = plays.down.fillna(0).astype(int)
plays.PosTeamScore = plays.PosTeamScore.fillna(0).astype(int)
plays.DefTeamScore = plays.DefTeamScore.fillna(0).astype(int)
plays.RushingMean = plays.RushingMean.fillna(0).astype(int)
plays.PassingMean = plays.PassingMean.fillna(0).astype(int)
plays.ScoreDiff = plays.ScoreDiff.fillna(0).astype(int)
plays.PlayTimeDiff = plays.PlayTimeDiff.fillna(0).astype(int)
plays.GoalToGo = plays.GoalToGo.fillna(0).astype(int)

plays.RushingWA = plays.RushingWA.fillna(0).round(0).astype(int)
plays.PassingWA = plays.PassingWA.fillna(0).round(0).astype(int)
# play type changed to integer using map - removing others
# PlayTypes = {"Run": 0, "QB Kneel": 0, "Pass": 1, "Sack": 1, "Spike": 1}
# cle.PlayType = cle.PlayType.map(PlayTypes)
# cle.PlayType = cle.PlayType.fillna(0)
# cle.PlayType = cle.PlayType.astype(int)
plays = plays[(plays.PassAttempt == 1)|(plays.RushAttempt == 1)]
plays['PlayType'] = plays.apply(lambda x: 1 if x.PassAttempt == 1 else 0, axis=1)
plays.PlayType = plays.PlayType.fillna(0).astype(int)


# changing float64 to float32
plays.No_Score_Prob = plays.No_Score_Prob.fillna(0).astype(np.float32)
plays.Opp_Field_Goal_Prob = plays.Opp_Field_Goal_Prob.fillna(0).astype(np.float32)
plays.Opp_Safety_Prob = plays.Opp_Safety_Prob.fillna(0).astype(np.float32)
plays.Win_Prob = plays.Win_Prob.fillna(0).astype(np.float32)
plays.ExpPts = plays.ExpPts.fillna(0).astype(np.float32)


plays.No_Score_Prob = pd.qcut(plays['No_Score_Prob'], 5, labels=False)
plays.Opp_Field_Goal_Prob = pd.qcut(plays['Opp_Field_Goal_Prob'], 5, labels=False)
plays.Opp_Safety_Prob = pd.qcut(plays['Opp_Safety_Prob'], 5, labels=False)
plays.Win_Prob = pd.qcut(plays['Win_Prob'], 5, labels=False)
plays.ExpPts = pd.qcut(plays['ExpPts'], 5, labels=False)

# drop unneeded columns to begin to de-clutter the set
plays = plays[plays.down != 0]
plays = plays.drop(columns=['PlayAttempted','HomeTeam','Team','DefensiveTeam',
                        'HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre','RushAttempt','PassAttempt'])

plays = plays.rename(columns = {'Drive_x':'Drive'})
plays.head(5)
# Define our prediction data
plays_predictors = ['ydstogo','down','ScoreDiff','PosTO_PreSnap','No_Score_Prob','Drive','Season','TimeSecs','TimeUnder','PlayTimeDiff','Opp_Field_Goal_Prob']
X = plays[plays_predictors]

# Define the prediction target: PlayType
y = plays.PlayType
# Split our data into training and test data for both our target and prediction data sets
# random state = 0 means we get same result everytime if we want ot change later
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Decision Tree Classifier
desc_tree = DecisionTreeClassifier()
desc_tree.fit(train_X, train_y)

dt_predictions = desc_tree.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, dt_predictions)
dt_roc_auc = auc(false_positive_rate, true_positive_rate)
# Random Forest Classification
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_y)

rf_predictions = random_forest.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, rf_predictions)
rf_roc_auc = auc(false_positive_rate, true_positive_rate)
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train_X, train_y)

lr_predictions = log_reg.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, lr_predictions)
lr_roc_auc = auc(false_positive_rate, true_positive_rate)
# K-Means Clustering
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_X, train_y)

knn_predictions = knn.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, knn_predictions)
knn_roc_auc = auc(false_positive_rate, true_positive_rate)
# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_X, train_y)

gnb_predictions = gnb.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, gnb_predictions)
gnb_roc_auc = auc(false_positive_rate, true_positive_rate)
gbc = GradientBoostingClassifier()
gbc.fit(train_X, train_y)

gbc_predictions = gbc.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, gbc_predictions)
gbc_roc_auc = auc(false_positive_rate, true_positive_rate)
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN',
              'Naive Bayes', 'Gradient Boosting Classifier'],
    'AUC': [dt_roc_auc, rf_roc_auc, lr_roc_auc, knn_roc_auc, gnb_roc_auc, gbc_roc_auc]})
result_df = results.sort_values(by='AUC', ascending=False)
result_df = result_df.set_index('AUC')
result_df.head(7)
importances = pd.DataFrame({'feature':train_X.columns,'importance':np.round(gbc.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar(figsize=(20,9),rot=0)
x_train, x_test, y_train, y_test = train_test_split(X, y,random_state = 0)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
# a couple learning rates to see how they affect the outcome of our model
learning_rates = [0.2, 0.175 ,0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01]

train_results = []
test_results = []
train_results = []
test_results = []
for eta in learning_rates:
    model = GradientBoostingClassifier(learning_rate=eta)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()
# n_estimators to adjust to tune outcome
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

train_results = []
test_results = []
for estimator in n_estimators:
    model = GradientBoostingClassifier(n_estimators=estimator)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()
max_depths = np.linspace(1, 7, 7, endpoint=True)

train_results = []
test_results = []
for max_depth in max_depths:
    model = GradientBoostingClassifier(max_depth=max_depth)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    model = GradientBoostingClassifier(min_samples_split=min_samples_split)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()
max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    model = GradientBoostingClassifier(max_features=max_feature)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Maximum Features')
plt.show()
