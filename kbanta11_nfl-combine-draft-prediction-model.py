import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Combine all the csv's into one dataframe
def compile_csv(startYear, endYear):
    df_list = []
    for i in range(startYear, endYear + 1):
        off_filename = "../input/"+ str(i) + 'Offense.csv'
        def_filename = "../input/" + str(i) + 'Defense.csv'
        off_df = pd.read_csv(off_filename)
        off_df["Unit"] = ["Offense" if pos != "LS" else "Special" for pos in off_df["Pos"]]
        def_df = pd.read_csv(def_filename)
        def_df["Unit"] = ["Defense" if (pos != "K" and pos != "P") else "Special" for pos in def_df["Pos"]]
        df_list.append(off_df)
        df_list.append(def_df)
    data = pd.concat(df_list)
    return data

df = compile_csv(2000, 2017)
# Format player name
df["Player"] = [x.split("\\")[0] for x in df["Player"]]

# Parse out Drafted (tm/rnd/yr) column
df["Drafted (tm/rnd/yr)"] = df["Drafted (tm/rnd/yr)"].where(pd.notnull(df["Drafted (tm/rnd/yr)"]), None)
df["DraftTeam"] = [x.split(" / ")[0] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftRd"] = [x.split(" / ")[1] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftRd"] = df["DraftRd"].str.replace('[a-zA-Z]+', '')
df["DraftPick"] = [x.split(" / ")[2] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftPick"] = df["DraftPick"].str.replace('[a-zA-Z_]+', '')
df = df.drop(["Drafted (tm/rnd/yr)"], axis=1)

# Convert height to inches
def convert_height(x):
    feet = x.split("-")[0]
    inches = x.split("-")[1]
    height = (int(feet) * 12) + int(inches)
    return height
df['Height'] = df['Height'].apply(convert_height)

df.describe()
# Strip out unneeded columns and handle undrafted players
df2 = df[['Year','Pos', 'Height', 'Wt', '40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle','Unit', 'DraftRd']]
df2['DraftRd'] = df2['DraftRd'].fillna(8)

# impute missing values for events which players skipped with a -1
df2 = df2.fillna('-1')
df2 = df2.apply(pd.to_numeric, errors='ignore')

# calculate quartile in each event for each player within their combine year/position 
rank_40 = []
rank_vert = []
rank_bench = []
rank_broad = []
rank_3cone = []
rank_shuttle = []
for index, row in df2.iterrows():
    year = row['Year']
    position = row['Pos']
    #Calculate 40YD quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['40YD'] != -1)]["40YD"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['40YD'] != -1)]["40YD"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['40YD'] != -1)]["40YD"].quantile(.75)
    quartile = 4
    if row['40YD'] == -1:
        quartile = 5
    elif row['40YD'] <= quartile_1:
        quartile = 1
    elif row['40YD'] <= median:
        quartile = 2
    elif row['40YD'] <= quartile_3:
        quartile = 3
    rank_40.append(quartile)
    
    #Calculate vert quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Vertical'] != -1)]["Vertical"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Vertical'] != -1)]["Vertical"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Vertical'] != -1)]["Vertical"].quantile(.75)
    quartile = 4
    if row['Vertical'] == -1:
        quartile = 5
    elif row['Vertical'] >= quartile_3:
        quartile = 1
    elif row['Vertical'] >= median:
        quartile = 2
    elif row['Vertical'] >= quartile_1:
        quartile = 3
    rank_vert.append(quartile)
    
    #Calculate BenchReps quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['BenchReps'] != -1)]["BenchReps"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['BenchReps'] != -1)]["BenchReps"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['BenchReps'] != -1)]["BenchReps"].quantile(.75)
    quartile = 4
    if row['BenchReps'] == -1:
        quartile = 5
    elif row['BenchReps'] >= quartile_3:
        quartile = 1
    elif row['BenchReps'] >= median:
        quartile = 2
    elif row['BenchReps'] >= quartile_1:
        quartile = 3
    rank_bench.append(quartile)
    
    #Calculate Broad Jump quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Broad Jump'] != -1)]["Broad Jump"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Broad Jump'] != -1)]["Broad Jump"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Broad Jump'] != -1)]["Broad Jump"].quantile(.75)
    quartile = 4
    if row['Broad Jump'] == -1:
        quartile = 5
    elif row['Broad Jump'] >= quartile_3:
        quartile = 1
    elif row['Broad Jump'] >= median:
        quartile = 2
    elif row['Broad Jump'] >= quartile_1:
        quartile = 3
    rank_broad.append(quartile)
    
    #Calculate 3Cone quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['3Cone'] != -1)]["3Cone"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['3Cone'] != -1)]["3Cone"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['3Cone'] != -1)]["3Cone"].quantile(.75)
    quartile = 4
    if row['3Cone'] == -1:
        quartile = 5
    elif row['3Cone'] <= quartile_1:
        quartile = 1
    elif row['3Cone'] <= median:
        quartile = 2
    elif row['3Cone'] <= quartile_3:
        quartile = 3
    rank_3cone.append(quartile)
    
    #Calculate Shuttle quartile
    quartile_1 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Shuttle'] != -1)]["Shuttle"].quantile(.25)
    median = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Shuttle'] != -1)]["Shuttle"].quantile(.5)
    quartile_3 = df2[(df2['Year'] == year) & (df2['Pos'] == position) & (df2['Shuttle'] != -1)]["Shuttle"].quantile(.75)
    quartile = 4
    if row['Shuttle'] == -1:
        quartile = 5
    elif row['Shuttle'] <= quartile_1:
        quartile = 1
    elif row['Shuttle'] <= median:
        quartile = 2
    elif row['Shuttle'] <= quartile_3:
        quartile = 3
    rank_shuttle.append(quartile)
df2['40_quartile_yr_pos'] = rank_40 
df2['vert_quartile_yr_pos'] = rank_vert
df2['bench_quartile_yr_pos'] = rank_bench
df2['broad_quartile_yr_pos'] = rank_broad
df2['3cone_quartile_yr_pos'] = rank_3cone
df2['shuttle_quartile_yr_pos'] = rank_shuttle
df2
# split data into features and target sets
y = df2['DraftRd']
X = df2.drop(['DraftRd', 'Year'], axis=1)
X_encoded = pd.get_dummies(X, columns=['Pos', 'Unit'])
# X_encoded
X_encoded_quartiles = X_encoded.drop(['40YD', '3Cone', 'BenchReps', 'Broad Jump', 'Shuttle', 'Vertical'], axis=1)
X_encoded
clf = RandomForestClassifier()
grid_values = {"max_depth": [2, 8, 10, 20], "n_estimators": [10, 20, 50], "max_features": [5, 10, 20]}
grid_rf = GridSearchCV(clf, param_grid = grid_values, cv=5)
grid_rf.fit(X_encoded_quartiles, y)
print("Grid Search Complete: {}, {}".format(grid_rf.best_score_, grid_rf.best_params_))
# plain results: Grid Search Complete: 0.3749112845990064, {'max_depth': 8, 'max_features': 20, 'n_estimators': 50}
# likely collinearity between the event quartiles and events, will retry with just quartiles
# just quartiles: Grid Search Complete: 0.3692334989354152, {'max_depth': 8, 'max_features': 10, 'n_estimators': 50}
feature_importances = pd.Series(grid_rf.best_estimator_.feature_importances_, index=X_encoded_quartiles.columns)
feature_importances.sort_values(ascending=False)

skill = {'speed_skill': ['CB', 'SS', 'FS', 'WR', 'RB'], 'mid_skill': ['OLB', 'ILB', 'FB', 'TE', 'QB', 'P'], 'big_skill': ['DT', 'OT', 'OG', 'DE', 'LS']}

for key, value in skill.items():
    df_skill = df2[df2['Pos'].isin(value)]
    y = df_skill['DraftRd']
    X_encoded = pd.get_dummies(df_skill.drop(['DraftRd'], axis=1), columns=['Pos', 'Unit'])
    
    clf = RandomForestClassifier()
    grid_values = {"max_depth": [4, 6, 8, 10], "n_estimators": [10, 50, 100]}
    grid_skill = GridSearchCV(clf, param_grid = grid_values, cv=5)
    grid_skill.fit(X_encoded, y)
    print("------------\n{}\nScore: {}\nParameters: {}\nFeatures: {}".format(key, grid_skill.best_score_, grid_skill.best_params_, pd.Series(grid_skill.best_estimator_.feature_importances_, X_encoded.columns).sort_values(ascending=False)))

# Try with a k-nearest neighbors classifier - no improvement
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

for key, value in skill.items():
    df_skill = df2[df2['Pos'].isin(value)]
    y = df_skill['DraftRd']
    X_encoded = pd.get_dummies(df_skill.drop(['DraftRd'], axis=1), columns=['Pos', 'Unit'])
    scaler = MinMaxScaler()
#     scaler = StandardScaler()
    X_scaled_encoded = scaler.fit_transform(X_encoded)
    X_df = pd.DataFrame(X_scaled_encoded, columns=X_encoded.columns)
#     print(X_df)
    
    clf = KNeighborsClassifier(n_jobs=-1)
    grid_values = {"n_neighbors": [2, 5, 10, 20, 50, 100, 200]}
    grid_skill = GridSearchCV(clf, param_grid = grid_values, cv=5)
    grid_skill.fit(X_df, y)
    print("------------\n{}\nScore: {}\nParameters: {}\n".format(key, grid_skill.best_score_, grid_skill.best_params_))

# Try with logistic regression classifier - ran faster than random forest, fairly similar results (poor)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

for key, value in skill.items():
    df_skill = df2[df2['Pos'].isin(value)]
    y = df_skill['DraftRd']
    X_encoded = pd.get_dummies(df_skill.drop(['DraftRd'], axis=1), columns=['Pos', 'Unit'])
    scaler = MinMaxScaler()
#     scaler = StandardScaler()
    X_scaled_encoded = scaler.fit_transform(X_encoded)
    X_df = pd.DataFrame(X_scaled_encoded, columns=X_encoded.columns)
#     print(X_df)
    
    clf = LogisticRegression(penalty='l2')
    grid_values = {"C": [0.001, 0.01, 0.1, .5, 1, 5, 10, 50]}
    grid_skill = GridSearchCV(clf, param_grid = grid_values, cv=5)
    grid_skill.fit(X_df, y)
    print("------------\n{}\nScore: {}\nParameters: {}\n".format(key, grid_skill.best_score_, grid_skill.best_params_))
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y = df2['DraftRd']
X == df2.drop(['DraftRd', 'Year'], axis=1)
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_scaled = scaler.fit_transform(pd.get_dummies(X_train, columns=['Pos', 'Unit']))
X_test_scaled = scaler.transform(pd.get_dummies(X_test, columns=['Pos', 'Unit']))
grid_values = {'n_estimators': [10, 50, 100, 250], 'max_depth': [5, 10, 25, 50], 'learning_rate': [0.001, 0.01, 0.1, .5, 1]}
xgbClf = xgb.XGBClassifier()
grid_cv = GridSearchCV(xgbClf, param_grid = grid_values)
grid_cv.fit(X_train_scaled, y_train)
print("Best Score (accuracy): {}".format(grid_cv.best_score_))
# y_pred = xgbClf.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print("Accuracy: {}\nConfusion Matrix: {}".format(accuracy_score(y_test, y_pred), confusion_matrix(y_test,y_pred)))
print("Best Estimator: {}".format(grid_cv.best_params_))
import matplotlib.pyplot as plt
# create a column for drafted or not (0: not drafted, 1: drafted)
df2['Drafted'] = df2['DraftRd'].apply(lambda x: 0 if x == 8 else 1)

# see how unbalanced the distribution of drafted/undrafted
df2['Drafted'].value_counts().plot(kind='bar')
ax = plt.gca()
plt.show()
print("1: {}\n0: {}".format(len(df2[df2['Drafted'] == 1]), len(df2[df2['Drafted'] == 0])))
len(df2[df2['Drafted'] == 1])/len(df2)
# not too bad of a distribution with it being about 65/35 for drafted/undrafted
#take with grain of salt due to imputed values for skipped events
df2[['3Cone', '40YD', 'Shuttle', 'BenchReps', 'Broad Jump', 'Vertical', 'Wt', 'Height','Pos','Drafted']].groupby(['Pos', 'Drafted']).mean()
from sklearn.ensemble import RandomForestClassifier

df2
y = df2['Drafted']
X = df2.drop(['Year', 'Drafted', 'DraftRd', '40_quartile_yr_pos', 'broad_quartile_yr_pos', 'vert_quartile_yr_pos', 'shuttle_quartile_yr_pos', 'bench_quartile_yr_pos', '3cone_quartile_yr_pos'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train_encoded = pd.get_dummies(X_train, columns=['Pos', 'Unit'])

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

rf = RandomForestClassifier()
grid_values = {'max_depth': [8, 10, 15], 'n_estimators': [50, 100, 150], 'max_features': [10, 20]}
rf_grid = GridSearchCV(rf, param_grid = grid_values, cv=5)
rf_grid.fit(X_train_scaled, y_train)

# Best Score: 0.6978704525288376
# Best Params: {'max_depth': 10, 'max_features': 10, 'n_estimators': 100}
# Best Score: 0.6989795918367347
# Best Params: {'max_depth': 10, 'max_features': 10, 'n_estimators': 50}
# re-running without quartiles
# Best Score: 0.6967613132209406
# Best Params: {'max_depth': 8, 'max_features': 20, 'n_estimators': 100}

print("Best Score: {}\nBest Params: {}\n".format(rf_grid.best_score_, rf_grid.best_params_))

print(pd.Series(rf_grid.best_estimator_.feature_importances_, index = X_train_encoded.columns).sort_values(ascending=False))
# Gradient boost for this binary case with out positions
X_train_xgb = X_train.drop(['Pos'], axis=1)
X_test_xgb = X_test.drop(['Pos'], axis=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(pd.get_dummies(X_train_xgb, columns=['Unit']))
X_test_scaled = scaler.transform(pd.get_dummies(X_test_xgb, columns=['Unit']))

grid_values = {'n_estimators': [10, 50, 100, 250], 'max_depth': [5, 10, 25, 50], 'learning_rate': [0.001, 0.01, 0.1, .5, 1]}
xgbClf = xgb.XGBClassifier()
grid_cv = GridSearchCV(xgbClf, param_grid = grid_values, scoring='recall')

grid_cv.fit(X_train_scaled, y_train)
print("Best Score (recall): {}".format(grid_cv.best_score_))
y_pred = grid_cv.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print("Accuracy: {}\nConfusion Matrix: {}".format(accuracy_score(y_test, y_pred), confusion_matrix(y_test,y_pred)))
print("Best Estimator: {}".format(grid_cv.best_params_))
