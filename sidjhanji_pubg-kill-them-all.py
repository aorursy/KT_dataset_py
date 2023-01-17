# Importing the library
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# training set
train = pd.read_csv("../input/train.csv")
train.head()
# test dataset
test = pd.read_csv("../input/test.csv")
test.head()
train.shape
train.info()
import seaborn as sns
correlations = train.corr()
sns.heatmap(correlations)
# Lets define a custom function to get a better view
# custom function to set the style for heatmap
import numpy as np
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
plot_correlation_heatmap(train)
# Calling the dataframe.pivot_table() function for assists
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "assists", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()
# Calling the dataframe.pivot_table() function for boosts
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "boosts", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()
# Calling the dataframe.pivot_table() function for DBNOs
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "DBNOs", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()
# Calling the dataframe.pivot_table() function for kills
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "kills", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()
# predictors  = [ "assists",
#                 "boosts",
#                 "damageDealt",        
#                 "DBNOs",              
#                 "headshotKills",      
#                 "heals",     
#                 "killPlace",          
#                 "killPoints",         
#                 "kills",              
#                 "killStreaks",
#                 "longestKill",        
#                 "maxPlace",           
#                 "numGroups",          
#                 "revives",            
#                 "rideDistance",       
#                 "roadKills",          
#                 "swimDistance",       
#                 "teamKills",          
#                 "vehicleDestroys",    
#                 "walkDistance",       
#                 "weaponsAcquired",    
#                 "winPoints"]

# x_train = train[predictors]
# x_train.head()
# Some Feature Engineering
train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
# train["healthpack"] = train["boosts"] + train["heals"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.head()
test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
# test["healthpack"] = test["boosts"] + test["heals"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test["distance"].head()
predictors  = [ "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "boosts",
                "heals",
                "revives",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "skill",
                "assists",
                "damageDealt",
                "DBNOs",
                "killPlace",
                "killPoints",
                "vehicleDestroys",
                "longestKill"
               ]
x_train = train[predictors]
x_train.head()
y_train = train["winPlacePerc"]
y_train.head()
# # Using Random Forest Regressor
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# regressor.fit(x_train, y_train)
# # Finding the cross validation score with 10 folds
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(regressor, x_train, y_train, cv=10)
# print(scores)
# accuracy = scores.mean()
# print(accuracy)
# from sklearn.feature_selection import RFECV
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# selector = RFECV(regressor, cv = 10)
# selector.fit(x_train, y_train)

# optimized_predictors = x_train.columns[selector.support_]
# print(optimized_predictors)
# predictors = train[optimized_predictors]
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# regressor.fit(predictors, y_train)
# # Finding the cross validation score with 10 folds
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(regressor, x_train, y_train, cv=10)
# print(scores)
# accuracy = scores.mean()
# print(accuracy)
# import lightgbm as lgb
# from sklearn.feature_selection import RFECV
# regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5)

# selector = RFECV(regressor, cv = 10)
# selector.fit(x_train, y_train)

# optimized_predictors = x_train.columns[selector.support_]
# print(optimized_predictors)
import lightgbm as lgb
regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.8)
regressor.fit(x_train, y_train)
# Finding the cross validation score with 10 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, x_train, y_train, cv=10)
print(scores)
accuracy = scores.mean()
print(accuracy)
x_test = test[predictors]
x_test.head()
y_predict = regressor.predict(x_test)
print(y_predict)
y_predict[y_predict > 1] = 1
test['winPlacePercPredictions'] = y_predict

aux = test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = test[['Id','winPlacePerc']]
submission.to_csv("kill_them_all.csv", index=False)
lgb.plot_importance(regressor, max_num_features=20, figsize=(10, 8));
plt.title('Feature importance');
# Lets take the top 10 features
important_predictors  = [ "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "killPlace",
                "killPoints",
                "longestKill"
               ]
x_train = train[important_predictors]
x_train.head()
regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.8)
regressor.fit(x_train, y_train)
x_test = test[important_predictors]
x_test.head()
y_predict = regressor.predict(x_test)
print(y_predict)
y_predict[y_predict > 1] = 1
test['winPlacePercPredictions'] = y_predict

aux = test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
test.head()
# Finding the cross validation score with 10 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, x_train, y_train, cv=10)
print(scores)
accuracy = scores.mean()
print(accuracy)