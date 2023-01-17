import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
%matplotlib inline
# Import training & test flat files
os.chdir('C:\\Users\\933832\\Documents\\Kaggle Data Projects\\Competitions\\PUBG Finish Placement Prediction')
source_train = pd.read_csv('train_V2.csv')
source_test = pd.read_csv('test_V2.csv')
source_train.head()
source_train.info()
d_variable = 'winPlacePerc'
all_variables = source_train.columns
source_train.isna().sum()
source_train = source_train.dropna()
cor_matrix = source_train.corr()
cor_matrix.to_csv("Correlation Matrix" +".csv")
i_variables = [
    #'Id', 
    #'groupId',
    #'matchId',
    #'assists',
    'boosts',
    #'damageDealt',
    #'DBNOs',       
    #'headshotKills',
    #'heals',
    'killPlace',
    #'killPoints',
    #'kills',
    #'killStreaks',
    #'longestKill',
    #'matchDuration',
    #'matchType',
    #'maxPlace',
    #'numGroups',
    #'rankPoints',
    #'revives',
    #'rideDistance',
    #'roadKills',
    #'swimDistance',
    #'teamKills',
    #'vehicleDestroys',
    'walkDistance',
    'weaponsAcquired',
    #'winPoints',
    #'winPlacePerc'
]
X_train = source_train[i_variables]
y_train = source_train[d_variable]
X_test = source_test[i_variables]
#from sklearn.preprocessing import Normalizer
#scaler = Normalizer().fit(X_train)
#normalized_X_train = scaler.transform(X_train)
#normalized_X_test = scaler.transform(X_test)
#XGBoost
#import xgboost
#xgb = xgboost.XGBRegressor()
#xgb.fit(normalized_X_train,y_train)
#preds = xgb.predict(normalized_X_test)
# BayesianRidge
from sklearn import linear_model
br = linear_model.BayesianRidge(normalize=True)
br.fit(X_train,y_train)
preds = br.predict(X_test)
#Create DataFrame for predictions
preds_list = pd.DataFrame(list(preds),columns = ['Predictions'])

# Create DataFrame for submission csv file
submission = pd.DataFrame(source_test['Id'])

# Create new column for presictions in submission DataFrame
submission[d_variable] = preds_list
# Drop index and save as csv file
submission.to_csv('C:\\Users\\933832\\Documents\\Kaggle Data Projects\\Competitions\\PUBG Finish Placement Prediction\\Submissions\\Submission ' + 'v1' + '.csv',index=False)