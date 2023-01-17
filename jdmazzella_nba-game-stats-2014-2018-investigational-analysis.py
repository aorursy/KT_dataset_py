import numpy as np  # linear algebra
import pandas as pd # data process cvs file 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# results writen to current dir are saved as output
import pandas as pd
import xgboost as xgb
from decimal import Decimal
from IPython.display import display
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data = pd.read_csv("/kaggle/input/nba-games-stats-from-2014-to-2018/nba.games.stats.csv")
# observing data relationships
scatter_matrix(data[['TeamPoints', 'FieldGoals', 'X3PointShots', 'FreeThrows', 'Steals', 'Blocks', 'Turnovers', 'TotalFouls', 'TotalRebounds']], figsize=(15,15))
# prep data
X_all = data.drop(['WINorLOSS', 'Date', 'OpponentPoints', 'Opp.FieldGoals','Opp.FieldGoalsAttempted','Opp.FieldGoals.','Opp.3PointShots',
                   'Opp.3PointShotsAttempted','Opp.3PointShots.','Opp.FreeThrows','Opp.FreeThrowsAttempted','Opp.FreeThrows.',
                   'Opp.OffRebounds','Opp.TotalRebounds','Opp.Assists','Opp.Steals','Opp.Blocks','Opp.Turnovers','Opp.TotalFouls'], 1)
Y_all = data['WINorLOSS']
# this standardizes the data, by normalizing it on a scale of -1 to 1
from sklearn.preprocessing import scale

# centralize the mean and component-wise scale to unit variance.
cols = [["TeamPoints","FieldGoals","FieldGoalsAttempted","FieldGoals.","X3PointShots",
         "X3PointShotsAttempted","X3PointShots.","FreeThrows","FreeThrowsAttempted","FreeThrows.","OffRebounds","TotalRebounds",
         "Assists","Steals","Blocks","Turnovers","TotalFouls"]]
for col in cols:
    X_all[col] = scale(X_all[col])
    
# remove catagorical variables if necessary
def remove_cats(unscaledX):
    ''' Preprocesses the basketball data and converts categorical variables into dummy variables. '''
    
    # initialize new DataFrame
    output = pd.DataFrame(index = unscaledX.index)

    # investigate each feature column for data
    for col, col_data in unscaledX.iteritems():

        # convert to dummy if catagorical
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        output = output.join(col_data)
    
    return output

X_all = remove_cats(X_all)
display(X_all.head())


from sklearn.model_selection import train_test_split

# randomly split data into two sets: test and train
training_set_x, testing_set_x, training_set_y, testing_set_y = train_test_split(X_all, Y_all, 
                                                    test_size = 50,
                                                    random_state = 2,
                                                    stratify = Y_all)

display(training_set_x)
display(training_set_y)

display(testing_set_x)
display(testing_set_y)
from time import time 
from sklearn.metrics import f1_score

def train_clf(clf, training_set_x, training_set_y):
    
    # start timer
    start = time()
    # train classifier
    clf.fit(training_set_x, training_set_y)
    display(clf)
    # stop timer
    end = time() 
    # print results
    print(f"Model is trained in {end-start} seconds")

    
def prediction(clf, features, target):
    
    # start timer
    start = time()
    # make prediction
    y_predict = clf.predict(features)
    display(y_predict)
    # stop timer
    end = time()
    # print results
    print (f"Predictions made in {end-start} seconds.")
    
    return f1_score(target, y_predict, pos_label='W'), sum(target == y_predict) / float(len(y_predict))


def predictor_trainer(clf, training_set_x, training_set_y, testing_set_x, testing_set_y):
    
    print (f"Using {clf.__class__.__name__} to training a set size of {len(training_set_x)}. . .")
    train_clf(clf, training_set_x, training_set_y) 
    f1, accuracy = prediction(clf, training_set_x, training_set_y)
    print (f"{f1:.4f}, {accuracy:.4f}")
    print (f"F1 score and accuracy score for training set: {f1} , {accuracy}.")
    f1, accuracy = prediction(clf, testing_set_x, testing_set_y)
    print (f"F1 score and accuracy score for test set: {f1} , {accuracy}.")


clf_logReg = LogisticRegression(random_state = 42)
clf_SVC = SVC(random_state = 912, kernel='rbf')
clf_XGB = xgb.XGBClassifier(seed = 420)

predictor_trainer(clf_logReg, training_set_x, training_set_y, testing_set_x, testing_set_y)
print ('')
predictor_trainer(clf_SVC, training_set_x, training_set_y, testing_set_x, testing_set_y)
print ('')
predictor_trainer(clf_XGB, training_set_x, training_set_y, testing_set_x, testing_set_y)
print ('')