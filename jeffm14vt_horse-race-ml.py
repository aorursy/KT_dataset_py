# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

rawData = pd.read_csv('../input/horses.csv')
rawData = rawData.dropna()

# Any results you write to the current directory are saved as output.
rawData.head()
basicFeatures = rawData[['runner_id', 'days_since_last_run', 'condition', 'barrier', 'blinkers', 'favourite_odds_win', 'sex', 'jockey_sex', 'vic_tote', 'venue_name', 'race_number', 'soft_starts', 'overall_wins']]
#basicFeatures = rawData['runner_id', 'days_since_last_run']
#basidFeaturesENC = pd.get_dummies(basicFeatures)
targetDF = rawData['position']
featuresFinal = pd.get_dummies(basicFeatures)
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(featuresFinal, 
                                                    targetDF, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    #start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    #end = time() # Get end time
    
    # TODO: Calculate the training time
    #results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    #start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    #end = time() # Get end time
    
    #print (type(predictions_test))
    #print (predictions_train)
    # TODO: Calculate the total prediction time
    #results['pred_time'] = end - start
    
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    # 3rd param is Beta
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    #print ("{} trained on {} samples.").format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results

len(y_train)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
samples = 177344
train_predict(clf, samples, X_train, y_train, X_test, y_test)
X_train.columns
print(len(clf.feature_importances_))
print(len(featuresFinal))
feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp