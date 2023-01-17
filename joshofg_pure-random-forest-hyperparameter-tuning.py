# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



# Load the data in. Remember to drop the ID column or put it as the index

train_data = pd.read_csv("/kaggle/input/learn-together/train.csv", index_col='Id')

test_data = pd.read_csv("/kaggle/input/learn-together/test.csv", index_col='Id')



X_train = train_data.drop('Cover_Type', axis='columns')

y_train = train_data['Cover_Type']



# This just puts the model into a pipeline. Useful for when things get more hairy later.

rf = RandomForestClassifier()

model = Pipeline(steps=[('model',rf),])



#====== IMPORTANT =========

# I left the commented lines running for a few hours this morning, and then used the best parameters

# from it.



# In this notebook, I just use the parameters from this random search that gave me the best results

#param_grid = {'model__n_estimators': np.logspace(2,3.5,8).astype(int),

#              'model__max_features': [0.1,0.3,0.5,0.7,0.9],

#              'model__max_depth': np.logspace(0,3,10).astype(int),

#              'model__min_samples_split': [2, 5, 10],

#              'model__min_samples_leaf': [1, 2, 4],

#              'model__bootstrap':[True, False]}



# Here, I just manually put in the optimal parameters found above.



param_grid = {'model__n_estimators': [719],

              'model__max_features': [0.3],

              'model__max_depth': [464],

              'model__min_samples_split': [2],

              'model__min_samples_leaf': [1],

              'model__bootstrap':[False]}





grid = RandomizedSearchCV(estimator=model, 

                          param_distributions=param_grid, 

                          n_iter=1, # This was set to 100 in my offline version

                          cv=3, 

                          verbose=3, 

                          n_jobs=1,

                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 

                          refit='NLL')



# fit, predict, and output!

grid.fit(X_train, y_train)

preds = grid.predict(test_data)



output = pd.DataFrame({'Id': test_data.index,

                       'Cover_Type': preds})

output.to_csv('submission.csv', index=False)
