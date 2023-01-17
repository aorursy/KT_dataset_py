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
df_train=pd.read_csv('../input/bits-f464-l1/train.csv')

df_test=pd.read_csv('../input/bits-f464-l1/test.csv')

sub=pd.read_csv('../input/bits-f464-l1/sampleSubmission.csv')
from sklearn.model_selection import train_test_split# test_size: what proportion of original data is used for test set

data,label=df_train.drop(columns=['id','label']),df_train['label']
train_data, test_data, train_lbl, test_lbl = train_test_split( data, label, test_size=0.5, random_state=123)
testdata=df_test.drop(columns=['id'])

testdata.info()
'''from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



# Create the model with 100 trees

model = RandomForestRegressor()

# Fit on training data

model.fit(train_data, train_lbl)

predicted = model.predict(test_data)

expected = test_lbl

print(mean_squared_error(expected, predicted))'''
#model.feature_importances_
def display_scores(scores):

    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)



scores = []



for train_index, test_index in kfold.split(data):   

    X_train, X_test = data.iloc[train_index], data.iloc[test_index]

    y_train, y_test = label[train_index], label[test_index]



    rf_model = RandomForestRegressor()

    rf_model.fit(X_train,y_train)

    

    y_pred = rf_model.predict(X_test)

    

    scores.append(mean_squared_error(y_test, y_pred))

    

display_scores(np.sqrt(scores))
'''from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest



rf = RandomForestRegressor(random_state = 42)

n_estimators = [int(x) for x in np.linspace(start = 60, stop = 140, num = 8)]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(50, 150, num = 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(train_data, train_lbl)



rf_random.cv_results_'''
prediction=model.predict(testdata)

prediction
sub['label']=prediction

sub=sub.drop(columns=['id'])

sub.to_csv('submission.csv')
# Get numerical feature importances

importances = list(model.feature_importances_)# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(train_data, importances)]# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
from matplotlib import pyplot as plt

# list of x locations for plotting

x_values = list(range(len(importances)))# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)# Tick labels for x axis

plt.xticks(x_values, train_data. columns, rotation='vertical')# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
# List of features sorted from most to least important

sorted_importances = [importance[1] for importance in feature_importances]

sorted_features = [importance[0] for importance in feature_importances]# Cumulative importances

cumulative_importances = np.cumsum(sorted_importances)# Make a line graph

plt.plot(x_values, cumulative_importances, 'g-')# Draw line at 95% of importance retained

plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')# Format x ticks and labels

plt.xticks(x_values, sorted_features, rotation = 'vertical')# Axis labels and title

plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
print('Number of features for 98% importance:', np.where(cumulative_importances > 0.98)[0][0] + 1)
important_feature_names = [feature[0] for feature in feature_importances[0:10]]

print(important_feature_names)

#important_indices = [list(train_data.columns).index(feature) for feature in important_feature_names]

important_train_features = train_data[important_feature_names]

important_test_features = test_data[important_feature_names]

important_train_features.head()
important_test_features.head()