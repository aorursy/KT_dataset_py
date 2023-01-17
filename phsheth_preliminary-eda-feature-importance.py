# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



import seaborn as sns

sns.set()



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")
train.head()
test.head()
plt.scatter(train['Elevation'], train['Slope'], c=train['Cover_Type'], s = 0.5)



plt.show()
sns.pairplot(train[['Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points']])
sns.boxplot(x = train['Cover_Type'], y = train['Elevation'])
sns.swarmplot(x = train['Cover_Type'], y = train['Elevation'])
sns.boxplot(x = train['Cover_Type'], y = train['Elevation'])
# get target

y = train['Cover_Type']



# get features (TODO feature extraction)

X = train.drop(['Cover_Type'],axis=1)

Xtest = test



# split data into training and validation data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 42)
train_X = train_X.drop(['Id'], axis = 1)
val_X = val_X.drop(['Id'], axis = 1)
model_fetimp = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
#thit is how we get the feature importance with simple steps:

model_fetimp.fit(train_X.values, train_y.values.ravel())

# display the relative importance of each attribute

importances = model_fetimp.feature_importances_

#Sort it

print ("Sorted Feature Importance:")

sorted_feature_importance = sorted(zip(importances, list(train_X.columns)), reverse=True)

print (sorted_feature_importance)
features = train_X.columns

importances = model_fetimp.feature_importances_

indices = np.argsort(importances)

figure(num=None, figsize=(6, 12), dpi=300, facecolor='w', edgecolor='k')

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
print([features[i] for i in indices])
importances[indices]
train_X = train_X.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25', 'Soil_Type9', 'Soil_Type36', 'Soil_Type28', 'Soil_Type27', 'Soil_Type21', 'Soil_Type34', 'Soil_Type19', 'Soil_Type26', 'Soil_Type37', 'Soil_Type14', 'Soil_Type18', 'Soil_Type16', 'Soil_Type5', 'Soil_Type1', 'Soil_Type20', 'Soil_Type35', 'Soil_Type31', 'Soil_Type24', 'Soil_Type6', 'Soil_Type11', 'Soil_Type33', 'Wilderness_Area2', 'Soil_Type12', 'Soil_Type23', 'Soil_Type32', 'Soil_Type29', 'Soil_Type22', 'Soil_Type13', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type40', 'Soil_Type4'], axis = 1)
val_X = val_X.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25', 'Soil_Type9', 'Soil_Type36', 'Soil_Type28', 'Soil_Type27', 'Soil_Type21', 'Soil_Type34', 'Soil_Type19', 'Soil_Type26', 'Soil_Type37', 'Soil_Type14', 'Soil_Type18', 'Soil_Type16', 'Soil_Type5', 'Soil_Type1', 'Soil_Type20', 'Soil_Type35', 'Soil_Type31', 'Soil_Type24', 'Soil_Type6', 'Soil_Type11', 'Soil_Type33', 'Wilderness_Area2', 'Soil_Type12', 'Soil_Type23', 'Soil_Type32', 'Soil_Type29', 'Soil_Type22', 'Soil_Type13', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type40', 'Soil_Type4'], axis = 1)
rf = RandomForestRegressor(random_state = 42)

from pprint import pprint# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())
train_X.head()
train_y.head()
val_X.head()
val_y.head()
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
train_X.values
train_y.values.ravel()
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model

rf_random.fit(train_X.values, train_y.values.ravel())
rf_random.best_params_
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)

base_model.fit(train_X.values, train_y.values.ravel())

base_accuracy = evaluate(base_model, val_X.values, val_y.values.ravel())
best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, val_X.values, val_y.values.ravel())
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
test_ids = Xtest["Id"]

test = Xtest.drop(["Id", 'Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25', 'Soil_Type9', 'Soil_Type36', 'Soil_Type28', 'Soil_Type27', 'Soil_Type21', 'Soil_Type34', 'Soil_Type19', 'Soil_Type26', 'Soil_Type37', 'Soil_Type14', 'Soil_Type18', 'Soil_Type16', 'Soil_Type5', 'Soil_Type1', 'Soil_Type20', 'Soil_Type35', 'Soil_Type31', 'Soil_Type24', 'Soil_Type6', 'Soil_Type11', 'Soil_Type33', 'Wilderness_Area2', 'Soil_Type12', 'Soil_Type23', 'Soil_Type32', 'Soil_Type29', 'Soil_Type22', 'Soil_Type13', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type40', 'Soil_Type4'], axis = 1)
test_pred = best_random.predict(test)

test_pred = np.ceil(test_pred)
# Save test predictions to file

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)