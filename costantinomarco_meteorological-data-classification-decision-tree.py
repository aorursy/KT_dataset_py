import numpy as np

import pandas as pd



import graphviz



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv', parse_dates=['Date'])

print(data.shape)

data.head(3)
condition = data['Events'].str.contains(' ')

to_be_kept = ['Date','TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'Events']



# excluding clear weather

polished_data = data[condition == False]



# i keep only a few choosen columns

polished_data = polished_data.loc[:, polished_data.columns.intersection(to_be_kept)]

X = polished_data[set(list(polished_data.columns))-set(['Events', 'Date'])]

y = polished_data['Events']
from sklearn import tree

from sklearn.metrics import accuracy_score



# this function generates n = len(max_depths) decision trees. Each of depth max_depths[i]

# fits on x_train, y_train, predicts on x_test

# returns classifiers and predictions

def many_decision_trees(max_depths, criterion, x_train, y_train, x_test, y_test):

    clf_array = []

    predictions_array = []

    

    for depth in max_depths:

        clf = tree.DecisionTreeClassifier(max_depth=depth, criterion=criterion)

        clf = clf.fit(x_train, y_train) 

        prediction = clf.predict(x_test)

        

        print('Tree depth: {} Accuracy: {}'.format(depth, accuracy_score(y_test, prediction)))

        

        clf_array.append(clf)

        predictions_array.append(prediction)

    return clf_array, predictions_array
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



depths = list(range(3, 8))



# lets try the decision tree

# first predicting on training data

print('Prediction on training data -------------------------')

clfs_tr, predictions_tr = many_decision_trees(depths, 'entropy', X_train, y_train, X_train, y_train)  



# then predicting on test data

print('\nPrediction on test data -----------------------------')

clfs_te, predictions_te = many_decision_trees(depths, 'entropy', X_train, y_train, X_test, y_test)  
# lets print a tree a see what we've got

print('Classes: {}'.format(clfs_te[0].classes_))

print('Features: {}'.format(list(X)))



dot_data = tree.export_graphviz(clfs_te[0], out_file=None, rounded=True, class_names=clfs_te[0].classes_, 

                                feature_names=list(X), filled=True)

graph = graphviz.Source(dot_data) 

graph
data = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv', parse_dates=['Date'])



to_be_kept = ['Date','TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'Events']

condition = data['Events'].str.contains(',')



# first i remove instances of events that are in the form 'Fog, Rain, Thunderstorm'

polished_data_2 = data[condition == False]



polished_data_2 = polished_data_2.loc[:, polished_data_2.columns.intersection(to_be_kept)]

polished_data_2.Events.replace([' '], ['Clear'], inplace=True)



# trying the code above with this data led to an error: ValueError: could not convert string to float: '-'

# this means we are missing some data in the dataset, for scikit-learn's to understand a value as missing it needs to be NaN

# so we replace '-' with NaN

polished_data_2 = polished_data_2.replace('-', float('NaN'))



X = polished_data_2[set(list(polished_data_2.columns))-set(['Events', 'Date'])]

# Date to days of year since integers are easier to deal with

days_of_year = [date.dayofyear for date in polished_data_2['Date']]

X['DayOfYear'] = days_of_year

y = polished_data_2['Events']

X.head(3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.impute import SimpleImputer



imp = SimpleImputer(missing_values=float('NaN'), strategy='mean')

imp = imp.fit(X_train)

X_train = imp.transform(X_train)

imp = imp.fit(X_test)

X_test = imp.transform(X_test)

    

depths = list(range(3, 8))

# lets try the decision tree

# first predicting on training data

print('Prediction on training data -------------------------')

clfs_tr, predictions_tr = many_decision_trees(depths, 'entropy', X_train, y_train, X_train, y_train)  



# then predicting on test data

print('\nPrediction on test data -----------------------------')

clfs_te, predictions_te = many_decision_trees(depths, 'entropy', X_train, y_train, X_test, y_test)  
# again printing the tree

print('Classes: {}'.format(clfs_te[0].classes_))

print('Features: {}'.format(list(X)))



dot_data = tree.export_graphviz(clfs_te[0], out_file=None, rounded=True, class_names=clfs_te[0].classes_, 

                                feature_names=list(X), filled=True)

graph = graphviz.Source(dot_data) 

graph
classes_occurrences = polished_data_2['Events'].value_counts().to_frame()

classes_occurrences = classes_occurrences.sort_index(axis=0)

print(classes_occurrences)