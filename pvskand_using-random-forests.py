

import csv as csv

import numpy as np

import pandas as pd

from statistics import mode

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split





def input(train_df):

	train_df['class'] = train_df['class'].map( {'p': 0, 'e': 1} ).astype(int)

	train_df['cap-shape'] = train_df['cap-shape'].map( {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's' : 5 } ).astype(int)

	train_df['cap-surface'] = train_df['cap-surface'].map( {'f': 0, 'g': 1, 'y': 2, 's' : 3} ).astype(int)

	train_df['cap-color'] = train_df['cap-color'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9} ).astype(int)

	train_df['bruises'] = train_df['bruises'].map( {'f': 0, 't': 1} ).astype(int)

	train_df['odor'] = train_df['odor'].map( {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8} ).astype(int)

	train_df['gill-attachment'] = train_df['gill-attachment'].map( {'a': 0, 'd': 1, 'f': 2, 'n': 3} ).astype(int)

	train_df['gill-spacing'] = train_df['gill-spacing'].map( {'c': 0, 'w': 1, 'd': 2} ).astype(int)

	train_df['gill-size'] = train_df['gill-size'].map( {'n': 0, 'b': 1} ).astype(int)

	train_df['gill-color'] = train_df['gill-color'].map( {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g' : 4, 'r' : 5, 'o' : 6, 'p' : 7, 'u' : 8, 'e' : 9, 'w' : 10, 'y' : 11} ).astype(int)

	train_df['stalk-shape'] = train_df['stalk-shape'].map( {'t': 0, 'e': 1} ).astype(int)

	# missing values in stalk-root 

	train_df['stalk-root'] = train_df['stalk-root'].map( {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?' : -1} ).astype(int)

	train_df['stalk-surface-above-ring'] = train_df['stalk-surface-above-ring'].map( {'f': 0, 'y': 1, 'k': 2, 's': 3} ).astype(int)

	train_df['stalk-surface-below-ring'] = train_df['stalk-surface-below-ring'].map( {'f': 0, 'y': 1, 'k': 2, 's': 3} ).astype(int)

	train_df['stalk-color-above-ring'] = train_df['stalk-color-above-ring'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o' : 4, 'p' : 5, 'e' : 6, 'w' : 7, 'y' : 8} ).astype(int)

	train_df['stalk-color-below-ring'] = train_df['stalk-color-below-ring'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o' : 4, 'p' : 5, 'e' : 6, 'w' : 7, 'y' : 8} ).astype(int)

	train_df['veil-type'] = train_df['veil-type'].map( {'p': 0, 'u': 1} ).astype(int)

	train_df['veil-color'] = train_df['veil-color'].map( {'n': 0, 'o': 1, 'w': 2, 'y': 3} ).astype(int)

	train_df['ring-number'] = train_df['ring-number'].map( {'n': 0, 'o': 1, 't': 2} ).astype(int)

	train_df['ring-type'] = train_df['ring-type'].map( {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n' : 4, 'p' : 5, 's' : 6, 'z' : 7} ).astype(int)

	train_df['spore-print-color'] = train_df['spore-print-color'].map( {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r' : 4, 'o' : 5, 'u' : 6, 'w' : 7, 'y' : 8} ).astype(int)

	train_df['population'] = train_df['population'].map( {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v' : 4, 'y' : 5} ).astype(int)

	train_df['habitat'] = train_df['habitat'].map( {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u' : 4, 'w' : 5, 'd' : 6} ).astype(int)



def missing(train_data):

# replacing the missing values with the mode of that class

	edible = []

	pois = []

	for i in range(len(train_data)):

		if train_data[i][0] == 0 and train_data[i][11] != -1:

			pois.append(train_data[i][11])

		elif train_data[i][0] == 1 and train_data[i][11] != -1:

			edible.append(train_data[i][11])



	edible_mode = mode(edible)

	pois_mode = mode(pois)



	for i in range(len(train_data)):

		if train_data[i][0] == 0 and train_data[i][11] == -1:

			train_data[i][11] = pois_mode

		elif train_data[i][0] == 1 and train_data[i][11] == -1:

			train_data[i][11] = edible_mode





train_df = pd.read_csv('../input/mushrooms.csv', header=0)        # Load the train file into a dataframe





input(train_df)



train_data = train_df.values

missing(train_data)

X_train, X_test, y_train, y_test = train_test_split( train_data[0::,1::], train_data[0::,0], test_size=0.33	, random_state=42)







# Training the validation set

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( X_train, y_train )





# predicting on test set

output = forest.predict(X_test).astype(int)



accuracy = 0

for i in range(len(output)):

	if output[i] == y_test[i]:

		accuracy = accuracy + 1



print("Accuracy on Test data set = ",accuracy*100/len(output), "%")
