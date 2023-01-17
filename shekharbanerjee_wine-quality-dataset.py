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
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/wine-quality/winequalityN.csv')
df.shape
df.head()
encoder = LabelEncoder()

encoder.fit(df['type'])

df['type'] = encoder.transform(df['type'])
df = df[['fixed acidity', 'volatile acidity', 'citric acid',

       'residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',

       'quality','type']]
df.columns[:-1]
df[df.columns[:-1]] = df[df.columns[1:]].fillna(0)   

for i in df.columns[:-1]:

    df[i] = df[i].apply(lambda x : float(x))

X = df[df.columns[:-1]]

y = df[['type']]
X.columns
y.columns
df.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

train = X_train.join(y_train)
# Make Predictions with Naive Bayes On The Iris Dataset

from csv import reader

from math import sqrt

from math import exp

from math import pi



# Load a CSV file

def load_csv(filename):

	dataset = list()

	with open(filename, 'r') as file:

		csv_reader = reader(file)

		for row in csv_reader:

			if not row:

				continue

			dataset.append(row)

	return dataset



# Convert string column to float

def str_column_to_float(dataset, column):

	for row in dataset:

		row[column] = float(row[column].strip())



# Convert string column to integer

def str_column_to_int(dataset, column):

	class_values = [row[column] for row in dataset]

	unique = set(class_values)

	lookup = dict()

	for i, value in enumerate(unique):

		lookup[value] = i

		print('[%s] => %d' % (value, i))

	for row in dataset:

		row[column] = lookup[row[column]]

	return lookup



# Split the dataset by class values, returns a dictionary

def separate_by_class(dataset):

	separated = dict()

	for i in range(len(dataset)):

		vector = dataset[i]

		class_value = vector[-1]

		if (class_value not in separated):

			separated[class_value] = list()

		separated[class_value].append(vector)

	return separated



# Calculate the mean of a list of numbers

def mean(numbers):

	return sum(numbers)/float(len(numbers))



# Calculate the standard deviation of a list of numbers

def stdev(numbers):

	avg = mean(numbers)

	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)

	return sqrt(variance)



# Calculate the mean, stdev and count for each column in a dataset

def summarize_dataset(dataset):

	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]

	del(summaries[-1])

	return summaries



# Split dataset by class then calculate statistics for each row

def summarize_by_class(dataset):

	separated = separate_by_class(dataset)

	summaries = dict()

	for class_value, rows in separated.items():

		summaries[class_value] = summarize_dataset(rows)

	return summaries



# Calculate the Gaussian probability distribution function for x

def calculate_probability(x, mean, stdev):

	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))

	return (1 / (sqrt(2 * pi) * stdev)) * exponent



# Calculate the probabilities of predicting each class for a given row

def calculate_class_probabilities(summaries, row):

	total_rows = sum([summaries[label][0][2] for label in summaries])

	probabilities = dict()

	for class_value, class_summaries in summaries.items():

		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)

		for i in range(len(class_summaries)-2):

			mean, stdev, _ = class_summaries[i]

			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

	return probabilities



# Predict the class for a given row

def predict(summaries, row):

	probabilities = calculate_class_probabilities(summaries, row)

	best_label, best_prob = None, -1

	for class_value, probability in probabilities.items():

		if best_label is None or probability > best_prob:

			best_prob = probability

			best_label = class_value

	return best_label



# Make a prediction with Naive Bayes on Iris Dataset



dataset = train.values

model = summarize_by_class(dataset)

# define a new record

test = X_test.values

label = []

# predict the label

for i in range(len(test)):

    label.append(predict(model,list(test[i][:])))

    

model
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, label))
X = df[df.columns[:-1]]

y = df['type'].ravel()
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train)

print(gnb.score(X_test, y_test))
from sklearn.ensemble import AdaBoostClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
from sklearn import ensemble





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01}



xg_model = ensemble.GradientBoostingClassifier(**params)





xg_model.fit(X_train,y_train)

xg_model.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

lin = LogisticRegression(solver = 'liblinear')

lin.fit(X_train,y_train)

print(lin.score(X_test,y_test))
X = df[df.columns[:-1]]

y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
dtree.score(X_test,y_test)
X = df[df.columns[:-1]]

y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

train_data = X_train.join(y_train)
# k-nearest neighbors on the Iris Flowers Dataset

from random import seed

from random import randrange

from csv import reader

from math import sqrt





# Find the min and max values for each column

def dataset_minmax(dataset):

	minmax = list()

	for i in range(len(dataset[0])):

		col_values = [row[i] for row in dataset]

		value_min = min(col_values)

		value_max = max(col_values)

		minmax.append([value_min, value_max])

	return minmax



# Rescale dataset columns to the range 0-1

def normalize_dataset(dataset, minmax):

	for row in dataset:

		for i in range(len(row)):

			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])





# Calculate accuracy percentage

def accuracy_metric(actual, predicted):

	correct = 0

	for i in range(len(actual)):

		if actual[i] == predicted[i]:

			correct += 1

	return correct / float(len(actual)) * 100.0







# Calculate the Euclidean distance between two vectors

def euclidean_distance(row1, row2):

	distance = 0.0

	for i in range(len(row1)-1):

		distance += (row1[i] - row2[i])**2

	return sqrt(distance)



# Locate the most similar neighbors

def get_neighbors(train, test_row, num_neighbors):

	distances = list()

	for train_row in train:

		dist = euclidean_distance(test_row, train_row)

		distances.append((train_row, dist))

	distances.sort(key=lambda tup: tup[1])

	neighbors = list()

	for i in range(num_neighbors):

		neighbors.append(distances[i][0])

	return neighbors



# Make a prediction with neighbors

def predict_classification(train, row, num_neighbors):

    neighbors = get_neighbors(train, row, num_neighbors)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction



# kNN Algorithm

def k_nearest_neighbors(train, test, num_neighbors):

	predictions = list()

	for row in test:

		output = predict_classification(train, row, num_neighbors)

		predictions.append(output)



	return(predictions)



# Test the kNN on the Iris Flowers dataset

train = train_data.values

test = X_test.values

n_folds = 5

num_neighbors = 5



scores = k_nearest_neighbors(train, test, num_neighbors)

#print('Scores: %s' % scores)

#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

print(accuracy_score(y_test, scores))