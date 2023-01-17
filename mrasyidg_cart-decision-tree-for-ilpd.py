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
from __future__ import print_function # For Python 2 / 3 compatability
"""Create the column names and load the dataset with it"""



col_names = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine',

            'aspartate', 'total_protein', 'albumin', 'A/G Ratio', 'label']



dataset = pd.read_csv("../input/liver-patients/liver_patient.csv", header=None, names=col_names)



"""Split the dataset into features/attributes and target/label"""



feature_cols = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine',

            'aspartate', 'total_protein', 'albumin', 'A/G Ratio']



classification_data = dataset[feature_cols] # Features

classification_label = dataset.label # Target variable
"""Create array for each dataset."""



dataset_array = dataset.to_numpy()

class_dataset = classification_data.to_numpy()

label_dataset = classification_label.to_numpy()



"""Also create a header for the Questions."""



header = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine', 

          'aspartate', 'total_protein', 'albumin', 'A/G Ratio', 'label']
to_be_splitted = pd.read_csv("../input/liver-patients/liver_patient.csv", header=None, names=col_names) #load the dataset



dataset_copy = to_be_splitted.copy()

train_set = dataset_copy.sample(frac=0.60, random_state=0)

test_set = dataset_copy.drop(train_set.index)
print("Training Set: ")

display(train_set)

print("Test Set: ")

display(test_set)
# Effectively we can do that with this code (with Pandas):

train_mean = train_set.mean()

test_mean = test_set.mean()



print("Train Set: (Look at the Output) You can see the difference, as NaN values are replaced.")

train_set.fillna(train_mean).round(3) # Decimal values are rounded to 3 decimal places.
print("Test Set: (look at the Output) You can see the difference, as NaN values are replaced.")

test_set.fillna(test_mean).round(3) # Decimal values are rounded to 3 decimal places.
class DecisionTreeClassifier:

    

    

    def __init__(self, tree):

        self.tree = DecisionTreeClassifier.build_tree(tree)

        

    def unique_vals(rows, col):

        # Used to find the unique values for "a" column in a dataset

        return set([row[col] for row in rows])

    

    def unique_label(rows):

        # Used to find the unique values for classification_label, note that there is only one column for label. # Rasyid

        return set([row for row in rows])

    

    def class_counts(rows):

        # Used for dataset array. Returns a dictionary of label -> count.

        counts = {}

        for row in rows:

            # in our dataset format, the label is always the last column

            label = row[-1]

            if label not in counts:

                counts[label] = 0

            counts[label] += 1

        return counts

        

    def partition(rows, question):

        # Used to split a dataset into true set and false set.

        true_rows, false_rows = [], []

        for row in rows:

            if question.match(row):

                true_rows.append(row)

            else:

                false_rows.append(row)

        return true_rows, false_rows

        

    def gini(rows):

        # Used to count the impurity.

        counts = DecisionTreeClassifier.class_counts(rows)

        impurity = 1

        for lbl in counts:

            prob_of_lbl = counts[lbl] / float(len(rows))

            impurity -= prob_of_lbl**2

        return impurity

    

    def info_gain(left, right, current_uncertainty):

        p = float(len(left)) / (len(left) + len(right))

        return current_uncertainty - p * DecisionTreeClassifier.gini(left) - (1 - p) * DecisionTreeClassifier.gini(right)

    

    def find_best_split(rows):

        best_gain = 0  # keep track of the best information gain

        best_question = None  # keep train of the feature / value that produced it

        current_uncertainty = DecisionTreeClassifier.gini(rows)

        n_features = len(rows[0]) - 1  # number of columns



        for col in range(n_features):  # for each feature



            values = set([row[col] for row in rows])  # unique values in the column



            for val in values:  # for each value



                question = Question(col, val)



                # try splitting the dataset

                true_rows, false_rows = DecisionTreeClassifier.partition(rows, question)



                # Skip this split if it doesn't divide the dataset.

                if len(true_rows) == 0 or len(false_rows) == 0:

                    continue



                # Calculate the information gain from this split

                gain = DecisionTreeClassifier.info_gain(true_rows, false_rows, current_uncertainty)



                if gain >= best_gain:

                    best_gain, best_question = gain, question

        return best_gain, best_question

    

    def build_tree(rows):

        gain, question = DecisionTreeClassifier.find_best_split(rows)



        if gain == 0:

            return Leaf(rows)



        # If we reach here, we have found a useful feature / value to partition on.

        true_rows, false_rows = DecisionTreeClassifier.partition(rows, question)



        # Recursively build the true branch.

        true_branch = DecisionTreeClassifier.build_tree(true_rows)



        # Recursively build the false branch.

        false_branch = DecisionTreeClassifier.build_tree(false_rows)



        return Decision_Node(question, true_branch, false_branch)
class Question:

    

    

    # Question Class

    def __init__(self, column, value):

        self.column = column

        self.value = value

        

    def is_numeric(value):

        # To test if a value is numeric.

        return isinstance(value, int) or isinstance(value, float)

    

    def match(self, example):

        # Compare the feature value in an example to the feature value in this question.

        val = example[self.column]

        if Question.is_numeric(val):

            return val >= self.value

        else:

            return val == self.value



    def __repr__(self):

        # This is just a helper method to print the question in a readable format.

        condition = "=="

        if Question.is_numeric(self.value):

            condition = ">="

        return "Is %s %s %s?" % (

            header[self.column], condition, str(self.value))
class Leaf:

    

    

    def __init__(self, rows):

        self.predictions = DecisionTreeClassifier.class_counts(rows)

    

    def print_leaf(counts):

        total = sum(counts.values()) * 1.0

        probs = {}

        for lbl in counts.keys():

            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

        return probs
class Decision_Node:

    

    

    def __init__(self,

                 question,

                 true_branch,

                 false_branch):

        self.question = question

        self.true_branch = true_branch

        self.false_branch = false_branch

        

    def print_tree(node, spacing=""):

        # Base case: we've reached a leaf

        if isinstance(node, Leaf):

            print (spacing + "Predict", node.predictions)

            return



        # Print the question at this node

        print (spacing + str(node.question))



        # Call this function recursively on the true branch

        print (spacing + '--> True:')

        Decision_Node.print_tree(node.true_branch, spacing + "  ")



        # Call this function recursively on the false branch

        print (spacing + '--> False:')

        Decision_Node.print_tree(node.false_branch, spacing + "  ")

        

    def classify(row, node):

        if isinstance(node,Leaf):

            return node.predictions



        if node.question.match(row):

            return Decision_Node.classify(row, node.true_branch)

        else:

            return Decision_Node.classify(row, node.false_branch)

    

    def testing_result(testing_dataset, tree):

        for row in testing_dataset:

            print("Actual: %s. Predicted: %s" %

                (row[-1], Leaf.print_leaf(Decision_Node.classify(row, tree))))
d_tree = DecisionTreeClassifier.build_tree(dataset_array)

print("Tree:\n")

Decision_Node.print_tree(d_tree)
train_set = train_set.to_numpy()

test_set_rows = test_set.to_numpy()



evaluation_tree = DecisionTreeClassifier.build_tree(train_set)
Decision_Node.testing_result(test_set_rows, evaluation_tree)