import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/Data.txt')
sns.pairplot(data)
scplot = sns.scatterplot(data.X,data.y,hue=data.res,palette=['orange','blue'])

scplot.plot([2.937,2.937], [8,0], 'r-', linewidth = 2)

scplot.plot([0,8],[4.55,4.55],'g-',linewidth=2)

scplot.plot([4.79,4.79],[0,8],'y-',linewidth=2)
data.res.value_counts()
# Using readlines() 

file1 = open('/kaggle/input/Data.txt', 'r') 

Lines = file1.readlines() 
li_data = []

for line in Lines[1:]:

    li1 = [x for x in line.split(',')]

    li1[2] = li1[2][0]

    li1 = [float(x) for x in li1]

    li_data.append(li1)
len(li_data)

train_data = li_data[:220]

test_data = li_data[220:]
# Column labels.

# These are used only to print the tree.

header = ['X','y','classify']
def unique_vals(rows, col):

    """Find the unique values for a column in a dataset."""

    return set([row[col] for row in rows])
unique_vals(train_data, 2)
def class_counts(rows):

    """Counts the number of each type of example in a dataset."""

    counts = {}  # a dictionary of label -> count.

    for row in rows:

        # in our dataset format, the label is always the last column

        label = row[-1]

        if label not in counts:

            counts[label] = 0

        counts[label] += 1

    return counts
#######

# Demo:

class_counts(train_data)

class_counts(test_data)

#######
def is_numeric(value):

    """Test if a value is numeric."""

    return isinstance(value, int) or isinstance(value, float)

#######

# Demo:

is_numeric(7.1)

# is_numeric("Red")

#######


class Question:

    """A Question is used to partition a dataset.



    This class just records a 'column number' (e.g., 0 for Color) and a

    'column value' (e.g., Green). The 'match' method is used to compare

    the feature value in an example to the feature value stored in the

    question. See the demo below.

    """



    def __init__(self, column, value):

        self.column = column

        self.value = value



    def match(self, example):

        # Compare the feature value in an example to the

        # feature value in this question.

        val = example[self.column]

        if is_numeric(val):

            return val >= self.value

        else:

            return val == self.value



    def __repr__(self):

        # This is just a helper method to print

        # the question in a readable format.

        condition = "=="

        if is_numeric(self.value):

            condition = ">="

        return "Is %s %s %s?" % (

            header[self.column], condition, str(self.value))
q1 = Question(0,2.9377)

q1
q2 = Question(1,4.55)

q2
q3 = Question(0,4.79)

q3
def partition(rows, question):

    """Partitions a dataset.



    For each row in the dataset, check if it matches the question. If

    so, add it to 'true rows', otherwise, add it to 'false rows'.

    """

    true_rows, false_rows = [], []

    for row in rows:

        if question.match(row):

            true_rows.append(row)

        else:

            false_rows.append(row)

    return true_rows, false_rows
#######

# Demo:

# Let's partition the training data based on whether rows are Red.

true_rows, false_rows = partition(train_data, q1)

# This will contain all the 'Red' rows.

true_rows
# This will contain everything else.

false_rows

#######
def gini(rows):

    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was

    the most concise. See:

    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    """

    counts = class_counts(rows)

    impurity = 1

    for lbl in counts:

        prob_of_lbl = counts[lbl] / float(len(rows))

        impurity -= prob_of_lbl**2

    return impurity
# this will return 0.5 - meaning, there's a 50% chance of misclassifying

gini(train_data)
#this return 0 as the test data has no impurity, thus only one label is present there 

gini(test_data)
igX = []

igY = [] 

ig = []
def info_gain(left, right, current_uncertainty):

    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of

    two child nodes.

    """

    p = float(len(left)) / (len(left) + len(right))

    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
# Demo:

# Calculate the uncertainy of our training data.

current_uncertainty = gini(train_data)

current_uncertainty
# How much information do we gain by partioning on q1?

true_rows, false_rows = partition(train_data, q1)

info_gain(true_rows, false_rows, current_uncertainty)
def find_best_split(rows):

    """Find the best question to ask by iterating over every feature / value

    and calculating the information gain."""

    best_gain = 0  # keep track of the best information gain

    best_question = None  # keep train of the feature / value that produced it

    current_uncertainty = gini(rows)

    n_features = len(rows[0]) - 1  # number of columns



    for col in range(n_features):  # for each feature



        values = set([row[col] for row in rows])  # unique values in the column



        for val in values:  # for each value



            question = Question(col, val)



            # try splitting the dataset

            true_rows, false_rows = partition(rows, question)



            # Skip this split if it doesn't divide the

            # dataset.

            if len(true_rows) == 0 or len(false_rows) == 0:

                continue



            # Calculate the information gain from this split

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            ig.append(gain)

            # You actually can use '>' instead of '>=' here

            # but I wanted the tree to look a certain way for our

            # toy dataset.

            if gain >= best_gain:

                best_gain, best_question = gain, question



    return best_gain, best_question


#######

# Demo:

# Find the best question to ask first for our toy dataset.

best_gain, best_question = find_best_split(train_data)

best_question

# FYI: is color == Red is just as good. See the note in the code above

# where I used '>='.

#######
#now i want to check the info gain for the best first question

true_rows, false_rows = partition(li_data, best_question)

info_gain(true_rows, false_rows, current_uncertainty)

class Leaf:

    """A Leaf node classifies data.



    This holds a dictionary of class (e.g., "Apple") -> number of times

    it appears in the rows from the training data that reach this leaf.

    """



    def __init__(self, rows):

        self.predictions = class_counts(rows)


class Decision_Node:

    """A Decision Node asks a question.



    This holds a reference to the question, and to the two child nodes.

    """



    def __init__(self,

                 question,

                 true_branch,

                 false_branch):

        self.question = question

        self.true_branch = true_branch

        self.false_branch = false_branch
c=0


def build_tree(rows):

    """Builds the tree.



    Rules of recursion: 1) Believe that it works. 2) Start by checking

    for the base case (no further information gain). 3) Prepare for

    giant stack traces.

    """



    # Try partitioing the dataset on each of the unique attribute,

    # calculate the information gain,

    # and return the question that produces the highest gain.

    gain, question = find_best_split(rows)



    # Base case: no further info gain

    # Since we can ask no further questions,

    # we'll return a leaf.

    if gain == 0:

        return Leaf(rows)

    print(question)

    # If we reach here, we have found a useful feature / value

    # to partition on.

    true_rows, false_rows = partition(rows, question)



    # Recursively build the true branch.

    true_branch = build_tree(true_rows)



    # Recursively build the false branch.

    false_branch = build_tree(false_rows)



    # Return a Question node.

    # This records the best feature / value to ask at this point,

    # as well as the branches to follow

    # dependingo on the answer.

    return Decision_Node(question, true_branch, false_branch)
"""#now we want to develop a tree with the tree questions 



#for q1

t1,f1 = partition(train_data,q1)

gini(t1)

gini(f1) #0



#this means i have to split only tb1

if(gini(t1) == 0)

"""



def print_tree(node, spacing=""):

    """World's most elegant tree printing function."""



    # Base case: we've reached a leaf

    if isinstance(node, Leaf):

        print (spacing + "Predict", node.predictions)

        return



    # Print the question at this node

    print (spacing + str(node.question))



    # Call this function recursively on the true branch

    print (spacing + '--> True:')

    print_tree(node.true_branch, spacing + "  ")



    # Call this function recursively on the false branch

    print (spacing + '--> False:')

    print_tree(node.false_branch, spacing + "  ")
my_tree = build_tree(train_data)
len(ig)

train_data

X = []

y = []

for elem in train_data:

  X.append(elem[0])

  y.append(elem[1])

len(X)

sns.distplot(ig)
print_tree(my_tree)

def classify(row, node):

    """See the 'rules of recursion' above."""



    # Base case: we've reached a leaf

    if isinstance(node, Leaf):

        return node.predictions



    # Decide whether to follow the true-branch or the false-branch.

    # Compare the feature / value stored in the node,

    # to the example we're considering.

    if node.question.match(row):

        return classify(row, node.true_branch)

    else:

        return classify(row, node.false_branch)


#######

# Demo:

# The tree predicts the 1st row of our

# training data is an apple with confidence 1.

classify(train_data[0], my_tree)

#######
def print_leaf(counts):

    """A nicer way to print the predictions at a leaf."""

    total = sum(counts.values()) * 1.0

    probs = {}

    for lbl in counts.keys():

        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

    return probs


#######

# Demo:

# Printing that a bit nicer

print_leaf(classify(li_data[0], my_tree))

#######
true = 0

false = 0
for row in test_data:

  actual = row[-1]

  pred = print_leaf(classify(row, my_tree))

  if actual == list(pred)[0]: 

    true+=1

  else:

    false+=1

  print ("Actual: %s. Predicted: %s" % (actual, pred))

true,false

accuracy = (true/(true+false))*100
print("the accuracy of our model from scratch is : ", accuracy)
X = data.drop('res',axis=1)

y = data['res']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3,criterion='gini')

clf.fit(X_train,y_train)

print("the accuracy with model from sklearn : ", clf.score(X_test,y_test))



from sklearn.tree import plot_tree

plot_tree(clf)