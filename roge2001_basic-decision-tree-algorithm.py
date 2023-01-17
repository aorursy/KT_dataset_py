# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import math as math

import pandas.api



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file="../input/titanic/train.csv"

# Input file

df = pd.read_csv(file)



# check missing values

print(df.isnull().sum().sort_values(ascending=False))

###Clean data

def clean_data(dataframe):

    # Embarked

    dataframe['Embarked'].fillna(value='S', inplace=True)



    # Cabin

    dataframe['Cabin'].fillna(value='NC', inplace=True)



    # Age

    mean = dataframe['Age'].mean()

    dataframe['Age'].fillna(value=mean, inplace=True)



    # Bin into age groups of 5

    bins = [0, 5, 15, 20, 30, 50, 60, 80]

    group_names = ['baby', 'young', 'teen', 'youngAdult', 'adult', 'olderAdult', 'oldestAdult']

    dataframe['Agegroup'] = pd.cut(dataframe['Age'], bins, labels=group_names)



    return dataframe



c_df=clean_data(df)



print(c_df.isnull().sum().sort_values(ascending=False))

# Remove unneeded data

# passenger id does not matter in the case of training data(it is index+1 if needed)

d_ref = c_df.drop(['Name', 'Age', 'Ticket', 'Fare'], axis=1)[['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Agegroup','PassengerId','Survived']]

d_ref.head()
# Split data into rows

data_rows = []

for index in range(0, d_ref.shape[0]):

    data_rows.append(d_ref.iloc[index])

data_rows[0]
#######The tree and helper methods

# Header for columns

header = ["Pclass","Sex", "SibSp", "Parch", "Cabin", "Embarked", "Agegroup","PassengerId","Survived"]



def class_counts(rows):

    # """Counts the number of each type of example in a dataset."""

    counts = {}  # a dictionary of label -> count.

    for row in rows:

        # in our dataset format, the label is always the last column

        # use survived to isolate and see how much it splits the data

        label = row[-1]

        if label not in counts:

            counts[label] = 0

        counts[label] += 1

    return counts





def is_numeric(value):

    # """Test if a value is numeric."""

    return isinstance(value, int) or isinstance(value, float)





# class Question

class Question:



    def __init__(self, column, value):

        # assign back to object 

        self.column = column

        self.value = value



    def match(self, example):

        # compares value to example

        val = example[self.column]

        # if it is numberic or string

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

        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))





def partition(rows, question):



    true_rows, false_rows = [], []

    for row in rows:

        #HERE

        if question.match(row):

            #print("true")

            true_rows.append(row)

        else:

            false_rows.append(row)

            #print("false")

    return true_rows, false_rows





def gini(rows):

    # """Calculate the Gini Impurity for a list of rows.

    # There are a few different ways to do this, I thought this one was

    # the most concise. See:

    # https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    # """

    counts = class_counts(rows)

    impurity = 1

    for lbl in counts:

        prob_of_lbl = counts[lbl] / float(len(rows))

        impurity -= prob_of_lbl ** 2

    return impurity





def info_gain(left, right, current_uncertainty):

    # """Information Gain.

    # The uncertainty of the starting node, minus the weighted impurity of

    # two child nodes.

    # """

    p = float(len(left)) / (len(left) + len(right))

    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)





def find_best_split(rows):

    best_gain = 0

    best_question = None

    current_uncertainty = gini(rows)

    n_features = len(rows[0]) - 2  # number of columns



    for col in range(n_features):



        values = set([row[col] for row in rows])  # unique values in column



        for val in values:



            question = Question(col, val)



            # try splitting the dataset

            true_rows, false_rows = partition(rows, question)



            # skip this if it doesnt divide the dataset

            if len(true_rows) == 0 or len(false_rows) == 0:

                continue



            # calculate the gain from this split

            gain = info_gain(true_rows, false_rows, current_uncertainty)



            if gain >= best_gain:

                best_gain, best_question = gain, question



    return best_gain, best_question





class Leaf:

    # """A Leaf node classifies data.

    # This holds a dictionary of class (e.g., "Apple") -> number of times

    # it appears in the rows from the training data that reach this leaf.

    # """



    def __init__(self, rows):

        self.predictions = class_counts(rows)

        # predictions is a dictionary of 1 and 0s





class Decision_Node:

    #     """A Decision Node asks a question.

    #     This holds a reference to the question, and to the two child nodes.

    #     """



    def __init__(self, question, true_branch, false_branch):

        self.question = question

        self.true_branch = true_branch

        self.false_branch = false_branch





def build_tree(rows):



    info, question = find_best_split(rows)



    if info == 0:

        return Leaf(rows)





    true_rows, false_rows = partition(rows, question)



    true_branch = build_tree(true_rows)



    false_branch = build_tree(false_rows)



    return Decision_Node(question, true_branch, false_branch)





def classify(row, node):

    if isinstance(node, Leaf):

        return node.predictions



    if node.question.match(row):

        return classify(row, node.true_branch)

    else:

        return classify(row, node.false_branch)







def print_leaf(counts):

    """A nicer way to print the predictions at a leaf."""

    total = sum(counts.values()) * 1.0

    probs = {}

    for lbl in counts.keys():

        probs[lbl] = int(counts[lbl] / total * 100)

    return probs





def print_tree(node, spacing=""):

    """World's most elegant tree printing function."""



    # Base case: we've reached a leaf

    if isinstance(node, Leaf):

        print(spacing + "Predict", node.predictions)

        return



    # Print the question at this node

    print(spacing + str(node.question))



    # Call this function recursively on the true branch

    print(spacing + '--> True:')

    print_tree(node.true_branch, spacing + "  ")



    # Call this function recursively on the false branch

    print(spacing + '--> False:')

    print_tree(node.false_branch, spacing + "  ")



# unique values

def unique_vals(rows, col):

    """Find the unique values for a column in a dataset."""

    return set([row[col] for row in rows])
#Build tree

root=build_tree(data_rows)
def test(test_data,root):

    n_df=pd.DataFrame(columns=['PassengerId','Survived'])



    test_rows = []

    for index in range(0,test_data.shape[0]):

        test_rows.append(test_data.iloc[index])



    for index in range(0,test_data.shape[0]):

        passId=test_rows[index]['PassengerId']

        surv_count=print_leaf(classify(test_rows[index],root))

        #print(surv_count)

        for lbl in surv_count:

            if (surv_count[lbl]>=50):

                surv=lbl

                #print(lbl)

                break

        n_df=n_df.append({'PassengerId':passId,'Survived':surv},ignore_index=True)



    return n_df
test_data=clean_data(pd.read_csv("../input/titanic/test.csv")).drop(['Name', 'Age', 'Ticket', 'Fare'], axis=1)[['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Agegroup','PassengerId']]



predict_df=test(test_data,root)



predict_df.to_csv('predictions.csv',index=False,encoding='utf-8')
