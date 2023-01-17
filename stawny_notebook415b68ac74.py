# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import time

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
full = pd.read_csv('../input/spam.csv', encoding='latin-1')

full.head()
full[ 'Winner' ] = full[ 'v2' ].map( lambda text : True if re.search("winner|win|won|award|selected", text, re.IGNORECASE) else False)

full[ 'Free' ] = full[ 'v2' ].map( lambda text : True if re.search("free\W", text, re.IGNORECASE) else False)

full[ 'Free2' ] = full[ 'v2' ].map( lambda text : True if re.search("you\ free|feel\ free|isnt\ free|not\ free|yr\ free", text, re.IGNORECASE) else False)

full[ 'Congratulation' ] = full[ 'v2' ].map( lambda text : True if re.search("congratulations|congrats", text, re.IGNORECASE) else False)

full[ 'Adult' ] = full[ 'v2' ].map( lambda text : True if re.search("xxx|babe|naked|dirty|flirty", text, re.IGNORECASE) else False)

full[ 'Attention' ] = full[ 'v2' ].map( lambda text : True if re.search("urgent|attention|bonus|immediately", text, re.IGNORECASE) else False)

full[ 'Ringtone' ] = full[ 'v2' ].map( lambda text : True if re.search("ringtone", text, re.IGNORECASE) else False)

full[ 'Spam' ] = full[ 'v1' ].map(lambda text : text == 'spam')

del full['Unnamed: 2']

del full['Unnamed: 3']

del full['Unnamed: 4']

del full[ 'v1' ]

del full[ 'v2' ]

#full = full.loc[ ~full[ 'Spam' ]].loc[ full[ 'Free' ]]

#full[['v2']]

full.head()
def class_counts(rows):

    counts = {}  # a dictionary of label -> count.

    #counts[True] = 0

    for index, row in rows.iterrows():

        label = row[ 'Spam' ]

        if label not in counts:

            counts[label] = 0

        counts[label] += 1

    return counts



def gini(rows):

    counts = class_counts(rows)

    impurity = 1

    for lbl in counts:

        prob_of_lbl = counts[lbl] / float(len(rows))

        impurity -= prob_of_lbl**2

    return impurity



def info_gain(left, right, current_uncertainty):

    p = float(len(left)) / (len(left) + len(right))

    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)



def find_best_split(rows):

    best_gain = 0  # keep track of the best information gain

    best_question = None  # keep train of the feature / value that produced it

    current_uncertainty = gini(rows)



    for col in rows:  # for each feature

        if col == 'Spam':

            continue

        values = rows[col].unique()#set([row[col] for row in rows])

        

        for val in values: 

            # try splitting the dataset

            true_rows = full.loc[ full[ col ] ]

            false_rows = full.loc[ ~full[ col ] ]



            # Skip this split if it doesn't divide the

            # dataset.

            if len(true_rows) == 0 or len(false_rows) == 0:

                continue



            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:

                best_gain, best_question = gain, col



    return best_gain, best_question



class Leaf:

    def __init__(self, rows):

        self.predictions = class_counts(rows)

class Decision_Node:

    def __init__(self,

                 question,

                 true_branch,

                 false_branch):

        self.question = question

        self.true_branch = true_branch

        self.false_branch = false_branch



def build_tree(rows):

    gain, question = find_best_split(rows)

    #print(len(rows), " ", question)

    if gain == 0:

        return Leaf(rows)

    true_branch = build_tree( rows.loc[ rows[ question ] ].drop(question, axis = 1) )

    false_branch = build_tree( rows.loc[ ~rows[ question ] ].drop(question, axis = 1) )

    if isinstance(true_branch, Leaf) and len(true_branch.predictions) == 0:

        return false_branch

    return Decision_Node(question, true_branch, false_branch)



def print_tree(node, spacing=""):

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



def classify(row, node):

    if isinstance(node, Leaf):

        return node.predictions



    if row[node.question]:

        return classify(row, node.true_branch)

    else:

        return classify(row, node.false_branch)

    

def test(rows, node):

    total = 0

    correct = 0

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for index, row in rows.iterrows():

        c = classify(row, my_tree)

        if max(c, key=c.get) and row['Spam']:

            tp += 1

        if ~max(c, key=c.get) and row['Spam']:

            fn += 1

        if max(c, key=c.get) and ~row['Spam']:

            fp += 1

        if ~max(c, key=c.get) and ~row['Spam']:

            tn += 1

        total += 1

    return (tp + tn) / total







test_data = full[ :1671]

time_data = []

error_rate = []

for x in range(1, 8):

    training = full[1671 : int(1671 + 5572 * x / 10)]

    training = training.sample(frac=1).reset_index(drop=True)

    start_time = time.time()

    my_tree = build_tree(training)

    

    #print_tree(my_tree)

    #print("--- %s seconds ---" % (time.time() - start_time), test(test_data, my_tree))

    time_data.append(time.time() - start_time)

    error_rate.append(1 - test(test_data, my_tree))

plt.plot(time_data)

plt.show()
plt.plot(error_rate)

plt.show()
test(test_data, my_tree)
print_tree(my_tree)


def test2(rows, node):

    total = 0

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for index, row in rows.iterrows():

        c = classify(row, my_tree)

        if max(c, key=c.get) and row['Spam']:

            tp += 1

        if ~max(c, key=c.get) and row['Spam']:

            fn += 1

        if max(c, key=c.get) and ~row['Spam']:

            fp += 1

        if ~max(c, key=c.get) and ~row['Spam']:

            tn += 1

        total += 1

    print("tp: ", tp / total, "\tfp: ", fp / total)

    print("fn: ", fn / total, "\ttn: ", tn / total)

test2(test_data, my_tree)