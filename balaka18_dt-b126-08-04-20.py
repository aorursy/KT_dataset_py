# Creating a toy dataset of types of fruits in the order [Color, Diameter, Label]

import numpy as np

import pandas as pd

# toy_dataset = [['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon'], ['Green', 3, 'Apple'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon']]

toy_dataset = [['Green', 3, 'Apple'], ['Yellow', 3, 'Apple'], ['Red', 1, 'Grape'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon']]

data = np.array(toy_dataset)

print(data)

d = pd.DataFrame({'Color':data[:, 0], 'Diameter':data[:, 1], 'Fruit_label':data[:, 2]})

print("\n",d)



print("\n", len(toy_dataset))
# Assigning labels to each column

head = ['Color', 'Diameter', 'Fruit_label']
def unique_values_in_a_col(rows, col):

    return set([row[col] for row in rows])



print(unique_values_in_a_col(toy_dataset, 1))
# Function to count the frequency of each type of a label in the dataset



def count(datast):

    count = {}  # A dict with key : value as label : freq

    for row in datast:

        label = row[-1]   # Since the above dataset has labels in the last column only

        if label not in count:

            count[label] = 0

        count[label] += 1

    return count



# Example

print(count(toy_dataset))
# Class that asks the question that best splits the dataset



class Ques:

    def __init__(self, column, value):

        self.column = column

        self.value = value

    

    # Function to check if the value passed (ex) is to be split into True or False branch based on the question asked.

    # Example to demonstrate use is given later

    def match(self, ex):

        val = ex[self.column]

        if type(val) == int:

            return val >= self.value     # Because one of the type of questions can be if diameter is >= 3

        else:

            return val == self.value     # Because another type of questions can be if color matches a particular color

    

    # Function to print out the question formulated by values passed in __init__() in a readable format

    def __repr__(self):

        det_cond = "=="           # Partitioning condition in the tree splitting

        if type(self.value) == int:

            det_cond = ">="

        return "Is %s %s %s ?" % (head[self.column], det_cond, str(self.value))

    

# Examples to show printing of splitting questions and the True/False branching [match() method]



qs = Ques(1, 3)

qs
qs.match(toy_dataset[3])      

# self.column = 1. So when we provide ex = toy_dataset[3], val becomes ex[3][1].It is = 3. Hence, True.
''' Function to partition the dataset into True branch end and False branch end. 

                                 _____________

                                |starting_node|   best split question

                                      /   \

                            True     /     \   False

                                    /       \

                                Next node   Next node

                                   or          or

                                Leaf node   Leaf node     ''' 



def partition(rows, ques):

    true, false = [], []

    for row in rows:

        if ques.match(row):

            true.append(row)

        else:

            false.append(row)

    return true, false



# Example of a split

T, F = partition(toy_dataset, Ques(0, 'Red'))

print("True branch :", T,"\n\nFalse branch :", F)
# Calculating the Gini impurity of list of rows



def gini_imp(rows):

    cnts = count(rows)

    impurity = 1

    for labl in cnts:

        p = cnts[labl]/float(len(rows))

        impurity -= p**2

    return impurity



# Example to show calculation of Gini Impurity

print("Initial impurity (label based, before split): ", gini_imp(toy_dataset))
''' Information Gain = uncertainty of starting node - (weighted impurity of the two child nodes)



    Simply, information gain(or just Gain) = G(before split) - sum(weight * G(after split))

    

Calculating the information gain of a split. High Gain = most likely split'''



def info_gain(left_child, right_child, before_split):

    weight = float(len(left_child))/(len(left_child) + len(right_child))

    return before_split - weight*gini_imp(left_child) - (1-weight)*gini_imp(right_child)



true, false = partition(toy_dataset, Ques(0, 'Red'))

info_gain(true, false, gini_imp(toy_dataset))
print(true, "\n\n", false)
''' And finally. The function that does the best splitting by iterating repetitively over all features to see the 

    possible questions that can be asked and asking that question that gives the highest info gain'''



def best_split(rows):

    initial_uncertainty = gini_imp(rows)

    # No. of coulmns

    n = len(rows[0]) - 1

    # Keep track of best information gain

    best_gain = 0

    # Keep track of the question that gave the best information gain

    best_question = None

    

    for column in range(n):

        values = set([row[column] for row in rows])

        for v in values:

            question = Ques(column, v)

            true, false = partition(rows, question)

            

            if len(true) == 0 or len(false) == 0:

                continue

            inf_gain = info_gain(true, false, gini_imp(rows))

            if inf_gain >= best_gain:

                best_gain, best_question = inf_gain, question

    return best_gain, best_question





# Example to find splitting question of starting node of our dataset

print(best_split(toy_dataset))
''' A class to define the leaf nodes of a tree. A leaf node is basically the count of a particular label at a specific row 

    from the training data that satisfies the conditions to be a leaf node.'''



class Leaf:

    def __init__(self, rows):

        self.predictions = count(rows)
''' Now, the class to create a splitting node or the Decsision Node'''



class Dec_node:

    def __init__(self, question, true, false):

        self.question = question

        self.true = true

        self.false = false
''' Finally, the we write the function to build the tree. First we do the start split, to decide the root of the tree. 

    Obviously it is split into the True and False branches. Then, we recursively build the tree on true branch and the false

    branch. This is continued till information gain at the node are = 0. They are assigned as Leaf nodes.'''



def build_Tree(rows):

    # We are finding the first best split question to zero down on the root node.

    i_gain, ques = best_split(rows)

    # Next we see if information gain is zero. If yes, no split occurs, Leaf Node is assigned. (Base condition)

    

    if i_gain == 0:

        return Leaf(rows)

    

    # However, if gain is not zero, we split the dataset into the true and false branches. The nodes at the end of the branches

    # become decision nodes...the function build_Tree is again called recursively on these decision nodes.

  

    true_rows, false_rows = partition(rows, ques)

    

    true_node = build_Tree(true_rows)

    false_node = build_Tree(false_rows)

    

    return Dec_node(ques, true_node, false_node)
''' Now we write the function to print the tree. '''



def print_Tree(node, spacing = ""):

    # First we check the base condition,i.e, if we've reached a Leaf

    if isinstance(node, Leaf):

        print(spacing + "Predict", node.predictions)

        return

    

    # If not a leaf node,

    

    print(spacing + str(node.question))

    

    # The True branch

    print(spacing + "--> True")

    print_Tree(node.true, spacing + " ")

    

    # The False branch

    print(spacing + "--> False")

    print_Tree(node.false, spacing + " ")
this_tree = build_Tree(toy_dataset)

print_Tree(this_tree)
''' Creating the classifier '''



def classify(row, node):

    if isinstance(node, Leaf):

        return node.predictions

    if node.question.match(row):

        return classify(row, node.true)

    else:

        return classify(row, node.false)
classify(toy_dataset[0], this_tree)
def print_l(counts):

    """ Print the predictions at a leaf, in percentage """

    total = sum(counts.values()) * 1.0

    probs = {}

    for lbl in counts.keys():

        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

    return probs



# Example

print_l(classify(toy_dataset[1], this_tree))
# Now we use this self-made decision true on a test sample for prediction



test_data = [['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon'], ['Green', 3, 'Apple'], ['Red', 1, 'Grape'], ['Yellow', 3, 'Lemon']]



for row in test_data:

    print("Actual: %s. Predicted: %s" % (row[-1], print_l(classify(row, this_tree))))