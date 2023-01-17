# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
import graphviz
# import category encoders
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
import copy
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_car= pd.read_csv("/kaggle/input/dataset-car/data_car.csv")
training_data = data_car.iloc[:864,:].values

# Column labels.
# These are used only to print the tree.
header = data_car.columns

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])



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

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


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




def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)




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

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

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
# my_tree = build_tree(training_data)
# classify(training_data[0], my_tree)
#######

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def print_class(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    classe = {}
    for lbl in counts.keys():
        classe[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return classe


if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    # Evaluate
    testing_data =data_car.iloc[864:,:].values
    acerto=0
    for row in testing_data:
        #print ("Actual: %s. Predicted: %s" %
               if(row[-1], print_class(classify(row, my_tree))):
                   acerto=acerto+1

    print ("taxa de acerto:", acerto/len(testing_data))
    
def k_fold_cross_validation(num_split,nun_repeat,data_car):
    
    kf=RepeatedStratifiedKFold(n_splits=num_split, n_repeats=nun_repeat,random_state=42)
    x,y=data_car.iloc[:,:],data_car.iloc[:,:1]
    kf.get_n_splits()
    k=0
    
    for train_index, test_index in kf.split(x,y):

        
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        c=pd.concat([x_train,y_train])
        print(c.iloc[:,:].values)
        # Evaluate
        #my_tree = build_tree(c.iloc[:,:].values)
        #testing_data =pd.concat([x_test,y_test])
        acerto=0
        
        
        
        
k_fold_cross_validation(10,5,data_car)
def kfoldcv(indices, k):
    
    size = len(indices)
    subset_size = round(size / k)
    
    subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
    
    k_fold_classif=[]
  
    
    for i in range(len(subsets)):
        
            data_test=car_data.iloc[subsets[i]]
            x_test=data_test.iloc[:,:6]
            y_test=data_test['class   ']
            
            #excluir subset from subsets
            new_subsets = subsets.copy()
            del new_subsets[i]
            data_train=[]
            
            for j in range(len(new_subsets)):
                data_train.append(car_data.iloc[new_subsets[j]])
                
            data_train=pd.concat([data_train[0],data_train[1],data_train[2],data_train[3],
                                    data_train[4],data_train[5],data_train[6],data_train[7],
                                    data_train[8]])
             
            x_train=data_train.iloc[:,:6]
            y_train=data_train['class   ']
            
            # encode variables with ordinal encoding
            encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
            x_train = encoder.fit_transform(x_train)
            x_test = encoder.transform(x_test)
            
            #Classification
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(x_train,y_train)
            k_fold_classif.append(clf.score(x_test, y_test))
    
                   
    return k_fold_classif
subsets=kfoldcv(car_data.index, 10)
subsets


def train_test_set(car_data):
    #Slipt test and train set
    x=car_data.iloc[:,:6]
    y=car_data['class   ']
    return train_test_split(x, y,test_size=0.5,random_state=1)

def encode(x_train, x_test):
    # encode variables with ordinal encoding
    encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    x_train = encoder.fit_transform(x_train)
    x_test = encoder.transform(x_test)
    return x_train, x_test

def holdout(x_train, x_test, y_train, y_test ):
    
    lista=[]
    treeClassifier=[]
    for k in range(1,4):
        
        #Classification
        clf = tree.DecisionTreeClassifier(min_samples_leaf=k*10)
        clf = clf.fit(x_train,y_train)
        treeClassifier.append(clf)
        
        #Visualize decision-trees    
        dot_data = tree.export_graphviz(clf, out_file=None, 
        feature_names=x_train.columns,  
        class_names=y_train.values,  
        filled=True, rounded=True,  
        special_characters=True)  
        graph = graphviz.Source(dot_data) 
        
        lista.append(graph)
    return lista,treeClassifier

x_train, x_test, y_train, y_test =train_test_set(car_data)
x_train, x_test=encode(x_train, x_test)
source,treeClassifier=holdout(x_train, x_test, y_train, y_test )
source[0].render("car_1")
source[1].render("car_2")
source[2].render("car_3")
def show_trainig_score(treeClassifier,x_train, y_train):
    return { 'Training set score min_samples_leaf=10: {:.4f}'.format(treeClassifier[0].score(x_train, y_train)),
            'Training set score min_samples_leaf=20: {:.4f}'.format(treeClassifier[1].score(x_train, y_train)),
             'Training set score min_samples_leaf=30: {:.4f}'.format(treeClassifier[2].score(x_train, y_train))
            }
             

show_trainig_score(treeClassifier,x_train, y_train)

def show_testing_score(treeClassifier,x_test, y_test):
    return { 'Testing set score min_samples_leaf=10: {:.4f}'.format(treeClassifier[0].score(x_test, y_test)),
            'Testing set score min_samples_leaf=20: {:.4f}'.format(treeClassifier[1].score(x_test, y_test)),
             'Testing set score min_samples_leaf=30: {:.4f}'.format(treeClassifier[2].score(x_test, y_test))
            }
             

show_testing_score(treeClassifier,x_test, y_test)


def plot_confusion_matrix_by_decision_tree(y_pred,y_train,tittle,labels):

    cm=confusion_matrix(y_train, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(tittle)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
labels = ['unacc', 'acc','good','v-good']
#Número mínimo de folhas igual a 10
plot_confusion_matrix_by_decision_tree(treeClassifier[0].predict(x_test),y_test,'Confusion matrix of min_samples_leaf=10',labels)
#Número mínimo de folhas igual a 20
plot_confusion_matrix_by_decision_tree(treeClassifier[1].predict(x_test),y_test,'Confusion matrix of min_samples_leaf=20',labels)
#Número mínimo de folhas igual a 30
plot_confusion_matrix_by_decision_tree(treeClassifier[2].predict(x_test),y_test,'Confusion matrix of min_samples_leaf=30',labels)
def show_number_of_leaves(treeClassifier):
    return  { 'Number of leaves ':treeClassifier[0].get_n_leaves(),
            'Number of leaves min_samples_leaf=20':treeClassifier[1].get_n_leaves(),
             'Number of leaves min_samples_leaf=30':treeClassifier[2].get_n_leaves()
            }
             

show_number_of_leaves(treeClassifier)
def show_number_of_nodes(treeClassifier):
    return  { 'Number of nodes min_samples_leaf=10':treeClassifier[0].tree_.node_count,
            'Number of nodes min_samples_leaf=20':treeClassifier[1].tree_.node_count,
             'Number of nodes min_samples_leaf=30':treeClassifier[2].tree_.node_count
            }
             

show_number_of_nodes(treeClassifier)
#importando dataset
cancer_data= pd.read_csv("/kaggle/input/brest-cancer/breast-cancer.data")
cancer_data.columns=['class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig',
                     'breast','breast-quad','irradiat']
def train_test_set(cancer_data):
    #Slipt test and train set
    x=cancer_data.iloc[:,1:]
    y=cancer_data['class']
    return train_test_split(x, y,test_size=0.5,random_state=1)

def encode(x_train, x_test):
    # encode variables with ordinal encoding
    encoder = ce.OrdinalEncoder(cols=['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig',
                                     'breast','breast-quad','irradiat'])
    x_train = encoder.fit_transform(x_train)
    x_test = encoder.transform(x_test)
    return x_train, x_test
x_train_cancer, x_test_cancer, y_train_cancer, y_test_cancer =train_test_set(cancer_data)
x_train_cancer, x_test_cancer=encode(x_train_cancer, x_test_cancer)
source,cancertreeClassifier=holdout(x_train_cancer, x_test_cancer, y_train_cancer, y_test_cancer )
source[0].render("cancer_1")
source[1].render("cancer_2")
source[2].render("cancer_3")

show_trainig_score(cancertreeClassifier,x_train_cancer, y_train_cancer)

show_testing_score(cancertreeClassifier,x_test_cancer, y_test_cancer)

#Número mínimo de folhas igual a 10
labels=['no-recurrence-events', 'recurrence-events']
plot_confusion_matrix_by_decision_tree(cancertreeClassifier[0].predict(x_test_cancer),y_test_cancer,'Confusion matrix of min_samples_leaf=10',labels)

#Número mínimo de folhas igual a 20

plot_confusion_matrix_by_decision_tree(cancertreeClassifier[1].predict(x_test_cancer),y_test_cancer,'Confusion matrix of min_samples_leaf=20',labels)



#Número mínimo de folhas igual a 30
plot_confusion_matrix_by_decision_tree(cancertreeClassifier[2].predict(x_test_cancer),y_test_cancer,'Confusion matrix of min_samples_leaf=30',labels)
show_number_of_leaves(cancertreeClassifier)

show_number_of_nodes(cancertreeClassifier)
