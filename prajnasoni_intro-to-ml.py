# Import the libraries we will be using



import os

import numpy as np

import pandas as pd

import math

import matplotlib.patches as patches

import matplotlib.pylab as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn import tree

from sklearn import metrics

from sklearn import datasets

from sklearn.model_selection import train_test_split

from IPython.display import Image

from sklearn.tree import export_graphviz





%matplotlib inline



#-----------------------------------------------------------------------------------



# A function that picks a color for an instance, depending on its target variable

# We use 0 for "No" and "1" for yes.

# The function returns a list of items, one item for each instance (in the order given)

def Color_Data_Points(target):

    color = ["red" if t == 0 else "blue" for t in target]

    return color





# A function to plot the data in a scatter plot

# data: The data we want to visualize

# v1: The name used to access the X-axis variable information in the data parameter

# v2: The name used to access the Y-axis variable information in the data parameter

# tv: The name used to access the target variable information in the data parameter

def Plot_Data(data, v1, v2, tv):



    # Make the plot square

    plt.rcParams['figure.figsize'] = [12.0, 8.0]

    

    # Color

    color = Color_Data_Points(data[tv])

    

    # Plot and label

    plt.scatter(data[v1], data[v2], c=color, s=50)

    plt.xlabel(v1)

    plt.ylabel(v2)

    plt.xlim([min(data[v1]) , max(data[v1]) ])

    plt.ylim([min(data[v2]) , max(data[v2]) ])



#-----------------------------------------------------------------------------------

    

# Set the randomness

np.random.seed(36)



# Number of users, i.e. number of instances in our dataset

n_users = 100



# Features that we know about each user. The attributes below are for illustration purposes only!

variable_names = ["name", "age", "years_neighbour"]

variables_keep = ["years_neighbour", "age"]

target_name = "response"



# Generate data with the "datasets" function from SKLEARN (package)

# This function returns two variables: predictors and target



predictors, target = datasets.make_classification(n_features=3, n_redundant=0, 

                                                  n_informative=2, n_clusters_per_class=2,

                                                  n_samples=n_users)



# We will write this data in a dataframe (pandas package)



data = pd.DataFrame(predictors, columns=variable_names)



# We want to take each column of the dataframe to change the values 



data['age'] = data['age'] * 10 + 50

data['years_neighbour'] = (data['years_neighbour'] + 6)/2

data[target_name] = target



# Our variables (features) will be stored in one variable called X

X = data[variables_keep]



# Our target will be stored in one variable called Y

Y = data[target_name]



# Show the first 5 values of our data

pd.concat([X, Y], axis=1).head(5)



plt.figure(figsize=[7,6])

Plot_Data(data, "years_neighbour", "age", "response")
plt.rcParams['figure.figsize'] = [15.0, 2.0]



color = color = Color_Data_Points(data["response"])

plt.scatter(X['age'], Y, c=color, s=50)

plt.xlabel('age')
plt.rcParams['figure.figsize'] = [15.0, 2.0]



color = color = Color_Data_Points(data["response"])

plt.scatter(X['years_neighbour'], Y, c=color, s=50)

plt.xlabel('years_neighbour')
# A function that creates the surface of a decision tree



def Decision_Surface(data, target, model):

    # Get bounds

    x_min, x_max = data[data.columns[0]].min(), data[data.columns[0]].max()

    y_min, y_max = data[data.columns[1]].min(), data[data.columns[1]].max()



    # Create a mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max,0.01), np.arange(y_min, y_max,0.01))

    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])



    plt.figure(figsize=[12,7])

    Z = model.predict(meshed_data).reshape(xx.shape)



    plt.title("Decision surface")    

    plt.ylabel("age")

    plt.xlabel("years_customer")



    color = Color_Data_Points(target)

    cs = plt.contourf(xx, yy, Z, levels=[-1,0,1], colors=['#ff6666', '#66b2ff'] )

    plt.scatter(data[data.columns[0]], data[data.columns[1]], color=color, edgecolor='black' )

# Let's define the model (tree)

my_tree = DecisionTreeClassifier(max_depth=5,criterion="entropy") 

#entropy is one measure of determining where to split



# Let's tell the model what is the data

my_tree.fit(X, Y)



#Let's print an image with the results

Decision_Surface(X,Y,my_tree)
print ( "Accuracy = %.3f" % (metrics.accuracy_score(my_tree.predict(X), Y)) )
tree.export_graphviz(my_tree, out_file = 'tree.dot', feature_names = variables_keep, class_names = ['no', 'yes'], filled = True)

# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in python

import matplotlib.pyplot as plt

plt.figure(figsize = (14, 18))

plt.imshow(plt.imread('tree.png'))

plt.axis('off');

plt.show();
zoo_class = pd.read_csv("../input/zoo-animal-classification/class.csv")

zoo = pd.read_csv("../input/zoo-animal-classification/zoo.csv")

zoo.head()
#PREPARING THE DATASET

# Here, we split the dataset into the characteristics/features (X) and the class (Y)

X = zoo.drop('class_type', axis=1)

X = X.drop('animal_name',  axis=1)

Y = zoo['class_type']



# We then split the data into a training (80%) and testing (20%) dataset

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.20, random_state=262, stratify=Y)



#Let's look at the first 5 rows of our train_X data set. 

#This is the dataset with characteristics/features that we will be training on

train_X.head()

#declare decision tree classifier classifiying based on entropy 

zkTree = DecisionTreeClassifier(max_depth = 3, criterion ='entropy')

#train decision tree classifier on training data

zkTree.fit(train_X, train_Y)

#get predicted results for given test_X

zkTreePred = zkTree.predict(test_X)

acc_score = metrics.accuracy_score(test_Y, zkTreePred)

print(acc_score)

metrics.confusion_matrix(test_Y, zkTreePred)
#Let's visualise the tree

tree.export_graphviz(zkTree, out_file = 'zkTree.dot', feature_names = X.columns, class_names = zoo_class['Class_Type'], filled = True)

# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'zkTree.dot', '-o', 'zkTree.png', '-Gdpi=600'])



# Display in python

import matplotlib.pyplot as plt

plt.figure(figsize = (14, 18))

plt.imshow(plt.imread('zkTree.png'))

plt.axis('off');

plt.show();