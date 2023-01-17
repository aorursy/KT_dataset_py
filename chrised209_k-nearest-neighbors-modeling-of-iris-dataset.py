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
# Import the needed matplotlib functionality for scatter plot visualization.

import matplotlib.pyplot as plt

# import the needed dataset.

from sklearn.datasets import load_iris

# Import the model.

from sklearn.neighbors import KNeighborsClassifier

# Import the confusion matrix tool

from sklearn.metrics import confusion_matrix

# Import the train-test split functionality

from sklearn.model_selection import train_test_split

# Import the unique_labels function to support plotting of the confusion matrix

from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax





np.set_printoptions(precision=2)
# Load the iris dataset from scikit-learn (note the use of from [library] import [function] above)

iris = load_iris()



# Define X values from the measurements.

X = iris.data

# Define Y values from the classification indices. 

y = iris.target

# Define the classifications of each sample.

class_names = iris.target_names



# Split the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3333, random_state=0)
# Define the classifier model object...

classifier = KNeighborsClassifier()



# ... to fit the decision tree classifier model on the training data only.



y_pred = classifier.fit(X_train, y_train).predict(X_test)



# Set the size of the figure used to contain the confusion matrices to be generated.

plt.figure(figsize=(15,15))



# Plot non-normalized confusion matrix comparing the predicted y_pred labels to the actual y_test values

plot_confusion_matrix(y_test, y_pred, classes=class_names,

                      title='Confusion Matrix Without Normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,

                      title='Normalized Confusion Matrix')



plt.show()

   
# Define a list of numbers of neighbors to consdider

number_of_neighbors = [7,10,12,15,17,20,22,25]



# Initialize a list of classifier models



classifiers = []



# Initalize a list of y_pred values

y_preds = []



# Iterating through each neighbor

for i, neighbor_count in enumerate(number_of_neighbors):



    # Define the classifier model object...

    classifier = KNeighborsClassifier(n_neighbors = neighbor_count)

    classifiers.append(classifier)

    # ... to fit the decision tree classifier model on the training data only.



    y_pred = classifier.fit(X_train, y_train).predict(X_test)



    # Set the size of the figure used to contain the confusion matrices to be generated.

    plt.figure(figsize=(15,15))



    # Plot non-normalized confusion matrix comparing the predicted y_pred labels to the actual y_test values

    plot_confusion_matrix(y_test, y_pred, classes=class_names,

                      title= f'Confusion Matrix Without Normalization ({neighbor_count} neighbors)')



plt.show()