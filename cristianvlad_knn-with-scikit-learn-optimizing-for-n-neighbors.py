# Based on/Adapted from Introduction to Machine Learning by Andreas C. Muller and Sarah Guido 

# Credits to Andreas C. Muller and Sarah Guido



# Making the necessary imports



from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt



%matplotlib inline
# Loading the dataset. I suspect many don't know that this is included in the 'default' 

# scikit-learn datasets



cancer = load_breast_cancer()



# print(cancer.DESCR) - A clean/concise description of the data



# Using the KNN Model



# First, we split the data into training and testing

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 

                                                    stratify=cancer.target, random_state=66)



# We create two lists to keep training and test accuracies. We'll later use them to evaluate an 

# appropriate number of neighbors

training_accuracy = []

test_accuracy = []



# We define a range of 1 to 10 (included) neighbors that will be tested

neighbors_settings = range(1,11)



# We loop the KNN model through the range of possible neighbors to evaluate which one would be 

# appropriate for this analysis



for n_neighbors in neighbors_settings:

    

    # creating the KNN classifier

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    # fitting the model

    clf.fit(X_train, y_train)

    #recording the accuracy of the training set

    training_accuracy.append(clf.score(X_train, y_train))

    #recording the accuracy of the test set

    test_accuracy.append(clf.score(X_test, y_test))

    

# Data Visualization - Evaluating the accuracy of both the training and the testing sets against 

# n_neighbors

    

plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the Training Set')

plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the Test Set')

plt.ylabel('Accuracy')

plt.xlabel('Number of Neighbors')

plt.legend()