# Let's start off by importing the relevant libraries

import pandas as pd

import numpy as np

import math

import itertools

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix



# Import training and test sets into the scripts

raw_training_df = pd.read_csv("../input/train.csv") # creates a Pandas data frame for training set

raw_test_df  = pd.read_csv("../input/test.csv") # similarly, creates a Pandas data frame for test set



# print(training_df) # Display training data (optional)

# print(test_df) # Display test data (optional)
# Plot a frequency histograms for the classes

count_classes = pd.value_counts(raw_training_df['Survived'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Class Label Histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
# Remove the 'Name' and 'Ticket' columns from the dataframes 

training_df = raw_training_df.drop(['Name', 'Ticket'], axis=1)



# print(training_df.columns) # List column names (Optional)
# Fill in emtpy fields in the 'Cabin' column with the string 'unknown'

training_df['Cabin'] = training_df['Cabin'].fillna('unknown') # Mean age of the passengers happens to be 30.
# Use the mean 'Age' value for empty fields in the 'Age' column

training_df['Age'] = training_df['Age'].fillna(math.ceil(training_df['Age'].mean()))



# training_df.tail(100) # Show last 100 entries (Optional)
'''

    Credit for the following function goes to Chris Strelioff 

    (http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html)

'''

def encode_target(df, target_column):

    df_mod = df.copy()

    targets = df_mod[target_column].unique()

    map_to_int = {name: n for n, name in enumerate(targets)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)



    return (df_mod, targets) # Returns modified dataframe and an array containing the different values encountered in a column



training_df, sex_targets = encode_target(training_df, "Sex")



training_df, embarked_targets = encode_target(training_df, "Embarked")



training_df, cabin_targets = encode_target(training_df, "Cabin")



# training_df.tail(100) # Display the last 100 entries (Optional)
# Assign class features to the variable 'X'

X = training_df.ix[:, training_df.columns != 'Survived']



# Assign class labels to the variable 'y'

y = training_df.ix[:, training_df.columns == 'Survived']
# Create a cross-validation (CV) set from the existing data set

X_train, X_CV, y_train, y_CV = train_test_split(X,y,test_size = 0.3, random_state = 0)
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99) # Declare a decision tree clasifier

dt.fit(X_train, y_train) # Fit the classifier to the training set

y_pred = dt.predict(X_CV) # Predict class labels for the cross-validation set
'''

    Credit for the following confuction matrix function goes to

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

'''

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_CV, y_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0, 1],

                      title='Confusion matrix')



plt.show()
prec = cnf_matrix[1,1] / (cnf_matrix[0,1] + cnf_matrix[1,1])

print("The precision of the ML model is ", round(prec, 3))



recl = cnf_matrix[1,1] / (cnf_matrix[1,0] + cnf_matrix[1,1])

print("The recall of the ML model is ", round(recl, 3))



f1 = 2*((recl*prec)/(recl+prec))

print("The f1-score of the ML model is %f." % round(f1, 3))



acc = (cnf_matrix[1,1] + cnf_matrix[0,0]) / ((cnf_matrix[0,1] + cnf_matrix[1,1]) + cnf_matrix[0,0] + cnf_matrix[1,0])

print("The accuracy of the ML model is ", round(acc, 3))
'''

    Before we feed the dataset into the classier,

    Let's define a function that pre-processes it using the same operations.

''' 

def pre_process(df):

    mod_df = df.drop(['Name', 'Ticket'], axis=1)

    mod_df['Cabin'] = df['Cabin'].fillna('unknown')

    mod_df['Fare'] = df['Fare'].fillna(math.ceil(df['Fare'].mean())) # This is a new line see next line for explanation

    mod_df['Age'] = df['Age'].fillna(math.ceil(df['Age'].mean()))

    

    mod_df, sex_targets = encode_target(mod_df, "Sex")

    mod_df, embarked_targets = encode_target(mod_df, "Embarked")

    mod_df, cabin_targets = encode_target(mod_df, "Cabin")

    

    return mod_df # return modified dataframe



# Pre-process the whole training dataset

whole_training_df = pre_process(raw_training_df)

# Remember we need to pre-process the test dataset similarly 

test_df = pre_process(raw_test_df)



# Assign class features to the variable 'X' for training set 

X = whole_training_df.ix[:, whole_training_df.columns != 'Survived']

# Assign class labels to the variable 'y' for training set

y = whole_training_df.ix[:, whole_training_df.columns == 'Survived']
# The following code was used to debug the error caused by the test set 



# np.all(np.isfinite(test_df)) # This returned true

# np.any(np.isnan(test_df)) # This returned true

# test_df.isnull().any() # This was used to check for NaN values in all columns
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99) # Declare a decision tree clasifier

dt.fit(X, y) # Train the classifier using the entire training set

y_pred = dt.predict(test_df) # Predict class labels for test set



# Now lets create a new table for submission to Kaggle accuracy scoring.

# This should only contain the PassengerIds and predicted class labels(Survived).

submission =  pd.DataFrame(

    {'PassengerId': raw_test_df['PassengerId'], 

     'Survived': y_pred

    })



# Save submission file

submission.to_csv("submission_file.csv", sep=',', encoding='utf-8', index=False)

# submission_file.csv will submitted to Kaggle for accuracy evaluation



# Now lets create another table that contains the PassengerIds, Sexes and class labels (Survived).

# This will be used for data visulisation in the next step.

results =  pd.DataFrame(

    {'PassengerId': raw_test_df['PassengerId'], 

     'Sex': raw_test_df['Sex'], 

     'Survived': y_pred

    })



# results.head(100) # Show the table (Optional)



# Save output file

results.to_csv("class_predictions.csv", sep=',', encoding='utf-8', index=False)



# Check the 'Output' tab at the top of the web page to view the csv file.