import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn import model_selection

from sklearn import preprocessing

from sklearn import metrics



from sklearn import neighbors

from sklearn import naive_bayes

from sklearn import discriminant_analysis

from sklearn import linear_model

from sklearn import tree

from sklearn import ensemble

from sklearn import svm

data_file = "../input/lung-cancer-dataset-by-staceyinrobert/survey lung cancer.csv"

GENDER = "GENDER"

AGE = "AGE"

SMOKING = "SMOKING"

LUNG_CANCER = "LUNG_CANCER"

CHRONIC_DISEASE = "CHRONIC DISEASE"

data = pd.read_csv(data_file)



# Convert the "1/2" categorical values to "0/1" and "No/Yes" in the lung cancer column into "0/1"

for col in data.columns:

    if col != AGE:

        data[col] = data[col].astype('category').cat.codes      

data 
# Separate the data into people without lung cancer and those with it

no_data = data[data[LUNG_CANCER] == 0]

yes_data = data[data[LUNG_CANCER] == 1]
yes_data[AGE].plot(title="Age vs. lung cancer", kind="hist")

no_data[AGE].plot(kind="hist")
yes_data[SMOKING].value_counts().plot(title="Smoking vs. lung cancer", kind="bar")
no_data[SMOKING].value_counts().plot(title="Non-smoking vs. lung cancer", kind="bar")
yes_data[CHRONIC_DISEASE].value_counts().plot(title="Chronic disease vs. lung cancer", kind="bar")
no_data[CHRONIC_DISEASE].value_counts().plot(title="Chronic disease vs. lung cancer", kind="bar")
# Separate the data into training and validation sets

data_X = data.iloc[:, 0:15]

data_X = preprocessing.scale(data_X) # Scaling helps LinearSVC converge

data_y = data.iloc[:, 15]

train_X, test_X, train_y, test_y = model_selection.train_test_split(data_X, data_y, test_size=0.5, random_state=0)
def run_classifiers(classifiers, train_X, train_y, test_X, test_y):

    """

    Fits each classifier to the training data and runs it on the test data.

    Prints out the training and test accuracies. 

    """

    results = [] # list of 3-tuples: (classifier name, train accuracy, test accuracy)

    

    # Baseline: a random predictor that guesses "YES" for each data point

    rand_train_pred = np.full(train_y.size, 1)

    rand_test_pred = np.full(test_y.size, 1)

    

    rand_train_results = (rand_train_pred == train_y)    

    rand_train_acc = np.count_nonzero(rand_train_results) / train_y.size

    

    rand_test_results = (rand_test_pred == test_y)    

    rand_test_acc = np.count_nonzero(rand_test_results) / test_y.size

    

    # Print out precision/recall details for test class

    conf_mat = metrics.confusion_matrix(test_y, rand_test_pred)   

    

    print(f"Random classifier")

    

    # Precision rate = # true positives / (# predicted positives)

    # Recall rate = # true positives / # positive data points



    precision = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1])

    recall = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0])

    f1_score = 2 * ((precision * recall) / (precision + recall))



    print(f"Precision rate = {precision}")    

    print(f"Recall rate = {recall}")

    print(f"F1 score = {f1_score}\n")

    

    # Add random results to list

    results.append( ("Random Classifier", rand_train_acc, rand_test_acc))

    

    for clf in classifiers:       

        # Run classifier on train and test data

        clf.fit(train_X, train_y)

        train_pred = clf.predict(train_X)       

        train_acc = metrics.accuracy_score(train_y, train_pred)

        

        test_pred = clf.predict(test_X)

        test_acc = metrics.accuracy_score(test_y, test_pred)

        

        # Print out misclassification metrics

        conf_mat = metrics.confusion_matrix(test_y, test_pred)   

        precision = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1])

        recall = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0])

        f1_score = 2 * ((precision * recall) / (precision + recall))

        

        print(f"{type(clf).__name__}")

        print(f"Precision rate = {precision}")    

        print(f"Recall rate = {recall}")

        print(f"F1 score = {f1_score}\n")



        # Store results

        results.append( ((type(clf).__name__), train_acc, test_acc) )

        

    return results
classifiers = [

    neighbors.KNeighborsClassifier(),

    naive_bayes.GaussianNB(),

    discriminant_analysis.LinearDiscriminantAnalysis(),

    linear_model.LogisticRegression(solver="lbfgs", max_iter=200),

    tree.DecisionTreeClassifier(),

    ensemble.AdaBoostClassifier(),

    ensemble.RandomForestClassifier(n_estimators=100),

    svm.LinearSVC(C=0.01, max_iter=100)

]

results = run_classifiers(classifiers, train_X, train_y, test_X, test_y)

results
# Graph the results

classifier_names = [clf[0] for clf in results]

train_acc = [clf[1] for clf in results]

test_acc = [clf[2] for clf in results]



fig, ax = plt.subplots(1, len(classifier_names), figsize=(18, 3), sharey=True)

for i in range(len(classifier_names)):

    ax[i].scatter(classifier_names[i], train_acc[i])
fig, ax = plt.subplots(1, len(classifier_names), figsize=(18, 3), sharey=True)

for i in range(len(classifier_names)):

    ax[i].scatter(classifier_names[i], test_acc[i])
print(f"Number of data points belonging to class lung cancer = {yes_data.shape[0]}")

print(f"Number of data points *not* belonging to class lung cancer = {no_data.shape[0]}")

print(f"Percentage of data points belonging to class lung cancer = {yes_data.shape[0] / (data.shape[0])}")

print(f"Percentage of data points *not belonging to class lung cancer = {no_data.shape[0] / (data.shape[0])}")