from sklearn import naive_bayes

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn import neighbors



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time
random_state=51831421

input_path = '../input/mse-bb-4-ss2020-mal-congressional-voting/'

output_path = '../output/kaggle/working/'

data_train = pd.read_csv(input_path + 'CongressionalVotingID.shuf.train.csv')

data_test = pd.read_csv(input_path + 'CongressionalVotingID.shuf.test.csv') 
data_test = data_test.replace('unknown', np.nan)

data_test.set_index('ID', inplace=True)

data_test = data_test.replace('y', 1)

data_test = data_test.replace('n', 0)



data_train = data_train.replace('unknown', np.nan)

data_train = data_train.replace('y', 1)

data_train = data_train.replace('n', 0)

data_train.set_index('ID', inplace=True)

y_train = data_train['class']

X_train = data_train.drop('class', axis=1)





X_test = data_test



X_train = X_train.apply(lambda x: x.fillna(x.median()),axis=0)

X_test = X_test.apply(lambda x: x.fillna(x.median()),axis=0)
def initialize_classifiers():

    classifiers = {}



    # K Neighbors Classifier

    n_neighbors = [5, 10, 15]

    for entry in n_neighbors:

        key = 'KNeighbors_' + str(entry)

        classifiers[key] = neighbors.KNeighborsClassifier(entry)



    # Naive Bayes

    classifiers['GaussianNaiveBayes'] = naive_bayes.GaussianNB()



    # Perceptron

    classifiers['Perceptron'] = Perceptron(random_state=random_state)



    # Decision Trees

    # With Information Gain Full Tree

    classifiers['DecisionTree_Full'] = DecisionTreeClassifier(criterion = 'entropy', 

                                                              random_state = random_state)

    # With Information Gain Pruned Tree

    classifiers['DecisionTree_Pruned'] = DecisionTreeClassifier(criterion = 'entropy', max_depth=3,

                                                         random_state = random_state)

    

    # Random_Forests

    classifiers['RandomForest_100'] = RandomForestClassifier(n_estimators=100, random_state = random_state)

    classifiers['RandomForest_300'] = RandomForestClassifier(n_estimators=300, random_state = random_state)

    

    # SVMs

    classifiers['SVC'] = SVC(gamma='auto', random_state=random_state)

    classifiers['SVC_Linear'] = LinearSVC(tol=1e-5, random_state=random_state)

    

    # DNN

    classifiers['MLP'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    

    return classifiers
classifiers = initialize_classifiers()

predictions = {}



for classifier_name in classifiers:



    classifier = classifiers[classifier_name]

    start_time_train = time.time()

    classifier.fit(X_train, y_train)

    end_time_train = time.time()



    # predict the test set on our trained classifier

    start_time_test = time.time()

    y_test_predicted = classifier.predict(X_test)

    end_time_test = time.time()

    predictions[classifier_name] = y_test_predicted

    



        
prediction_df_dict = {}

for classifier_name, prediction in predictions.items():

    prediction_df_dict[classifier_name] = pd.DataFrame({'ID':X_test.index.tolist()

                                                        ,'class':prediction})

    prediction_df_dict[classifier_name].set_index('ID', inplace=True)
prediction_df
for classifier_name, prediction_df in prediction_df_dict.items():

    prediction_df.to_csv(classifier_name + '-submission.csv', index = True, header=True)