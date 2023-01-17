# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
import ast
import glob
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
import graphviz
def process_features(dataframe):
    df = pd.DataFrame()
    df['event'] = dataframe['event'].apply(
        lambda x: np.array(ast.literal_eval(x)).flatten())

    new_columns = ["Input" + str(x) for x in range(800)]

    df2 = pd.DataFrame()
    df2[new_columns] = pd.DataFrame(
        df['event'].values.tolist(), columns=new_columns)
    df2 = df2.values
    return df2

def process_targets(frame):
    cat_type = CategoricalDtype(categories=['b', 'k', 'r'], ordered=True)
    targets = pd.DataFrame()
    targets['class'] = frame['class'].astype(cat_type).cat.codes.astype(int)
    targets = targets.values
    return targets

def split_dataframe(df):
    #  randomly permute examples
    src_df = df.reindex(np.random.permutation(df.index))

    training_df = src_df.iloc[:50, :]
    validation_df = src_df.iloc[50:90, :]
    testing_df = src_df.iloc[90:, :]

    return training_df, validation_df, testing_df
def train_decision_tree(training_examples, training_targets):
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(training_examples, training_targets)
    return classifier
    
def run_tests(classifier, examples, targets):
    classes = classifier.predict(examples)
    target_names = ['b', 'k', 'r']
    confusion_matrix = metrics.confusion_matrix(targets, classes)
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Decision tree accuracy: ", metrics.accuracy_score(targets, classes))
    print(metrics.classification_report(targets, classes, target_names=target_names))
src_df = pd.read_csv("../input/touch_events.csv", sep=",")

print("division of classes")
print(src_df.groupby(['class']).count())

# split into sets
traindf, valid_df, test_df = split_dataframe(src_df)

training_examples = process_features(traindf)
training_targets = process_targets(traindf)

validation_examples = process_features(valid_df)
validation_targets = process_targets(valid_df)

test_examples = process_features(test_df)
test_targets = process_targets(test_df)

classifier = train_decision_tree(training_examples, training_targets)

print("Prediction results for validation data set")
run_tests(classifier, validation_examples, validation_targets)

print("Prediction results for test data set")
run_tests(classifier, test_examples, test_targets)


print("Prediction results for random prediction model")
random_predictions = np.random.randint(0, 3, validation_targets.size)
target_names = ['b', 'k', 'r']
random_accuracy = metrics.accuracy_score(validation_targets, random_predictions)
print("random model accuracy: ", random_accuracy)
print(metrics.classification_report(validation_targets, random_predictions, target_names=target_names))

# create decision tree graph export to file sensor.pdf
dot_data = tree.export_graphviz(classifier,
                                out_file=None,
                                class_names=['b', 'k', 'r'],
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sensor")
