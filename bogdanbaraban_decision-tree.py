import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import random

from pprint import pprint
df = pd.read_csv("../input/Iris.csv")

df = df.drop("Id", axis=1)

print(df.shape)

df.head()
# function to split dataframe to train and test data 

def train_test_split(df, test_size):

    if isinstance(test_size, float):

        test_size = round(test_size * len(df))

    indices = df.index.tolist()

    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]

    train_df = df.drop(test_indices)

    return train_df, test_df
random.seed(0) # setting random seed to always have same split

train_df, test_df = train_test_split(df, test_size=20) # 20 rows compose a test dataframe, rest 130 - train
# checks number of classes in dataframe. If n=1 the data is pure, meaning it will overfit

def check_purity(data):

    label_column = data[:, -1]

    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:

        return True

    else:

        return False
def classify_data(data):

    label_column = data[:, -1]

    # returns unique classes and number of instances

    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    # returns index of the class with the biggest number of instances

    index = counts_unique_classes.argmax()

    # returns name of the class with the biggest number fo instances

    classification = unique_classes[index]

    return classification
def get_potential_splits(data):

    potential_splits = {}

    _, n_columns = data.shape

    # excluding the last column which is the label

    for column_index in range(n_columns - 1):

        # initialization of empty array for particular column index

        potential_splits[column_index] = [] 

        # takes all values from dataframe corresponding to column index

        values = data[:, column_index] 

        # takes unique values from values variable

        unique_values = np.unique(values) 

        for index in range(len(unique_values)):

            if index != 0:

                current_value = unique_values[index]

                previous_value = unique_values[index - 1]

                # takes mid point of two close unique values 

                potential_split = (current_value + previous_value) / 2 

                # add to potential splits dict

                potential_splits[column_index].append(potential_split) 

    return potential_splits
def split_data(data, split_column, split_value):

    # gets split column values

    split_column_values = data[:, split_column]

    # data below split

    data_below = data[split_column_values <= split_value]

    # data above split

    data_above = data[split_column_values >  split_value]

    return data_below, data_above
def calculate_entropy(data):

    # takes label of the row

    label_column = data[:, -1]

    # takes number of instances of each class

    _, counts = np.unique(label_column, return_counts=True)

    # array of probabilities of the classes (each number of one instance is divided by sum of the instances) )

    probabilities = counts / counts.sum()

    # calculates entropy

    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy
def calculate_overall_entropy(data_below, data_above):

    # number of data points

    n = len(data_below) + len(data_above)

    # p for the data below divided by number of data points

    p_data_below = len(data_below) / n

    # p for the data above divided by number of data points

    p_data_above = len(data_above) / n

    # overall entropy = p_below * entropy + p_above * entropy

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 

                      + p_data_above * calculate_entropy(data_above))

    return overall_entropy
def determine_best_split(data, potential_splits):

    overall_entropy = 9999

    for column_index in potential_splits: # compares entropy one vs all

        for value in potential_splits[column_index]:

            # splits data to below and above

            data_below, data_above = split_data(data, split_column=column_index, split_value=value)

            # current entropy of below and above splits

            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            # if current entropy  

            if current_overall_entropy <= overall_entropy:

                # update overall entropy 

                overall_entropy = current_overall_entropy

                # then best split columns is column_index

                best_split_column = column_index

                # best split value is value

                best_split_value = value

    

    return best_split_column, best_split_value
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):

    # data preparations

    if counter == 0:

        global COLUMN_HEADERS

        COLUMN_HEADERS = df.columns

        data = df.values

    else:

        data = df           

    

    # base cases

    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):

        # if data is pure => simply return classification

        classification = classify_data(data)  

        return classification

    

    # if data is not pure

    else:    

        counter += 1

        # helper functions 

        potential_splits = get_potential_splits(data)

        # takes split column and value

        split_column, split_value = determine_best_split(data, potential_splits)

        # 

        data_below, data_above = split_data(data, split_column, split_value)

        # instantiate sub-tree

        feature_name = COLUMN_HEADERS[split_column]

        question = "{} <= {}".format(feature_name, split_value)

        sub_tree = {question: []}

        

        # find answers (recursion)

        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)

        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        

        # If the answers are the same, then there is no point in asking the qestion.

        # This could happen when the data is classified even though it is not pure

        # yet (min_samples or max_depth base case).

        if yes_answer == no_answer:

            sub_tree = yes_answer

        else:

            sub_tree[question].append(yes_answer)

            sub_tree[question].append(no_answer)

        

        return sub_tree
dtree = decision_tree_algorithm(train_df, max_depth=3)

pprint(dtree)
from sklearn import tree

from sklearn.tree import export_graphviz

decision_tree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)

decision_tree.fit(train_df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']], train_df['Species'])

export_graphviz(decision_tree, out_file='tree.dot',feature_names = ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm'],

                class_names = df['Species'], filled=True, rounded=True, special_characters=True) 



!dot -Tpng tree.dot -o tree.png -Gdpi=600

from IPython.display import Image

Image(filename = 'tree.png')
test_df.head(20)
test_df.loc[0] = (5.1, 2.6, 3.1, 1.2, 'Iris-setose')

example = test_df.iloc[-1]

print(example)
def classify_example(example, dtree):

    question = list(dtree.keys())[0]

    feature_name, comparison_operator, value = question.split(" ")



    # ask question

    if example[feature_name] <= float(value):

        answer = dtree[question][0]

    else:

        answer = dtree[question][1]



    # base case

    if not isinstance(answer, dict):

        return answer

    

    # recursive part

    else:

        residual_tree = answer

        return classify_example(example, residual_tree)
classify_example(example, dtree)
def calculate_accuracy(df, tree):



    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))

    df["classification_correct"] = df["classification"] == df["Species"]

    

    accuracy = df["classification_correct"].mean()

    

    return accuracy
accuracy = calculate_accuracy(test_df, dtree)

accuracy