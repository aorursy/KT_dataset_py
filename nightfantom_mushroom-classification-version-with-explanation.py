import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder

from sklearn import tree

from sklearn.metrics import f1_score
mushrooms = pd.read_csv("../input/mushrooms.csv")
def analyze_proportions(feature):

    class_map = {}

    for pair in mushrooms[["class", feature]].iterrows():

        classes = class_map.get(pair[1][feature])

        if classes == None:

            classes = {"p": 0, "e": 0}

            class_map[pair[1][feature]] = classes

        amount = classes[pair[1]["class"]]

        amount = amount + 1

        classes[pair[1]["class"]] = amount

    return class_map
feature_map = {}

for feature in list(mushrooms):

    if feature == "class":

        continue

    feature_map[feature] = analyze_proportions(feature)
def extract_value_of_feature(dictonary):

    amount_of_keys = len(dictonary.keys())

    edible = np.zeros(amount_of_keys)

    poisonous = np.zeros(amount_of_keys)

    keys = []

    index = 0;

    for key, value in dictonary.items():

        edible[index] = value["e"]

        poisonous[index] = value["p"]

        index = index + 1

        keys.append(key)

    return edible, poisonous, keys



def show_bars_for_feature(name, dictonary):

    amount_of_keys = np.arange(len(dictonary.keys()))

    width = 0.35

    class_value = extract_value_of_feature(dictonary)



    fig, ax = plt.subplots()



    rects1 = ax.bar(amount_of_keys, class_value[0], width, color='r')

    rects2 = ax.bar(amount_of_keys + width, class_value[1], width, color='y')



    # add some text for labels, title and axes ticks

    ax.set_ylabel('Amount of mushrooms')

    ax.set_title('Amount edible and poisonous mushrooms slice ' + name)

    ax.set_xticks(amount_of_keys + width / 2)

    ax.set_xticklabels(class_value[2])



    ax.legend((rects1[0], rects2[0]), ('Edible', 'Poisonous'))

    plt.show()

for key, summary in feature_map.items():

    show_bars_for_feature(key, summary)
data = mushrooms[["class", "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]]

label_encoder = LabelEncoder()

categorical_columns = data.columns[data.dtypes == 'object']

for column in categorical_columns:

    data[column] = label_encoder.fit_transform(data[column])
amount = int(len(data) * 0.8)

train_data = data[0:amount][[ "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]].values

train_labels = data[0:amount][[ "class"]].values



test_data = data[amount:][[ "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]].values

test_labels = data[amount:][[ "class"]].values
clf = tree.DecisionTreeClassifier()

clf.fit(train_data, train_labels)
res = clf.predict(test_data)

f1_res = f1_score(test_labels, res)

print("F1 Score = " + str(f1_res))