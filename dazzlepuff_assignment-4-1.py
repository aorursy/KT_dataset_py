import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from IPython.display import Image

from sklearn import tree



dataset = pd.read_csv("../input/assignment_4_1_dataset.csv")

dataset = dataset.drop(columns=["No"])



dataset.describe()
features = []

for i in range(0,len(dataset.columns)-1):

    features.append(dataset.columns[i])

target = "Risk"



dataset.groupby(target).count()
sns.countplot(x="Credit History", hue=target, data=dataset)
sns.countplot(x="Debt", hue=target, data=dataset)
sns.countplot(x="Collatoral", hue=target, data=dataset)
sns.countplot(x="Income", hue=target, data=dataset)
def max_key_value(d): 

     v=list(d.values())

     k=list(d.keys())

     return k[v.index(max(v))], v[v.index(max(v))]



def get_total(table):

    total = 0

    for col in table.columns:

        total += np.sum(table[col])

    return total





def entropy_attr_calc(table):

    entropy = 0

    total = get_total(table)

    for col in table.columns:

        outcome_freq = np.sum(table[col].values)

        prob = outcome_freq / total

        entropy -= prob*np.log2(prob)

    return entropy

        



def entropy_value_calc(table):

    entropies = {}

    for col in table.columns:

        vals = table[col].values

        prob = 0

        entropy = 0

        for i in vals:

            if i>0:

                prob = i/np.sum(vals)

                entropy -= prob*np.log2(prob)

        entropies[col] = entropy

        print("entropy for {} is {}".format(col, entropy))

    return entropies



def information(entropy, table):

    total = get_total(table)

    info = 0

    for col in table.columns:

        info += (np.sum(table[col].values/total))*entropy.get(col)

    return info



def print_ig_info(info, entropy_attribute):

    print("info from {}".format(info))

    print("entropy for {}".format(entropy_attribute))

    print("information gain is {}".format(entropy_attribute - info))



def maximize_IG(dataset, target_name):

    X = dataset.drop(target_name, axis=1)    

    ig_dict = {}

    for col in X:

        crosstable = pd.crosstab(dataset["Risk"], dataset[col])

        crosstable_risk = pd.crosstab(dataset[col], dataset["Risk"])

        print("Calculating entropy for {}".format(col))

        entropy_attribute = entropy_attr_calc(crosstable_risk)

        entropy_values = entropy_value_calc(crosstable)

        info = information(entropy_values, crosstable)

        ig = (entropy_attribute - info)

        ig_dict[col] = ig

        print_ig_info(info, entropy_attribute)

    print(ig_dict)

    return max_key_value(ig_dict)



max_ig_key, max_ig_value = maximize_IG(dataset, "Risk")



print("Max IG attribute: {}".format(max_ig_key))

print("Max IG value: {}". format(max_ig_value))