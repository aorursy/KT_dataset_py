# import what you need here !

import csv

import numpy as np

from collections import defaultdict

from operator import itemgetter
# initialization

filename = "../input/GroceryStoreDataSet.csv"

columns = ["JAM", "MAGGI", "SUGAR", "COFFEE", "CHEESE", "TEA", "BOURNVITA", "CORNFLAKES", "BREAD", "BISCUIT", "MILK"]



reader = csv.reader(open(filename, 'rt'))

dataset = list(reader)



valid_rules = defaultdict(int)

invalid_rules = defaultdict(int)

num_occurances = defaultdict(int)
# create some functions

def convert_values(columns, values):

    processed = [0] * len(columns)

    for value in values[0].split(","):

        if value in columns:

            processed[columns.index(value)] = 1

        else:

            continue

    return processed



def preprocessing(raw_datasets):

    data_ready = []

    for data in raw_datasets:

        data_ready.append(convert_values(columns, data))

    return data_ready



def print_rule(premise, conclusion, support, confidence, features):

    premise_name = features[premise]

    conclusion_name = features[conclusion]

    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))

    print(" - Support: {0}".format(support[(premise, conclusion)]))

    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
# load data to numpy

X = np.array(preprocessing(dataset))
# computing support and confidence

for sample in X:

    for premise in range(len(columns) - 1):

        if sample[premise] == 0:

            continue

        num_occurances[premise] += 1

        for conclusion in range(len(columns)):

            if premise == conclusion:

                continue

            if sample[conclusion] == 1:

                valid_rules[(premise, conclusion)] += 1

            else:

                invalid_rules[(premise, conclusion)] += 1



support = valid_rules

confidence = defaultdict(float)

for premise, conclusion in valid_rules.keys():

    rule = (premise, conclusion)

    confidence[rule] = valid_rules[rule] / num_occurances[premise]
# run

sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

for index in range(len(columns)):

    print("Rule #{0}".format(index + 1))

    premise, conclusion = sorted_support[index][0]

    print_rule(premise, conclusion, support, confidence, columns)
sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for index in range(len(columns)):

    print("Rule #{0}".format(index + 1))

    premise, conclusion = sorted_confidence[index][0]

    print_rule(premise, conclusion, support, confidence, columns)