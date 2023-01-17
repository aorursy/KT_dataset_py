import itertools

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
x = pd.read_csv("../input/market-based-optimization/Market_Basket_Optimisation.csv", header=None)

x.describe()
products = pd.unique(pd.concat([x.iloc[:, col].dropna() for col in x.columns])).tolist()



print("Unique products:", len(products))
MIN_LENGTH = 2

MAX_LENGTH = 3



relations = {}

for idx, row in x.iterrows():

    row = row.dropna()

    for i in range(MIN_LENGTH, MAX_LENGTH + 1):

        for j in itertools.combinations(row, i):

            if frozenset(j) in relations:

                relations[frozenset(j)] += 1

            else:

                relations[frozenset(j)] = 1



print("Total associations:\t", len(relations))
MIN_SUPPORT = int(len(x) * 0.02)



supported_relations = {}

for key, value in relations.items():

    if value >= MIN_SUPPORT:

        supported_relations[key] = value



print("Supported associations:\t", len(supported_relations))
plot_relations = list(supported_relations.items())

plot_relations = sorted(plot_relations, key=lambda r: len(r[0]), reverse=True)



plt.subplots(figsize=(12, len(plot_relations) // 3))

plt.title("Itemsets Frequencies (support >= {})".format(MIN_SUPPORT), {"fontsize":20})

for key, value in plot_relations:

    product = "(" + str(len(key)) + ") " + ", ".join(sorted(list(key)))

    if value >= MIN_SUPPORT: 

        plt.barh(product, value, color="#3333ee")

    else:

        plt.barh(product, value, color="#cccccc")

    plt.text(value, product, str(value))