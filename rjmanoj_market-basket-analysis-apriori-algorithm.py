import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
dataset = pd.read_csv('../input/market-basket-optimization', header = None)
#Getting the list of transactions from the dataset
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
# Training Apriori algorithm on the dataset
rule_list = apriori(transactions, min_support = 0.003, min_confidence = 0.3, min_lift = 3, min_length = 2)
# Visualizing the list of rules
results = list(rule_list)
for i in results:
    print('\n')
    print(i)
    print('**********') 