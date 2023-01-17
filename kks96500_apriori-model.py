#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#importing the dataset

dataset=pd.read_csv('../input/Market_Basket_Optimisation.csv',header=None)
dataset


transactions=[]

for i in range(0,7501):

    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
#training the apriori on the dataset

from apyori import apriori

rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
results=list(rules)
results