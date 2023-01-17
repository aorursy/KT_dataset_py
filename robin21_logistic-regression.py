import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
data = pd.read_csv("../input/creditcard.csv")

data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
from sklearn.preprocessing import StandardScaler



data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)

data.head()