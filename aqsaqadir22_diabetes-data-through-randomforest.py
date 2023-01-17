import numpy as np

import pandas as pd
dataset = pd.read_csv('../input/diabetes/diabetes.csv')

#let see the data

dataset
len(dataset)
datasets=np.array(dataset)

datasets
dataset.shape
len(datasets)
train_data = datasets[1:1512,:8]

train_data
train_labels = datasets[1:1512,-1]

train_labels
test_data = datasets[1512:,:8]

test_data
test_labels = datasets[1512:,-1]

test_labels
test_data.shape
test_labels.shape
train_data.shape
train_labels.shape
train_data.dtype
import seaborn as sns

import matplotlib.pyplot as plt
corr = train_data[:8,:9]

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)
# RELATION BETWEEN SKIN THICKNESS AND HAVING DIABETIES OR NOT

ax = sns.barplot(x="SkinThickness", y="Outcome",data=dataset[:40])

## RELATION BETWEEN AGE AND HAVING DIABETIES OR NOT

Bax = sns.barplot(x="Age", y="Outcome",data=dataset[:30])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_data,train_labels)
model.score(train_data,train_labels)
predict= model.predict(test_data)
model.score(test_data,predict)