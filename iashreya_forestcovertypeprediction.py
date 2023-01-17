import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale

from sklearn.metrics import classification_report
dataset = pd.read_csv('../input/covtype.csv')
dataset.shape
dataset.head()
#plot 1

columns = ["Soil_Type"+str(i) for i in range(1,41)]

count_ones = []

for i in columns:

    count_ones.append(dataset[dataset[i]==1][i].count())

y_pos = np.arange(len(columns))

plt.figure(figsize=(10,5))

plt.bar(y_pos, count_ones, align="center", alpha=0.5)

plt.xticks(y_pos, [i for i in range(1,41)])

plt.ylabel("Number of Positive Examples")

plt.xlabel("Soil Type")

plt.title("Soil Type Analysis")

plt.show()



#plot 2

columns = ["Soil_Type"+str(i) for i in range(1,41)]

count_zeros = []

for i in columns:

    count_zeros.append(dataset[dataset[i]==0][i].count())

y_pos = np.arange(len(columns))

plt.figure(figsize=(10,5))

plt.bar(y_pos, count_zeros, align="center", alpha=0.5)

plt.xticks(y_pos, [i for i in range(1,41)])

plt.ylabel("Number of Negative Examples")

plt.xlabel("Soil Type")

plt.title("Soil Type Analysis")

plt.show()
dataset[dataset['Soil_Type7']==1]['Cover_Type'].value_counts()
dataset['Cover_Type'].value_counts()
y = dataset['Cover_Type']

x = dataset.drop(['Cover_Type'], axis=1)
x = scale(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
len(x_train), len(x_test)
lr_clf = LogisticRegression(penalty='l1', C=0.1)

lr_clf.fit(x_train, y_train)
predictions = lr_clf.predict(x_test)

print(classification_report(predictions, y_test))