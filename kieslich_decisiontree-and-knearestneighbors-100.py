import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler





pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.precision', 2)
# Load data and take a peek

df = pd.read_csv('../input/zoo-animal-classification/zoo.csv')

df.head(3)
animal_class = pd.read_csv('../input/zoo-animal-classification/class.csv')

animal_class
# There isn't any missing values

df.isnull().sum().sum()
# but there is a duplicate:

print(df.loc[df.duplicated(subset='animal_name', keep='first'), :])

# Two frogs but one is venomous so we will let them live to fight another day

print(df.loc[df.animal_name=='frog', :])
# Let's see how many there is of all the types

sns.countplot(df.class_type)
# And how many are there with x legs

sns.countplot(df.legs)
# Not many 5 or 8 legged creatures:

df.loc[(df.legs == 5) | (df.legs == 8), :]

# Sleipnir didn't make it to the Ark
# Does seasnakes breathe? YES

df.loc[76, 'breathes'] = 1
# Prepare the data for classification

X = df.drop(columns=['animal_name', 'class_type']).values

y = df.class_type



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(X_train, y_train)

print(f'Train score: {dtc.score(X_train, y_train)}')

print(f'Test score : {dtc.score(X_test, y_test)}')
# Finding the misclassification:

y_pred = dtc.predict(X_test)

dd = pd.DataFrame({'test': y_test, 'pred': y_pred})

dd.loc[dd.test != dd.pred, :]
print(df.loc[df.class_type==3, :])

print(df.loc[df.class_type==5, :])

# The seasnake is the problem. The seasnake is the only reptile that is aquatic, and is therefor classified as a bug. This can be seen from the next figure, bottom left,

# where x[5] <= 0.5 splits into class_type 3 or class_type 5.

fig = plt.figure(figsize=(14, 12))

plot_tree(dtc, rounded=True, precision=2, fontsize=10);
# Prepare the data for classification

X = df.drop(columns=['animal_name', 'class_type']).values

y = df.class_type



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)



knc = KNeighborsClassifier(n_jobs=15, n_neighbors=3, weights='distance', p=1)

knc.fit(X_train, y_train)

print(f'Train_score: {knc.score(X_train, y_train)}')

print(f'Test_score: {knc.score(X_test, y_test)}')

# Just to test it

y_pred = knc.predict(X_test)

print(sum(y_pred != y_test))