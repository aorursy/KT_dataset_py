import math

from collections import Counter

import numpy as np

import seaborn as sns

import pandas as pd

import scipy.stats as ss

import matplotlib.pyplot as plt

import sklearn.preprocessing as sp

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/mushrooms.csv')

data.head()
data.shape
data.isnull().values.any()
for feature in data.columns:

    uniq = np.unique(data[feature])

    print('{}: {} distinct values -  {}'.format(feature,len(uniq),uniq))
data = data.drop(['veil-type'], axis=1)
print('Known mushrooms: {}\nUnique mushrooms: {}'.format(len(data.index),len(data.drop_duplicates().index)))
print('Known mushrooms: {}\nMushrooms with same features: {}'.format(

    len(data.index),len(data.drop_duplicates(subset=data.drop(['class'],axis=1).columns).index)))
def conditional_entropy(x,y):

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def theil_u(x,y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x
theilu = pd.DataFrame(index=['class'],columns=data.columns)

columns = data.columns

for j in range(0,len(columns)):

    u = theil_u(data['class'].tolist(),data[columns[j]].tolist())

    theilu.loc[:,columns[j]] = u

theilu.fillna(value=np.nan,inplace=True)

plt.figure(figsize=(20,1))

sns.heatmap(theilu,annot=True,fmt='.2f')

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

ax=sns.countplot(x='odor',hue='class',data=data)

for p in ax.patches:

    patch_height = p.get_height()

    if np.isnan(patch_height):

        patch_height = 0

    ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.05, patch_height+10))

plt.show()
no_odor = data[data['odor'].isin(['n'])]

for j in range(0,len(columns)):

    u = theil_u(no_odor['class'].tolist(),no_odor[columns[j]].tolist())

    theilu.loc[:,columns[j]] = u

theilu.fillna(value=np.nan,inplace=True)

plt.figure(figsize=(20,1))

sns.heatmap(theilu,annot=True,fmt='.2f')

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

ax=sns.countplot(x='spore-print-color',hue='class',data=no_odor)

for p in ax.patches:

    patch_height = p.get_height()

    if np.isnan(patch_height):

        patch_height = 0

    ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.05, patch_height+10))

plt.show()
no_odor_w = no_odor[no_odor['spore-print-color'].isin(['w'])]

(len(data.index) - len(no_odor_w.index)) / len(data.index)
factorized_nw = no_odor_w.copy()

for column in factorized_nw.columns.values:

    f, _ = pd.factorize(factorized_nw[column])

    factorized_nw.loc[:,column] = f

ohe = sp.OneHotEncoder()

X = factorized_nw.drop(['class'],axis=1)

y = factorized_nw['class'].tolist()

ohe.fit(X)

X = ohe.transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)



for i in range(1,6):

    tree = DecisionTreeClassifier(max_depth=i, random_state=42)

    tree.fit(X_train,y_train)

    y_pred = tree.predict(X_test)

    print("Max depth: {} - accuracy:".format(i), accuracy_score(y_test, y_pred, normalize=True))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))