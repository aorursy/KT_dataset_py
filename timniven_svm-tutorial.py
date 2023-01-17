import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import random

import seaborn as sns; sns.set()

from sklearn import svm
file_name = '../input/SAheart.data'

data = pd.read_csv(file_name, sep=',', index_col=0)

print(len(data))

data.head()
data['famhist'] = data.famhist.apply(lambda x: x == 'Present')
data.chd.value_counts()
for feature in data.columns:

    if feature == 'chd':

        continue

    sns.jointplot(x=feature, y='chd', data=data, kind='kde')

    plt.title('Correlation between %s and Heart Disease' % feature)

    plt.show()
chd = data[data.chd == True]

nchd = data[data.chd == False]

y = plt.scatter(chd.age.values, chd.adiposity.values, c='r')

n = plt.scatter(nchd.age.values, nchd.adiposity.values, c='b')

plt.xlabel('age')

plt.ylabel('adiposity')

plt.legend((y, n), ['chd', 'healthy'])

plt.show()
n_test = int(math.ceil(len(data) * 0.3))

random.seed(42)

test_ixs = random.sample(list(range(len(data))), n_test)

train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]

train = data.iloc[train_ixs, :]

test = data.iloc[test_ixs, :]

print(len(train))

print(len(test))
#features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']

features = ['adiposity', 'age']

response = 'chd'

x_train = train[features]

y_train = train[response]

x_test = test[features]

y_test = test[response]
1. - y_test.mean()
model = svm.SVC(gamma='scale')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

np.mean(y_pred == y_test)
best_acc = 0.

best_c = None

for c in np.linspace(0.1, 1.0):

    model = svm.SVC(C=c, gamma='scale')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = np.mean(y_pred == y_test)

    if acc > best_acc:

        best_acc = acc

        best_c = c

print(best_acc)

print(best_c)
# practice here, using the code above as a template