import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import random
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
digits = load_digits()

# well, What is this dataset ?!

# What kind of information does it hold ?!

digits.data.shape
plt.figure(figsize=(25,10))

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):

    plt.subplot(1, 5, index + 1)

    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)

    plt.title('Training: %i\n' % label, fontsize = 20)
rand = random.randrange(digits.data.shape[0])

img = plt.imshow(digits.data[rand].reshape((8, 8)), cmap=plt.cm.gray)

print(digits.target[rand])
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
plt.figure(figsize=(25,10))

for i in range(10):

    rand = random.randrange(len(x_test))

    plt.subplot(1, 10, i + 1)

    plt.imshow(x_test[rand].reshape(8, 8), cmap=plt.cm.gray)

    print(y_test[rand])

    print(logisticRegr.predict(x_test[rand].reshape(1, -1)))

    print('-*^'*5)