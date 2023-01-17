import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import random



from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)

train.head(10)
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
y_test = test.iloc[:,0].values.astype('int32') # only labels i.e targets digits
print(X_train.shape)

print(y_train.shape)
plt.figure(figsize=(25,10))

X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(10):

    rand = random.randrange(len(X_train))

    plt.subplot(1, 10, i + 1)

    plt.imshow(X_train[rand], cmap=plt.get_cmap('gray'))

    plt.title(y_train[rand]);
logisticRegr = LogisticRegression()

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')

logisticRegr.fit(X_train, y_train)
plt.figure(figsize=(45,25))

for i in range(20):

    rand = random.randrange(len(X_train))

    plt.subplot(5, 4, i + 1)

    plt.imshow(X_train[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))

    title = str(y_train[rand])

    plt.title("real : %s , predicted :  %s" % (title, logisticRegr.predict(np.reshape(X_train[rand], (1, -1)))))
plt.figure(figsize=(45,25))

for i in range(20):

    rand = random.randrange(len(X_test))

    plt.subplot(5, 4, i + 1)

    plt.imshow(X_test[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.title("predicted :  %s" % (logisticRegr.predict(np.reshape(X_test[rand], (1, -1)))))
labels = logisticRegr.predict(X_test)

labels
output = pd.DataFrame({"ImageId": list(range(1, len(labels) + 1)), "Label": labels})

output.to_csv("output.csv")

output