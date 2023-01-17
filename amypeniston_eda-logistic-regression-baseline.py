import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

from sklearn.model_selection import cross_validate, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

import keras



params = {'legend.fontsize': 'large',

          'figure.figsize': (10, 8),

         'axes.labelsize': 'large',

         'axes.titlesize':'large',

         'xtick.labelsize':'large',

         'ytick.labelsize':'large'}

pylab.rcParams.update(params)



# Prevent Pandas from truncating displayed dataframes

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



sns.set(style="white")



SEED = 42
# Load master copies of data - these remain pristine

train_ = pd.read_csv("../input/digit-recognizer/train.csv")

test_ = pd.read_csv("../input/digit-recognizer/test.csv")

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")



# Take copies of the master dataframes

train = train_.copy()

test = test_.copy()
train.shape, test.shape
train.head()
# Separate the target variable from the digits

y = train.pop("label")
n_preview = 10

fig, ax = plt.subplots()



for i in range(n_preview):

    plt.subplot(2, 5, i+1)

    image = train.iloc[i].values.reshape((28,28))

    plt.imshow(image, cmap="Greys")

    plt.axis("off")



plt.suptitle("The First 10 MNIST Handwritten Digits", y=0.9)

plt.show()
y[0:10].values
digit_frequency = y.value_counts(normalize=True).to_frame()

unique_digits= np.sort(y.unique())

unique_digits_str = [str(d) for d in unique_digits]



plt.bar(digit_frequency.index, digit_frequency["label"].values)

plt.title("Training Digit Frequency")

plt.xticks(unique_digits, unique_digits_str)

plt.show()
train = train / np.max(np.max(train))

test = test / np.max(np.max(test))
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=SEED)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
lr = LogisticRegression(max_iter=1000)



lr.fit(X_train, y_train)
yhat = lr.predict(X_valid)

score = lr.score(X_valid, y_valid)

print("Baseline score: {:.1%}".format(score))
c_matrix = confusion_matrix(y_valid, yhat, normalize="true")



plt.figure()

sns.heatmap(c_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "Spectral_r")

plt.ylabel("Actual Label")

plt.xlabel("Predicted Label")

plt.title("Accuracy Score: {:.1%}".format(score), size = 15)

plt.show()
fig, ax = plt.subplots(1,2, figsize=(15,6))



sns.countplot(y_train, ax=ax[0])

sns.countplot(y_valid, ax=ax[1])

ax[0].set_title("Training Labels")

ax[1].set_title("Validation Labels")

plt.show()
lr = LogisticRegression(max_iter=1000)



cv_results = cross_validate(lr, train, y, cv=5, return_train_score=True)

cv_results.keys()
print("Train: {}, Validation: {}".format(cv_results["train_score"].mean(), cv_results["test_score"].mean()))
misclassified = []

for i, (pred, actual) in enumerate(zip(yhat, y_valid)):

    if pred != actual:

        misclassified.append(i)
n_preview = 15

samples = X_valid.iloc[misclassified]

fig, ax = plt.subplots(figsize=(18,10))



for i in range(n_preview):

    plt.subplot(3, 5, i+1)

    image = samples.iloc[i].values.reshape((28,28))

    plt.imshow(image, cmap="Greys")

    plt.title("Predicted: {} | Actual: {}".format(yhat[misclassified[i]], y_valid.values[misclassified[i]]))

    plt.axis("off")



plt.suptitle("10 Misclassified Digits: Predicted & Actual Labels", y=0.99)

plt.show()
lr = LogisticRegression(max_iter=1000)



lr.fit(X_train, y_train)



preds = lr.predict(test)
sample_submission["Label"] = preds

# sample_submission.to_csv("baseline-logistic-regression.csv", index=False)

sample_submission.head()
sns.countplot(preds)

plt.title("Count of Predicted Digits")

plt.show()