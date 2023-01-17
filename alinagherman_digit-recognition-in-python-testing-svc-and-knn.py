# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read sample_submission files

sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_submission.head()
# Read train files

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train.head()
# Show some images

for x in range(0, 20):

    image = train.loc[x,train.columns != "label"]

    plt.imshow(np.array(image).reshape((28, 28)), cmap="gray")

    plt.show()

    

    plt.hist(image)

    plt.xlabel("Pixel Intensity")

    plt.ylabel("Counts")

    plt.show()
# Number of train images

print("Number of images: %d" % len(train))

train.head()
# split the data

train_images = train.loc[:, train.columns != "label"] / 255

train_labels = train.label





#Split arrays or matrices into random train and test subsets

#Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.

x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)

print ('y_test ****************')

print(y_test)

print ('x_test ****************')

print(x_test)

print ('y_train ****************')

print(y_train)

print ('y_train ****************')

print(y_train)
# this takes about 20 minutes. accuray = 94.0



#SVC classifier

model = SVC()

model.fit(x_train, y_train)



# this takes about 5 min also

test_predicts = model.predict(x_test)

print(test_predicts)



from sklearn.metrics import accuracy_score

test_acc = round(accuracy_score(y_test, test_predicts) * 100)
test_acc
# KNN - RandomForestClassifier

# this takes about 10 minutes. accuray = 96.0



modelRandomForestClassifier = RandomForestClassifier(n_estimators=100)

modelRandomForestClassifier.fit(x_train, y_train)

testRandomForestClassifier_predicts = modelRandomForestClassifier.predict(x_test)

test_acc_RandomForestClassifier = round(accuracy_score(y_test, testRandomForestClassifier_predicts) * 100)
test_acc_RandomForestClassifier
# Read test files & apply model & write the output in a submission file

test_for_submission = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_for_submission.head()



test = test_for_submission.loc[:, :] / 255

submit = modelRandomForestClassifier.predict(test)

pd.DataFrame(submit).to_csv('submit.csv', index=False) 