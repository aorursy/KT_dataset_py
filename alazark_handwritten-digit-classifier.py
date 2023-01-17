# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

X_test = pd.read_csv("../input/test.csv")

print(train.shape)

print(X_test.shape)
#traning samples

train.head(2)
#reconstruction

from matplotlib import pyplot as plt

data =train.ix[0, 'pixel0':'pixel783'].reshape(28,28)

plt.subplot(221)

plt.imshow(data, cmap=plt.get_cmap('gray'))
#getting data ready for classifiers

X_train = train.ix[:,'pixel0'::]

y_train = train.ix[:,'label']

np.arange(1,2)

#X_test as above
#RandomForest algorithm

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100, random_state=0)

rfc.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rfc.score(X_train, y_train)))

y_pred = rfc.predict(X_test)
#GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor

grbr = GradientBoostingRegressor(random_state=0)

grbr.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(grbr.score(X_train, y_train)))

y_pred = grbr.predict(X_test)
'''from sklearn import svm, metrics

# Create a classifier: a support vector classifier

classifier = svm.SVC(gamma=0.091)



# We learn the digits on on the traning data

classifier.fit(X_train, y_train)

predicted_train = classifier.predict(X_train)

# Now predict the value of the digit on the test data

y_pred = classifier.predict(X_test)



print("Classification report for classifier on the tranining data %s:\n%s\n"

      % (classifier, metrics.classification_report(predicted_train, y_train)))

print("Confusion matrix on training data:\n%s" % metrics.confusion_matrix(predicted_train, y_train))'''
submission = pd.DataFrame({

        "ImageId": np.arange(1,len(X_test)+1),

        "Label": y_pred

    })

submission.to_csv('sample_submission.csv', index=False)