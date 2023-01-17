import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X = train.drop(['label'], axis=1)

y = train["label"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
def plot_confuction_matrix(cm):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title("Classification with Random Forest Classifier")

    plt.colorbar()

    plt.show()
print (metrics.classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

plot_confuction_matrix(confusion_matrix)
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

mlp_clf.fit(X_train, y_train)

y_pred = mlp_clf.predict(X_test)
print (metrics.classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

plot_confuction_matrix(confusion_matrix)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

X = np.array(X)

for train, test in kf.split(X):

    kX_train, kX_test, ky_train, ky_test = X[train], X[test], y[train], y[test]

    kclf = RandomForestClassifier()

    kclf.fit(kX_train, ky_train)

    ky_pred = kclf.predict(kX_test)

    print (metrics.classification_report(ky_test, ky_pred).split('\n')[-2])
y_pred = rf_clf.predict(test)
output = pd.DataFrame(y_pred, columns=["Label"])

output['ImageId'] = pd.Series(range(len(y_pred)), index=output.index)

output = output[['ImageId', "Label"]]

output.to_csv("output.csv", index=False)