# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_path = os.path.join("../input/", "train.csv")
human_activities = pd.read_csv(train_path)
test_path = os.path.join("../input/", "test.csv")
ha_test = pd.read_csv(test_path)

human_activities.info()
ha_test.info()
human_activities.head(5)

ha_test.head(5)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X = human_activities.drop("activity", axis=1)
y = human_activities["activity"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train_scaled, y_train)

print("Train set score: {:.2f}".format(svm.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
ha_test_scaled = scaler.transform(ha_test)
ha_test_pred = svm.predict(ha_test_scaled)
ha_test_pred
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

pca3d = PCA(n_components=3)
pca3d.fit(X_train_scaled)
X_train_3d = pca3d.transform(X_train_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def getcolor(c):
    if c == "LAYING":
        return 'red'
    elif c == "SITTING":
        return 'yellow'
    elif c == "STANDNG":
        return 'orange'
    elif c == "WALKING":
        return 'green'
    elif c == "WALKING_DOWNSTAIRS":
        return 'blue'
    else:
        return 'cyan'

cols = list(map(getcolor, y_train))

ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], color=cols)
from sklearn.metrics import confusion_matrix
y_pred = svm.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=["LAYING", "SITTNG", "STANDNG", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]))