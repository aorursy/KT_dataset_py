import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")



labels = data.iloc[:,:1]

images = data.iloc[:,1:]

#plt.imshow(images.values[i].reshape(28, 28))

import sklearn.ensemble as sk

clf = sk.RandomForestClassifier(n_estimators=10)

clf.fit(images, labels)

print(clf)