import pandas as pd

import numpy as np
%matplotlib notebook

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")
data.head()
L = np.sqrt(784)

L
def plotNum(ind):

    plt.imshow(np.reshape(np.array(data.iloc[ind,1:]), (28, 28)), cmap="gray")
plt.figure()

for ii in range(1,17):

    plt.subplot(4,4,ii)

    plotNum(ii)
X = data.iloc[:, 1:]

y = data['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
unknown = pd.read_csv("../input/test.csv")
unknown.head()
y_out = rfc.predict(unknown)
y_out