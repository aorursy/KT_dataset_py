from sklearn import naive_bayes

from sklearn.metrics import confusion_matrix

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns
NetworkData = pd.read_csv("../input/networks-data/train.csv").values
fig, axes = plt.subplots(3,3)

fig.set_size_inches(18.5, 7.5)

fig.tight_layout()



axes[0][0].hist(NetworkData[:,1])

axes[0][0].set_title("A2")

axes[0][1].hist(NetworkData[:,2])

axes[0][1].set_title("A3")

axes[0][2].hist(NetworkData[:,3])

axes[0][2].set_title("A4")



axes[1][0].hist(NetworkData[:,6])

axes[1][0].set_title("A7")

axes[1][1].hist(NetworkData[:,11])

axes[1][1].set_title("A12")

axes[1][2].hist(NetworkData[:,13])

axes[1][2].set_title("A14")



axes[2][0].hist(NetworkData[:,14])

axes[2][0].set_title("A15")

axes[2][1].hist(NetworkData[:,17])

axes[2][1].set_title("A18")



axes[2][2].hist(NetworkData[:,41])

axes[2][2].set_title("Result")
for i in [1,2,3]:

    le = LabelEncoder()

    le.fit(NetworkData[:,i])

    NetworkData[:,i] = le.transform(NetworkData[:,i])
X = NetworkData[:,[1,2,3,6,11,13,14,17]]

y = NetworkData[:,41]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
MultinomialNBModel = naive_bayes.MultinomialNB()

MultinomialNBModel.fit(X_train, y_train)
y_test_predicted = MultinomialNBModel.predict(X_test)
print(confusion_matrix(y_test, y_test_predicted))

accuracy_score(y_test, y_test_predicted)