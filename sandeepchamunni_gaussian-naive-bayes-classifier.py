from sklearn import naive_bayes

from sklearn.metrics import confusion_matrix

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns
CancerData = pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv").values
fig, axes = plt.subplots(2,3)

fig.set_size_inches(18.5, 3.5)

fig.tight_layout()

sns.distplot(CancerData[:,0],ax=axes[0][0]).set_title("Mean radius")

sns.distplot(CancerData[:,1],ax=axes[0][1]).set_title("Mean texture")

sns.distplot(CancerData[:,2],ax=axes[0][2]).set_title("Mean perimeter")

sns.distplot(CancerData[:,3],ax=axes[1][0]).set_title("Mean area")

sns.distplot(CancerData[:,4],ax=axes[1][1]).set_title("Mean smoothness")

axes[1][2].hist(CancerData[:,5])

axes[1][2].set_title("Diagnosis")
X = CancerData[:,0:5]

y = CancerData[:,5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
GaussianNBModel = naive_bayes.GaussianNB()

GaussianNBModel.fit(X_train, y_train)
y_test_predicted = GaussianNBModel.predict(X_test)
print(confusion_matrix(y_test, y_test_predicted))

accuracy_score(y_test, y_test_predicted)