import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
df = pd.read_csv('../input/heart.csv')

df.head()
df.info()
df.describe()
sb.set_style('whitegrid')

plt.figure(figsize=(30,15))

sb.pairplot(df, hue='target', palette='coolwarm')
for col in df.drop(['target'], axis=1).columns:

    sb.lmplot(data=df, x=col, y='target', fit_reg=False, hue='target',palette='Set2', legend=False, scatter_kws={'alpha':0.5})
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
X = df.drop(['target'], axis=1)

y = df['target']



from sklearn.preprocessing import StandardScaler

stds = StandardScaler()

X = stds.fit_transform(X)



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=101)

def knnprediction(k, Xtrain, ytrain, Xtest):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(Xtrain, ytrain)

    pred = knn.predict(Xtest)

    return pred
error_mean = []

for k in range(1, 41):

    pred_i = knnprediction(k, Xtrain, ytrain, Xtest)

    error_mean.append(np.mean(pred_i != ytest))
plt.plot(range(1,41), error_mean, linestyle='dashed', color='b', marker='o', markerfacecolor='r', markersize=10)

plt.title('K-Value vs Error Mean')

plt.show()
pred = knnprediction(14, Xtrain, ytrain, Xtest)
cm = confusion_matrix(ytest, pred)

print('True Negative : ' + str(cm[0][0]))

print('False Positive : ' + str(cm[0][1]))

print('False Neagtive  : ' + str(cm[1][0]))

print('True Positive : ' + str(cm[1][1]))
print(classification_report(ytest, pred))