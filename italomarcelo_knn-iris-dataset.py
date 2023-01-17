import math 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def euclideanDistance(p1,p2):

    dist=0

    for i in range(0,len(p1)):

        dist += (p2[i] - p1[i])**2

    x = math.sqrt(dist)

    return x



def manhattanDistance(p1,p2):

    dist=0

    for i in range(0,len(p1)):

        dist += p2[i] - p1[i]

#     x = (p2[0] - p1[0]) + (p2[1] - p1[1])

    return dist
ed = euclideanDistance([1,1], [5,4])

ed
md = manhattanDistance([1,1], [5,4])

md
from sklearn import datasets

iris = datasets.load_iris()
iris
df = pd.DataFrame(iris.data)
df.columns = ['sepalL', 'sepalW', 'petalL', 'petalW']
df.describe()
df.info()
df.isnull().sum().sum()
df['target'] = iris.target
df.head()
tnames = iris.target_names.tolist()
df['targetName'] = df.target.apply(lambda x: tnames[x])
df.sample(5)
import matplotlib.pyplot as plt

import seaborn as sns
view = df.groupby('targetName').count()['target']



labels = view.index

sizes = view.values



fig1, ax1 = plt.subplots(figsize=(10,7))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
X = df.iloc[:, :4]

y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

predict = model.predict(X_test)
predict
acc = accuracy_score(y_test, predict)

acc
cm = confusion_matrix(y_test, predict)

cm
pd.DataFrame(data=cm, index=iris.target_names, columns=iris.target_names)
len(y_test), len(predict)
pSpec = []

for i in predict:

    pSpec.append(tnames[i])
compare = pd.DataFrame()

compare['ySpecie'] = y_test.apply(lambda x: tnames[x]) 

compare['pSpecie'] = pSpec

compare
compare.value_counts()
compare.ySpecie.value_counts()
compare.pSpecie.value_counts()
k = 30

myKs = np.zeros(k-1)
myKs
for n in range(1,k):

    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)

    predN = neigh.predict(X_test)

    myKs[n-1]=accuracy_score(y_test, predN)
myKs
print(f'Best Acc: {myKs.max():.5f} e K={myKs.argmax()}')
sns.lineplot(x=range(1,k), y=myKs)
model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)

predict = model.predict(X_test)
confusion_matrix(y_test, predict)