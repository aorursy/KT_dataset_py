import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

iris_df = pd.read_csv("../input/Iris.csv")

iris_df.head()

#iris_df.info()
iris_df.describe()
iris_df.groupby('Species').count()
def plotvio(p,i):

    plt.subplot(2,2,i)

    g = sns.violinplot(y=p, x='Species', data=iris_df, inner = 'quartile')

    #plt.show()
plt.figure(figsize=(15,10))

plotvio('SepalLengthCm',1)

plotvio('SepalWidthCm',2)

plotvio('PetalLengthCm',3)

plotvio('PetalWidthCm',4)
#palette = {'red': 'Iris-setosa','blue': 'Iris-versicolor', 'green': 'Iris-virginica' }



#pd.plotting.scatter_matrix(iris_df, figsize = (10,10))

sns.pairplot(iris_df,hue='Species',diag_kind='kde')
iris_df.drop('Id',axis=1,inplace = True)

iris_df.head()
#iris_df.corr().head()

sns.heatmap(iris_df.corr(),annot=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

iris_df_pre = iris_df.drop('Species', axis=1)

iris_Spe = iris_df['Species']
train_p, test_p, train_t, test_t = train_test_split(iris_df_pre, iris_Spe, test_size = 0.333)
N = [] #No. of Neighbours

A = [] #Accuracy Score



for k in range(1,30):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(train_p, train_t)

    y_pred = model.predict(test_p)

    A.append(accuracy_score(test_t, y_pred))

    N.append(k)

    

plt.grid(True)

plt.plot(N,A)
model = SVC(gamma = 'scale')

model.fit(train_p, train_t)

y_pred = model.predict(test_p)

accuracy_score(test_t, y_pred)
model = DecisionTreeClassifier()

model.fit(train_p, train_t)

y_pred = model.predict(test_p)

accuracy_score(test_t, y_pred)
N = [] #No. of Neighbours

A = [] #Accuracy Score



for k in range(1,20):

    model = RandomForestClassifier(n_estimators=k)

    model.fit(train_p, train_t)

    y_pred = model.predict(test_p)

    A.append(accuracy_score(test_t, y_pred))

    N.append(k)

    

plt.grid(True)

plt.plot(N,A)