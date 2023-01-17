# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn import metrics

import matplotlib.pyplot as plt

print('import successful')
df = pd.read_csv('../input/iris/Iris.csv')

df
df['Species'].unique()
ax = plt.style.use('seaborn')

ax = df[df['Species']=='Iris-setosa'].plot(kind='scatter', x=['PetalLengthCm',

                                                              'PetalWidthCm'

                                                             ], y=['SepalLengthCm','SepalWidthCm'],

                                              color='DarkBlue', label='Iris-setosa', figsize=(10, 8))                                                            

df[df['Species']=='Iris-virginica'].plot(kind='scatter', x=['PetalLengthCm',

                                                            'PetalWidthCm'

                                                             ], y=['SepalLengthCm','SepalWidthCm'],

                                              color='#1695de', label='Iris-virginica', ax=ax)

df[df['Species']=='Iris-versicolor'].plot(kind='scatter', x=['PetalLengthCm',

                                                             'PetalWidthCm'

                                                             ], y=['SepalLengthCm', 'SepalWidthCm'],

                                              color='#03fcfc', label='Iris-versicolor', ax=ax)



plt.show()
ax = df[df['Species']=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',

                                              color='DarkBlue', label='Iris-setosa', figsize=(10, 8))    

df[df['Species']=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',

                                              color='#1695de', label='Iris-virginica', ax=ax)

df[df['Species']=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',

                                              color='#03fcfc', label='Iris-versicolor', ax=ax)
ax = df[df['Species']=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',

                                              color='DarkBlue', label='Iris-setosa', figsize=(10, 8))                                                            

df[df['Species']=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',

                                              color='#1695de', label='Iris-virginica', ax=ax)

df[df['Species']=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',

                                              color='#03fcfc', label='Iris-versicolor', ax=ax)
x = df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']].values

x[0:5]
y = df['Species'].values

y[0:5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('shape of x_train is {}'.format(x_train.shape))

print('shape of x_test is {}'.format(x_test.shape))

print('shape of y_train is {}'.format(y_train.shape))

print('shape of y_test is {}'.format(y_test.shape))
iris_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)

iris_classifier
iris_classifier.fit(x_train, y_train)

predict_iris = iris_classifier.predict(x_test)

result = pd.DataFrame(

    {'predicted': predict_iris,

     'actual_value': y_test

                      })

result
accuracy_score = metrics.accuracy_score(y_test, predict_iris)

print('='*100)

print('The Decision Tree accuracy  for the iris classification is {}'.format(accuracy_score))

print('='*100)
k = 4

neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)

neigh
yhat = neigh.predict(x_test)

result2 = pd.DataFrame({

    'predicted': yhat,

    'actual_value': y_test

})

result2
print('='*40)

print('the train_set accuracy is {}'.format(metrics.accuracy_score(y_train, neigh.predict(x_train))))

print('\nthe test_set accuracy is {}'.format(metrics.accuracy_score(y_test, yhat)))

print('='*40)
ks = 10

mean_acc = np.zeros((ks-1))

std_acc = np.zeros((ks-1))

confusion_mx = []

for n in range(1, ks):

    neigh =  KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)

    yhat=neigh.predict(x_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)

    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1, ks), mean_acc, 'g')

plt.fill_between(range(1, ks), mean_acc-1*std_acc, mean_acc+1*std_acc,alpha=0.10)

plt.legend(('Accuracy', '+/- 3std'))

plt.ylabel('Accuracy')

plt.xlabel('no.of neighbors(k)')

plt.show()