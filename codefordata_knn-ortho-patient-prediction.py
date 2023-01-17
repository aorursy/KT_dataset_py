# importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
#avoiding warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
# import data

df = pd.read_csv('../input/column_2C_weka.csv')

df.head()
# viewing the data info

df.info()
df.describe()
# viewing the target data

sns.countplot(df['class'],data=df)

df['class'].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

x,y = df.loc[:,df.columns != 'class'],df.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('prediction {}'.format(prediction))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=7)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('knn with k = 3 accuracy : ',knn.score(x_test,y_test))
iterations = np.arange(1,25)

train_accuracy = []

test_accuracy = []

for i,j in enumerate(iterations):

    knn = KNeighborsClassifier(n_neighbors=j)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train,y_train))

    test_accuracy.append(knn.score(x_test,y_test))



plt.figure(figsize=[13,10])

plt.plot(iterations,test_accuracy,label = 'Test Accuracy')

plt.plot(iterations,train_accuracy,label='Train Accuracy')



plt.legend()

plt.title('Accuracy curve')

plt.xlabel('No.of labels')

plt.ylabel('Accuracy')

plt.xticks(iterations)

print('Best Accuracy is {} with k={}'.format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
cm = confusion_matrix(y_test,prediction)

sns.heatmap(cm,annot=True,fmt='f')