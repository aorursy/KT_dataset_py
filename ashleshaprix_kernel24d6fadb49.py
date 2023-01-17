# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import os

import seaborn as sns

print(os.listdir("../input"))

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)
df=pd.read_csv("/kaggle/input/Social_Network_Ads.csv")
print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.info()
df.head()
color_list = ['red' if i=='M' else 'green' for i in df.loc[:,'Gender']]

pd.plotting.scatter_matrix(df.loc[:,df.columns != 'Gender'],

                          c = color_list,

                          figsize = [15,15],

                          diagonal = 'hist',

                          alpha = 0.5, 

                          s = 100,

                          marker = '*')

plt.show()
df.sample(frac=0.3)
df.iloc[:,0:3].dtypes
df.corr()

df.iloc[:,1:].corr()
df.isnull().values.any()
df.isnull().sum()

for i,col in enumerate(df.columns):

    print(i+1,". column is ",col)
#show count Gender

df['Gender'].value_counts()
#show Gender's unique

df['Gender'].unique()
#Gender show bar plot

sns.set(style='whitegrid')

ax=sns.barplot(x=df['Gender'].value_counts().index,y=df['Gender'].value_counts().values,palette="Blues_d",hue=['female','male'])

plt.legend(loc=8)

plt.xlabel('Gender')

plt.ylabel('Frequency')

plt.title('Show of Gender Bar Plot')

plt.show()
sns.barplot(x=df['Gender'].value_counts().index,y=df['Gender'].value_counts().values)

plt.title('Genders other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
df_knn = df[['User ID','Gender', 'EstimatedSalary']]
df_knn.head()
#KNN-2

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors = 5)

x,y = df_knn.loc[:,df_knn.columns != 'Gender'], df_knn.loc[:,'Gender']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction : {}'.format(prediction))

print('With KNN (K=3) accuracy is: ', knn.score(x_test,y_test))
#Best K value selection

neig = np.arange(1,30)

train_accuracy = []

test_accuracy = []

for i, k in enumerate (neig):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train,y_train))

    test_accuracy.append(knn.score(x_test,y_test))



# Plot

plt.figure(figsize=(13,8))

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value vs. Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print('Best Accuracy is {} with K = {}'.format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
#confusion matrix

y_pred=knn.predict(x_test)

y_true=y_test





from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

cm

#confusion matrix visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()