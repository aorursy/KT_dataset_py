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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline
df=pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
df.head()
df.info()
df.describe()
#to check corrupted cells

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis') #no empty cells
df.fruit_name.value_counts()
df.fruit_subtype.value_counts()
px.sunburst(df,path=['fruit_name','fruit_subtype'],color='mass',values='width')
px.scatter_3d(df,x='width',y='height',z='mass',color='color_score')
px.scatter(df,x='width',y='height',color='fruit_name',marginal_y='violin',marginal_x='box')
px.scatter_matrix(df,dimensions=['fruit_name','mass','width','height'],color='color_score')
plt.plot(df['height'],label='Height')

plt.plot(df['width'],label='Width')

plt.legend()
predicting=dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
X=df[['mass','width','height']]

y=df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
from sklearn.metrics import classification_report,confusion_matrix
print("Accuracy on training set: {:.3f}".format(knn.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))
pred=knn.predict(X_test)
print(classification_report(y_test,pred))
training_accuracy=[]

test_accuracy=[]

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:

    clf=KNeighborsClassifier(n_neighbors=n_neighbors)

    clf.fit(X_train,y_train)

    training_accuracy.append(clf.score(X_train,y_train))

    test_accuracy.append(clf.score(X_test,y_test))



plt.plot(neighbors_settings,training_accuracy,label='Training Accuracy')

plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')

plt.xlabel("Accuracy")

plt.ylabel('n_neighbors')

plt.show()
knn1 = KNeighborsClassifier(n_neighbors=1)



knn1.fit(X_train,y_train)

pred = knn1.predict(X_test)

print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
knn6 = KNeighborsClassifier(n_neighbors=6)



knn6.fit(X_train,y_train)

pred = knn6.predict(X_test)

print('WITH K=6')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn4 = KNeighborsClassifier(n_neighbors=4)



knn4.fit(X_train,y_train)

pred = knn6.predict(X_test)

print('WITH K=4')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
prediction1=knn4.predict([[100,6.3,8]])

predicting[prediction1[0]]
prediction2=knn4.predict([[105,6.1,6]])

predicting[prediction2[0]]