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
dataset=pd.read_csv('/kaggle/input/playback-singer-award-for-bollywood/FILMFARE 2020.csv')
dataset.head()
dataset.info()
dataset.describe(include='O')
dataset.describe()
for data in [dataset]:
    data['BS_2020'] = data['BS_2020'].astype(int)
data   
dataset.groupby(['BS_2020','gender'])['gender','BS_2020'].count()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="gender", col="BS_2020", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "BS_2020", color="steelblue", bins=bins)
Y=dataset.drop(columns=['BS_2020','BS_2019','BS_2018','BS_2017',''])
X=dataset['BS_2020']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=7,test_size=0.2)

Y_train=Y_train.values.reshape(-1,1)
y_test=Y_test.values.reshape(-1,1)
from sklearn.neighbors import KNeighborsClassifier
train_score = []
test_score = []
k_vals = []

for k in range(1, 51):
    k_vals.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    
    tr_score = knn.score(X_train, Y_train)
    train_score.append(tr_score)
    
    te_score = knn.score(X_test, Y_test)
    test_score.append(te_score)
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of K')
plt.ylabel('Model score')
plt.plot(k_vals, train_score, color = 'r', label = "training score")
plt.plot(k_vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()

knn = KNeighborsClassifier(n_neighbors = 30)

#Fit the model
knn.fit(X_train,Y_train)

#get the score
knn.score(X_test,Y_test)