# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

data.head()
"""Get the X and y data"""



X = np.array(data[['mass','width','height']])

y = np.array(data['fruit_name'])



print('X shape:',X.shape)

print('y shape:',y.shape)
def get_pipeline(k):

    pipeline = Pipeline(steps=[('scaler', StandardScaler()),

                              ('model', get_model(k))

                             ])

    pipeline.fit(X,y)

    return pipeline
def get_model(k):

    knn = KNeighborsClassifier(n_neighbors=k)

    return knn
def get_score(k):

    pipeline = get_pipeline(k)

    scores = cross_val_score(pipeline, X, y)

    return scores.mean()
"""Let´s try some k values and get the best"""

arr_k = [1,3,5,10,30]

dic_score = np.empty(0)

for k in arr_k:

       dic_score = np.append(dic_score,get_score(k))



fig = plt.figure()

fig.subplots_adjust(top=0.8)

ax1 = fig.add_subplot()

ax1.set_ylabel('Score')

ax1.set_xlabel('K value')



plt.plot(arr_k,dic_score,label='Score for k params')

plt.scatter(arr_k,dic_score,alpha=0.5)

plt.legend()

plt.show()
#Looks like 5 is the best value for K

pipeline = get_pipeline(5)

prediction = pipeline.predict(X)

print('scores',dic_score)
disp = plot_confusion_matrix(pipeline, X, y,

                             cmap=plt.cm.Blues,

                             normalize=None)

plt.show()
print(classification_report(y, prediction ))
print('Accuracy',accuracy_score(y, prediction))