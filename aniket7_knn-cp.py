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

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import datetime

from sklearn import datasets, metrics, neighbors

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
df_placement = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

dataset = pd.DataFrame(df_placement)
dataset['gender'] = dataset['gender'].replace(["M"], 1) 

dataset['gender'] = dataset['gender'].replace(["F"], 0) 

dataset['ssc_b'] = dataset['ssc_b'].replace(["Others"], 0) 

dataset['ssc_b'] = dataset['ssc_b'].replace(["Central"], 1) 

dataset['hsc_b'] = dataset['hsc_b'].replace(["Others"], 0) 

dataset['hsc_b'] = dataset['hsc_b'].replace(["Central"], 1) 

dataset['workex'] = dataset['workex'].replace(["Yes"], 1) 

dataset['workex'] = dataset['workex'].replace(["No"], 0) 

dataset['status'] = dataset['status'].replace(["Placed"], 1) 

dataset['status'] = dataset['status'].replace(["Not Placed"], 0) 

dataset['specialisation'] = dataset['specialisation'].replace(["Mkt&HR"], 1) 

dataset['specialisation'] = dataset['specialisation'].replace(["Mkt&Fin"], 0) 

dataset['degree_t'] = dataset['degree_t'].replace(["Sci&Tech"], 1) 

dataset['degree_t'] = dataset['degree_t'].replace(["Comm&Mgmt"], 0) 

dataset['degree_t'] = dataset['degree_t'].replace(["Others"], 2) 

dataset['hsc_s'] = dataset['hsc_s'].replace(["Commerce"], 0) 

dataset['hsc_s'] = dataset['hsc_s'].replace(["Science"], 1) 

dataset['hsc_s'] = dataset['hsc_s'].replace(["Arts"], 2) 
X = np.array(dataset.drop(['sl_no', 'salary', 'status'],1))

y = np.array(dataset['status'])
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=80, test_size=0.3)
k_range = range(1,20)

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    scores.append(knn.score(X_test, y_test))

    

plt.figure()

plt.xlabel('k')

plt.ylabel('Accuracy')

plt.scatter(k_range, scores)

plt.xticks([0.5, 10, 15, 20])

knn = KNeighborsClassifier(n_neighbors = 2, metric = 'euclidean')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Accuracy Percentage: {} %'.format((knn.score(X_test, y_test))*100))