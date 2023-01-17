# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.datasets import samples_generator

from sklearn.feature_selection import SelectKBest , f_regression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X, y = samples_generator.make_classification(n_samples=150 ,

                                            n_features=25 , n_classes=3 , n_informative=6,

                                            n_redundant =0 , random_state=7)
k_best_selector = SelectKBest(f_regression , k=9)
classifier = ExtraTreesClassifier(n_estimators=60 , max_depth=4)
processor_pipeline = Pipeline([('selector' , k_best_selector) , ('erf' , classifier)])
processor_pipeline.set_params(selector__k=7 , erf__n_estimators=30)
processor_pipeline.fit(X,y)
output = processor_pipeline.predict(X)

print("\nPredicted output:\n", output)



# Print scores

print("\n Score" , processor_pipeline.score(X,y))



# Print the selected features by selector block

status = processor_pipeline.named_steps['selector'].get_support()



# print indices of selected features

selected = [i for i ,x in enumerate(status) if x]

print("\n Indices of selected features:",','.join([str(x) for x in selected]))
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
# Input data 

X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9],  

        [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], 

        [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])
k = 5

test_datapoint = np.array([[4.3, 2.7]])
#plot the input data

plt.figure()

plt.title('Input data')

plt.scatter(X[:,0] , X[:,1], marker='o' , s=75 , color='red')
#Build K Nearest Neighbors Model

knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)

distances , indices = knn_model.kneighbors(test_datapoint)
print("\n K Nearest Neighbors :")

for rank , index in enumerate(indices[0][:k] , start=1):

    print(str(rank) + "-->" , X[index])
# Visualize the nearest neighbors along with the test datapoint  

plt.figure() 

plt.title('Nearest neighbors') 

plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k') 

plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1],  

        marker='o', s=250, color='k', facecolors='none') 

plt.scatter(test_datapoint[:,0], test_datapoint[:,1], 

        marker='x', s=75, color='k') 

 

plt.show()