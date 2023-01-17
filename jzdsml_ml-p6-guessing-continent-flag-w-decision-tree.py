# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt



flags = pd.read_csv('../input/flags.csv', header=0)
flags.head()
flags.columns
labels = flags[["Landmass"]]



data = flags[["Red" , "Green", "Blue", "Gold", "White", "Black", "Orange"]]



train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)



tree = DecisionTreeClassifier(random_state=1)



tree.fit(train_data, train_labels)



print(tree.score(test_data, test_labels))
scores = []

for i in range(1, 21):

  tree = DecisionTreeClassifier(max_depth = i)

  tree.fit(train_data, train_labels)

  scores.append(tree.score(test_data, test_labels))

  print(tree.score(test_data, test_labels))

  

plt.figure(figsize=(12,7))

plt.plot(range(1,21), scores)

plt.xlabel('tree depth')

plt.ylabel('score')

plt.show()

data = flags[["Red", "Green", "Blue", "Gold",

 "White", "Black", "Orange",

 "Circles",

"Crosses","Saltires","Quarters","Sunstars",

"Crescent","Triangle"]]





train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)



scores = []

for i in range(1, 21):

  tree = DecisionTreeClassifier(max_depth = i)

  tree.fit(train_data, train_labels)

  scores.append(tree.score(test_data, test_labels))

  print(tree.score(test_data, test_labels))

  

plt.figure(figsize=(12,7))

plt.plot(range(1,21), scores)

plt.xlabel('tree depth')

plt.ylabel('score')

plt.show()



#Now the graph looks more like what we’d expect. If the tree is too short, we’re underfitting and not accurately representing the training data. 

#If the tree is too big, we’re getting too specific and relying too heavily on the training data.