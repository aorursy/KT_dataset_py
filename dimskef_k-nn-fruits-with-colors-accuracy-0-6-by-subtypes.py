import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib notebook

import seaborn as sn

from sklearn.model_selection import train_test_split
fruits = pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

fruits.head()
# create a mapping from fruit label value to fruit name to make results easier to interpret

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   

lookup_fruit_name
fruits.shape
for i in fruits.fruit_label.unique():

    print(i,":",len(fruits[fruits['fruit_label']==i]),"(",lookup_fruit_name[i],")")
fruits2=fruits[:]
fruits2.fruit_subtype.unique()
fruits2.fruit_subtype.unique()[0]
lookup_fruit_name2 = dict()

lookup_fruit_name2
c = fruits2.fruit_subtype.unique()

cc = len(c)

cc
for i in range(cc):

    lookup_fruit_name2[fruits2.fruit_subtype.unique()[i]] = i

lookup_fruit_name2
o = fruits2.fruit_subtype

oo = len(o)

fruit_label2 = np.zeros(oo)



for i in range(oo):

    p = o[i]

    fruit_label2[i] = lookup_fruit_name2[o[i]]



fruit_label2 = np.array(fruit_label2, dtype=int)

fruit_label2
fruits4 = fruits2.assign(fruit_label2 = fruit_label2)

fruits4
fruits4.shape
fruits4.head()
def reverse_dict(x):

    q = list(lookup_fruit_name2.keys())[list(lookup_fruit_name2.values()).index(x)]

    return q
reverse_dict(8)
for i in fruits4.fruit_label2.unique():

    print(i,":",len(fruits4[fruits4['fruit_label2']==i]),"(",reverse_dict(i),")")
list(lookup_fruit_name2.keys())[list(lookup_fruit_name2.values()).index(8)]
len(fruits4[fruits4['fruit_label2']==0])
# For this example, we use the mass, width, and height features of each fruit instance

X = fruits4[['mass', 'width', 'height']]

y = fruits4['fruit_label2']



# default is 75% / 25% train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm

fruit_prediction = knn.predict([[100,6.3,8.5]])

# lookup_fruit_name2[fruit_prediction[0]]

fruit_prediction[0]
reverse_dict(fruit_prediction[0])
# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm

fruit_prediction = knn.predict([[20,4.3,5.5]])

reverse_dict(fruit_prediction[0])
fruit_prediction = knn.predict([[180,6.5,5]])

reverse_dict(fruit_prediction[0])
fr_pred = knn.predict(X_test)

fr_pred
fr_pred.shape
y_test
(fr_pred,np.array(y_test))
# its not the end, see below

knn.score(X_test,y_test)
fruits4
fruits4.fruit_label2.unique()
lookup_lname = dict()

lookup_lname
c = fruits4.fruit_label2

cc = len(c)

cc
for i in range(cc):

    lookup_lname[fruits4.fruit_label2[i]] = fruits4.fruit_label[i]

lookup_lname
fr_pred
y_te = np.array(y_test)

y_te
fr_pred = [lookup_lname[i] for i in fr_pred]

fr_pred
y_te = [lookup_lname[i] for i in y_te]

y_te
(fr_pred,y_te)
z = [a - b for a, b in zip(fr_pred,y_te)]

z
len(z)
z.count(0)
#accuracy

accur = (z.count(0) / len(z))

accur