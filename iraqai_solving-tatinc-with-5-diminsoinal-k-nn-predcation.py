import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/train.csv')



dataset.head()
X = dataset.iloc[:, [4 ,2,6,5 ]].values

y = dataset.iloc[:, 1].values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values =  "NaN" , strategy = "mean" , axis = 0)

imputer = imputer.fit(X[:, 1:])

X[:, 1:] = imputer.transform(X[:,1:])
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np



fig = plt.figure()

fig.set_size_inches(98.5, 90.5)

ax = fig.add_subplot(999, projection='3d')





i=0

for g , xs ,ys , zs  in X:

    c = y[i]

    i +=1

    if c == 1:

        cc='blue'  #servive

    else:

        cc='yellow'  #died

    if g == 1:

        m = "^"   # male

    else:

        m = "v"   #famle

    

    ax.scatter(xs, ys, zs , c=cc , marker=m)



ax.set_xlabel('class')

ax.set_ylabel('sibsp')

ax.set_zlabel('age')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 4 , metric = 'minkowski' , p = 2)

classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
def getscore(orgin , predicted):

    score = len(orgin)

    print(score)

    j = 0

    for i in predicted:

        if i == orgin[j]:

            j +=1

        else:

            score -=1

            j +=1

    result = (score / float(len(orgin))  ) * 100

    print(score)

    result = str(result)

    return result+'%'
getscore(y_test,y_pred)
fig = plt.figure()

fig.set_size_inches(98.5, 90.5)

ax = fig.add_subplot(999, projection='3d')





i=0

for g , xs ,ys , zs  in X_test:

    c = y_test[i]

    if c == 1:

        cc='blue'  #servive

    else:

        cc='red'  #died

    if g == 1:

        m = "^"   # male

    else:

        m = "v"   #famle

    if y_pred[i] == y_test[i]:

        if y_pred[i] == 1:

            cc = 'green'

        else:

            cc = 'yellow'

        m = "+"

    i +=1

    

    ax.scatter(xs, ys, zs , c=cc , marker=m )



ax.set_xlabel('class')

ax.set_ylabel('sibsp')

ax.set_zlabel('age')

plt.show()
realdata = pd.read_csv('../input/test.csv')

realdata.head()
#y_real = pd.read_csv('../input/gender_submission.csv')

#y_real.head()

# i try to include these file but its not work so  these step  skipped


X_real = realdata.iloc[:, [3 ,1,5,4 ]].values

#y_real = y_real.iloc[:, 1].values

X_real
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values =  "NaN" , strategy = "mean" , axis = 0)

imputer = imputer.fit(X_real[:, 1:])

X_real[:, 1:] = imputer.transform(X_real[:,1:])

X_real
from sklearn.preprocessing import LabelEncoder

labelencoder_X_real = LabelEncoder()

X_real[:,0] = labelencoder_X_real.fit_transform(X_real[:,0])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_real = sc.fit_transform(X_real)
y_real_pred = classifier.predict(X_real)

fig = plt.figure()

fig.set_size_inches(98.5, 90.5)

ax = fig.add_subplot(999, projection='3d')

i=0

for g , xs ,ys , zs  in X_real:

    c = y_real_pred[i]

    if c == 1:

        cc='blue'  #servive

    else:

        cc='red'  #died

    if g == 1:

        m = "^"   # male

    else:

        m = "v"   #famle



    i +=1

    

    ax.scatter(xs, ys, zs , c=cc , marker=m )



ax.set_xlabel('class')

ax.set_ylabel('sibsp')

ax.set_zlabel('age')

plt.show()
y_real_pred
ids = realdata.iloc[:, 0].values

mamon_predcation = np.vstack((ids, y_real_pred)).T

np.savetxt("mamon.csv", mamon_predcation, delimiter=",")
mamon_predcation