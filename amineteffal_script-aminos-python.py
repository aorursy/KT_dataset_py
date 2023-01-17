import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import os



print(os.listdir("../input"))



# Read data

data = pd.read_csv("../input/data.csv")



# Some lines

print("Fist five lines : \n", data.head(5))

print("Last five lines : \n", data.tail(5))



# Number of lines in data

print("Number of lines in data :",len(data))



# View columns

print("Columns in data : \n", list(data.columns))





data = data[list(data.columns)[:-1]]

# Some lines

print("Fist five lines : \n", data.head(5))

print("Last five lines : \n", data.tail(5))
pair_vars = []

cols = list(data.columns)

sns.set(style="ticks")

sns.pairplot(data[['diagnosis'] +  cols[2:6]], hue="diagnosis")

# Add (texture_mean, area_mean)

pair_vars.append(('texture_mean', 'area_mean'))

pair_vars.append(('texture_mean', 'perimeter_mean'))

pair_vars.append(('texture_mean', 'radius_mean'))



print(pair_vars)
sns.pairplot(data[['diagnosis'] +  cols[-5:]], hue="diagnosis")
pair_vars.append(('texture_worst', 'area_worst')) 

pair_vars.append(('texture_worst', 'perimeter_worst')) 

pair_vars.append(( 'perimeter_worst', 'compactness_worst'))

pair_vars.append(('area_worst' , 'compactness_worst'))

pair_vars.append(('area_worst' , 'smoothness_worst'))

pair_vars.append(('perimeter_worst' , 'smoothness_worst'))

print(pair_vars)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import numpy as np



for var in pair_vars:

    X = np.array(data[list(var)])

    y = np.array(data['diagnosis'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

    print('Score is : ', clf.score(X_test,y_test), ' for ', var)
# First let's re-fit the model

X = np.array(data[list(('texture_mean', 'area_mean'))])

y = np.array(data['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)



# Plot the decision boundary. For that, we will assign a color to each

# point in the mesh [x_min, x_max]x[y_min, y_max].

x_min, x_max = X[:,0].min() - .5, X[:, 0].max() + .5  # get the minimum and maximum of  'texture_worst'

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 # get the minimum and maximum of  'area_worst'



# step size in the mesh

h = .2  



# Create the mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# predict on mesh

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



# transform Z to int

Z[Z=='M'] = 2

Z[Z=='B'] = 0

Z = Z.reshape(xx.shape)

Z = np.array(Z, dtype='int')



# transform y, y_train and y_test to colors

y_bis = y.copy()

y_bis[y_bis=='M'] = 'red'

y_bis[y_bis=='M'] = 'green'



y_train_bis = y_train.copy()

y_train_bis[y_train_bis=='M'] = 'red'

y_train_bis[y_train_bis=='M'] = 'green'



y_test_bis = y_test.copy()

y_test_bis[y_test_bis=='M'] = 'red'

y_test_bis[y_test_bis=='M'] = 'green'



plt.figure(1, figsize=(15, 8))

plt.subplot(131)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



# Plot also the training points

plt.scatter(X[:, 0], X[:, 1], c=y_bis, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('texture_worst')

plt.ylabel('perimeter_worst')



plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.xticks(())

plt.yticks(())

plt.title('All data')







# plt.figure(2, figsize=(5, 5))

plt.subplot(132)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



# Plot also the training points

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_bis, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('texture_worst')

plt.ylabel('perimeter_worst')



plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.xticks(())

plt.yticks(())

plt.title('train data')







# plt.figure(2, figsize=(5, 5))

plt.subplot(133)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)



# Plot also the training points

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_bis, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('texture_worst')

plt.ylabel('perimeter_worst')



plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.xticks(())

plt.yticks(())

plt.title('test data')





plt.show()





y_pred = clf.predict(X)

data_missclass_all = data.loc[y_pred!=y,:]

temp = pd.DataFrame()

temp['All data'] = data[cols[2:]].mean()

temp['Misclassified_data'] = data_missclass_all[cols[2:]].mean()

print(temp)
starts = [2, 12, 22]

plt.figure(1, figsize=(20, 10))

j = 1

for i in starts:

    start_col = i

    end_col = i + 10

    plt.subplot(1,3,j)

    y_pos = np.arange(start = 1, stop = len(cols[start_col:end_col])*3, step = 3)

    plt.bar(y_pos, data[cols[start_col:end_col]].mean(), align='center',  label='All data')

    plt.bar(y_pos + 1, data_missclass_all[cols[start_col:end_col]].mean(), align='center',  label='Misclassified data')

    plt.xticks(y_pos, cols[start_col:end_col])

    plt.xticks(rotation=90)

    plt.legend()

    j = j+1

plt.show()
