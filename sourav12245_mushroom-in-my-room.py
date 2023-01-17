# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/mushrooms.csv")

pd.set_option('display.max_columns', None)

#pd.options.display.max_rows

data.head()



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

for col in data.columns:

    data[col] = LE.fit_transform(data[col])



data.head()
data = data.drop('veil-type',axis=1)

data.head()
data.columns.values
#new_data = data[data['class'] == 1]

plt.figure(figsize=[20,20])

a = data.corr()

sns.heatmap(data=a,annot=True,square=True,vmax=1,vmin=0)

plt.show()
def check_inf(i):

    sns.countplot(data=data, x = i)

    plt.show()

    sns.distplot(a=data[i],hist=False,kde=True)

    plt.show()

    sns.factorplot(data= data,x= i,y ='class')

    plt.show()

    sns.boxplot(data=data,y='class',x=i)

    plt.show()

    sns.regplot(data = data, y ='class', x = i )

    plt.show()

check_inf('cap-shape')
round (data[['class','cap-shape']].groupby(['cap-shape'],as_index = False).mean(),3)
pd.crosstab(data['class'],data['cap-shape'])
col = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',

       'spore-print-color', 'population', 'habitat']

corr_comp = []

for i in col:

    b = round (data['class'].corr(data[i]),3)

    print (i,round (b,3))

    #corr_comp.append(b)



from sklearn import cross_validation

new_mush = data

x = np.array(new_mush.drop(['class'],axis = 1))

y = np.array(new_mush['class'])

x_data_train, x_data_test, y_data_train, y_data_test =cross_validation.train_test_split(x, y,test_size = 0.2,random_state = 1)

print (x.shape)

print (y.shape)

print (x_data_train.shape)

print (x_data_test.shape)

print (y_data_train.shape)

print (y_data_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

b = knn.fit(x_data_train,y_data_train)

acc = knn.score(x_data_test,y_data_test)

print (acc)
from sklearn.neighbors import KNeighborsClassifier

kn = range(1,20)

score = []

for i in kn:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_data_train,y_data_train)

    score.append(knn.score(x_data_test,y_data_test))



plt.plot(kn,score)

plt.show()