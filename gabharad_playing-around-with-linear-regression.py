# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow

import sklearn

import pickle

from matplotlib import pyplot

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn.utils import shuffle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(os.path.join(dirname, 'student-mat.csv'), sep = ';')

data = data[['G1','G2','G3', 'studytime', 'failures', 'absences']]



data.head()

data.to_csv("sample.csv")
predict = 'G3'



X = np.array(data.drop([predict], 1))

y = np.array(data[predict])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)



best = 0

for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)



    linear = linear_model.LinearRegression()



    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(acc)



    if acc > best:

        best = acc

        with open('studentmodel.pickle', 'wb') as f:

            pickle.dump(linear, f)



pickle_in = open("studentmodel.pickle", 'rb')



linear = pickle.load(pickle_in)

    

print("Co: ", linear.coef_)

print("Intercept: ", linear.intercept_)
predictions = linear.predict(x_test)



for x in range(len(predictions)):

    print(predictions[x], x_test[x], y_test[x])
p = 'G1'

pyplot.style.use('ggplot')

pyplot.scatter(data[p], data['G3'])

pyplot.xlabel(p)

pyplot.ylabel('Final Grade')

pyplot.show()



p = 'G2'

pyplot.style.use('ggplot')

pyplot.scatter(data[p], data['G3'])

pyplot.xlabel(p)

pyplot.ylabel('Final Grade')

pyplot.show()



p = 'studytime'

pyplot.style.use('ggplot')

pyplot.scatter(data[p], data['G3'])

pyplot.xlabel(p)

pyplot.ylabel('Final Grade')

pyplot.show()



p = 'failures'

pyplot.style.use('ggplot')

pyplot.scatter(data[p], data['G3'])

pyplot.xlabel(p)

pyplot.ylabel('Final Grade')

pyplot.show()



p = 'absences'

pyplot.style.use('ggplot')

pyplot.scatter(data[p], data['G3'])

pyplot.xlabel(p)

pyplot.ylabel('Final Grade')

pyplot.show()
