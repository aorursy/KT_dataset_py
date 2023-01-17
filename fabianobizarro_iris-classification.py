# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv", index_col=0)



df.shape
df.head()
# change the labes for values

df.loc[df.Species == 'Iris-setosa', 'Species'] = 1

df.loc[df.Species == 'Iris-versicolor', 'Species'] = 2

df.loc[df.Species == 'Iris-virginica', 'Species'] = 3





df.head()
train_values = df.sample(frac=0.6).astype(float)

test_values = df.sample(frac=0.4).astype(float)



#train data

train_inputs = train_values.as_matrix(

    columns=['SepalLengthCm', 'SepalWidthCm',

    'PetalLengthCm', 'PetalWidthCm']).astype(float)



train_predicts = train_values['Species'].values.astype(int)





# test data

test_inputs = test_values.as_matrix(

    columns=['SepalLengthCm', 'SepalWidthCm',

    'PetalLengthCm', 'PetalWidthCm']).astype(float)



test_predicts = test_values['Species'].values.astype(int)
# methos to show the ratio of the dataset



def print_ratio(df, label):

    

    total_values = len(df.loc[df['Species']])



    iris_setosa_total = len(df.loc[df['Species'] == 1])

    iris_setosa_total_percent = iris_setosa_total / total_values  * 100



    iris_versicolor_total = len(df.loc[df['Species'] == 2])

    iris_versicolor_total_percent = iris_versicolor_total / total_values * 100



    iris_virginica_total = len(df.loc[df['Species'] == 3])

    iris_virginica_total_percent = iris_virginica_total / total_values * 100



    print(label)

    print('Iris-setosa: {0} {1:0.2f}%'.format(iris_setosa_total, iris_setosa_total_percent))

    print('Iris-versicolor: {0} {1:0.2f}%'.format(iris_versicolor_total, iris_versicolor_total_percent))

    print('Iris-virginica: {0} {1:0.2f}%'.format(iris_virginica_total, iris_virginica_total_percent))







print_ratio(df, 'Total')

print("")

print_ratio(train_values, 'Train values')

print("")

print_ratio(test_values, 'Test values')
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()



# aqui ta o pulo do gato

model.fit(train_inputs, train_predicts)
# metrics to accuracy

from sklearn import metrics



train_predicted_values = model.predict(train_inputs)



accuracy = metrics.accuracy_score(train_predicts, train_predicted_values) * 100

print('Accuracy {0:0.4f}%'.format(accuracy))
test_predicted_values = model.predict(test_inputs)



accuracy = metrics.accuracy_score(test_predicts, test_predicted_values) * 100

print('Accuracy {0:0.4f}%'.format(accuracy))
def label(number):

    _dict = {

        1: "Iris-setosa",

        2: "Iris-versicolor",

        3: "Iris-virginica"

    }

    return _dict[number]





predicted = model.predict([

    [6.4, 3.2, 4.5, 1.5]

])



flower = label(*predicted)



print('this flower is a {0}'.format(flower))