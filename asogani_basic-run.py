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
train_data = pd.read_csv(r'../input/train.csv')

test_data = pd.read_csv(r'../input/test.csv')

print("train_data_size : \n")

print(train_data.shape)

print("\n")

print("test_data_size : \n")

print(test_data.shape)



print("Follwoing are the features: \n")

print(train_data.columns)

print("\n")

print("Following are the different labels \n")

print(train_data.type.drop_duplicates())

print("\n")

print("Following are the different colors \n")

print(train_data.color.drop_duplicates())
df = pd.DataFrame(train_data.groupby(['type','color'])['id'].count())

df.columns = ['count_by_color_and_type']

df.reset_index(inplace = True)

df.pivot(index = 'type',columns = 'color', values = 'count_by_color_and_type')
type_dict = {'Ghost':1,'Ghoul':2,'Goblin':3}

train_data.groupby(['type'])['id'].count()
train_data['int_type'] = train_data['type'].apply(lambda x: type_dict[x])

x_variable = ['bone_length','rotting_flesh','hair_length','has_soul']

y_variable = 'int_type'

color_dummy = list(train_data.color.drop_duplicates())

for c in color_dummy:

    col_name = c + '_dummy'

    x_variable.append(col_name)

    train_data[col_name] = 0

    train_data.loc[train_data.color == c, col_name] = 1

print(train_data.head())

print(x_variable)

X = train_data[x_variable].as_matrix()

Y = train_data[y_variable].as_matrix()

from sklearn import model_selection 

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = 0.33)

from sklearn import linear_model

from sklearn import metrics

for c in [0.001,0.01,0.1,1,10,100,1000,10000,100000]:

    logreg = linear_model.LogisticRegression(C = c)

    logreg.fit(X_train,Y_train)

    predicted = logreg.predict(X_test)

    accu = metrics.accuracy_score(Y_test,predicted)

    print(" accuracy for c = " + str(c) + " is " + str(accu) + "\n")

   
