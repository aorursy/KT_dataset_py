# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/mushrooms.csv')

df.head()
label_encoder = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = label_encoder.fit_transform(df[col])



df.head()

y = df['class'].values  # now numpy array (matrix)

x = df.drop('class',1)  # ?? is this still a dataframe?

# print(y)

# print(x)

print(len(y))

print(len(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1337)

print(x_train.shape)

print(y_train.shape)
models = [DecisionTreeClassifier(),

         SVC(kernel='rbf', random_state=0),

         SVC(kernel='linear', random_state=0),

         LogisticRegression()

         ]



model_names = ['DecisionTree', 'SVC - ', 'SVC_linear', 'Logistic Regression']

for i, model in enumerate(models):

    model.fit(x_train, y_train)

    print('The accuracy of ' + model_names[i] + ' is ' + str(accuracy_score(y_test, model.predict(x_test))) )