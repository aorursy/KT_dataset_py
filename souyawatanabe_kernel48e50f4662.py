# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy  # linear algebra

import pandas# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pandas.read_csv('/kaggle/input/titanic/train.csv')

df_train = df_train.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

df_train.head()
df_train = df_train.dropna(subset=['Age']).reset_index(drop=True)

len(df_train)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_train['Sex'] = encoder.fit_transform(df_train['Sex'].values)

df_train.head()
from sklearn.preprocessing import StandardScaler



# 標準化

# Standardize numbers

standard = StandardScaler()

df_train_std = pandas.DataFrame(standard.fit_transform(df_train.loc[:, ['Age', 'Fare']]), columns=['Age', 'Fare'])



# Age を標準化

# Standardize Age

df_train['Age'] = df_train_std['Age']



# Fare を標準化

# Standardize Fare

df_train['Fare'] = df_train_std['Fare']



df_train.head()
relationship= {'Pclass', 'Sex', 'Age', 'Fare'}

w = np.random.rand(len(relationship)+1)

learning_rate = 0.01

epoch = 100

print(type(w))
X = df_train.loc[:, ['Pclass', 'Sex', 'Age', 'Fare']].values

Y = df_train.loc[:, ['Survived']].values

Y = numpy.reshape(Y,(-1))

len(X)

len(Y)
def perceptron(w,x):

    xT = np.array(x,ndmin=2).T

    h = np.dot(w[:4], xT)+w[4]

    return 1 * ( h > 0)
Y_pred = perceptron(w,X)

print(Y_pred)

confusion_matrix(Y, Y_pred)


for i in range(epoch):

    for j in range(len(Y)):

        error = Y[j] - perceptron(w, X[j])

        for k in range(len(w)):

            if k==4:

                w[k] += error*learning_rate*1

            else:

                w[k] += error*learning_rate*X[j, k]

from sklearn.metrics import confusion_matrix

Y_pred = perceptron(w,X)

print(Y_pred)

confusion_matrix(Y, Y_pred)
# test.csvを読み込む

# Load test.csv

df_test = pandas.read_csv('/kaggle/input/titanic/test.csv')



# 'PassengerId'を抽出する(結果と結合するため)

df_test_index = df_test.loc[:, ['PassengerId']]



# 'Survived', 'Pclass', 'Sex', 'Age', 'Fare'を抽出する

# Extract 'Survived', 'Pclass', 'Sex', 'Age', 'Fare'

df_test = df_test.loc[:, ['Pclass', 'Sex', 'Age', 'Fare']]



# 性別をLabelEncoderを利用して数値化する

# Digitize gender using LabelEncoder

df_test['Sex'] = encoder.transform(df_test['Sex'].values)



df_test_std = pandas.DataFrame(standard.transform(df_test.loc[:, ['Age', 'Fare']]), columns=['Age', 'Fare'])



# Age を標準化

# Standardize Age

df_test['Age'] = df_test_std['Age']



# Fare を標準化

# Standardize Fare

df_test['Fare'] = df_test_std['Fare']



# Age, Fare のNanを0に変換

# Convert Age and Fare Nan to 0

df_test = df_test.fillna({'Age':0, 'Fare':0})



df_test.head()
x_test = df_test.values

y_test = perceptron(w, x_test)

print(y_test)

df_output = pandas.concat([df_test_index, pandas.DataFrame(y_test, columns=['Survived'])], axis=1)

df_output.to_csv('result.csv', index=False)
