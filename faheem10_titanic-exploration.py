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
df = pd.read_csv('../input/train.csv')
df.head()
# df['alone'] = df.SibSp + df.Parch > 0
df['Name'] = df.Name.map(lambda x: x.split(",")[1].split(".")[0])
# df.Title.value_counts()
df.head()
df.columns
df[['Sex', 'Embarked', 'Name', 'Ticket']].describe()
len(df['Fare'].unique())
df.columns
X = df[[ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Fare', 

       'Parch', 'Embarked']]

y = df['Survived']
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
numeric = ['Fare', 'Age', 'SibSp', 'Parch']

numeric_pipeline = Pipeline(steps = [('imp', SimpleImputer(strategy = 'mean')), 

                             ('scale', StandardScaler())])

ctg = ['Sex', 'Pclass', 'Embarked', 'Name']

categorical_pipeline = Pipeline(steps = [('imp', SimpleImputer(strategy = 'constant', fill_value = 'missing')), 

                             ('enc', OneHotEncoder(handle_unknown = 'ignore'))])
from sklearn.compose import ColumnTransformer
preprocessing = ColumnTransformer(transformers = [

    ('num', numeric_pipeline, numeric),

    ('cat', categorical_pipeline, ctg)

] )
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

preprocessing.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size = 0.2, random_state = 0)
X_transformed = preprocessing.transform(X)
X_transformed.shape
preprocessing.transform(X_train[:2]).shape

from sklearn.decomposition import PCA



X_reduced = PCA(2).fit_transform((X_transformed.toarray()))
X_reduced.shape
import matplotlib.pyplot as plt

plt.scatter(X_reduced[:,0], X_reduced[:,1], c = y, cmap = plt.cm.Set1)

plt.show()
from keras.models import Sequential

from keras.layers import Dense

# from TensorBoardColab import *

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(2, activation = 'sigmoid', input_dim = (X_transformed.shape[1]), kernel_initializer = 'truncated_normal'))

# model.add(Dense(30, activation = 'sigmoid', kernel_initializer = 'truncated_normal'))

# model.add(Dense(128, activation = 'tanh'))

# model.add(Dense(10, activation = 'tanh'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])

model.summary()
es = EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit(X_train, y_train, batch_size= 32, epochs = 100,

            validation_split = 0.2)
plt.plot(history.epoch, history.history['val_loss'], c = 'red')

plt.plot(history.epoch, history.history['loss'], c = 'black')

plt.show()
clf = Pipeline(steps = [

    ('preprocessing', preprocessing),

    ('classifier', LogisticRegression())

])
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
test = pd.read_csv('../input/test.csv')
xt = test[['Sex', 'Pclass', 'Age', 'Fare', 'Embarked']]
y_pred = clf.predict(xt)
submission = pd.DataFrame({'PassengerId': test['PassengerId'],

             'Survived': y_pred})
submission.to_csv('submission.csv', index = False)