# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import tensorflow as tf

import random as rn

import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)

rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from xgboost import XGBClassifier



from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD, Adam

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, re

files = os.listdir('../input/titanic')

print(files)



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/titanic/'+files[0], engine='python')

df_test = pd.read_csv('../input/titanic/'+files[2], engine='python')

df = df_train.copy()
df['Title'] = df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
Title_Dictionary = {

        "Capt":       "Other",

        "Col":        "Other",

        "Major":      "Other",

        "Dr":         "Mr",

        "Rev":        "Other",

        "Jonkheer":   "Other",

        "Don":        "Mr",

        "Sir" :       "Mr",

        "the Countess":"Other",

        "Dona":       "Miss",

        "Lady" :      "Miss",

        "Mme":        "Miss",

        "Ms":         "Miss",

        "Mrs" :       "Miss",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Mr"

                   }



df['Title'] = df.Title.map(Title_Dictionary)

df_test['Title'] = df_test.Title.map(Title_Dictionary)
# Train

train_y = df['Survived']

df['Sex'] = df['Sex'].astype('category').cat.codes

df['Embarked'] = df["Embarked"].replace('s', 'S')

df['Embarked'] = df["Embarked"].replace('c', 'C')

df['Embarked'] = df["Embarked"].replace('q', 'Q')

df['Embarked'] = df['Embarked'].astype('category').cat.codes

df['Title'] = df['Title'].astype('category').cat.codes



df['Cabin'] = df['Cabin'].fillna("")

df['Family_Members'] = df['Parch'] + df['SibSp']

df['Family_Numerous'] = ((df['Family_Members'] > 1) & (df['Family_Members'] < 5)).astype(int)

df['IsChild'] = (df['Age'] < 5.0).astype(int)

df['IsAlone'] = np.where((df['SibSp'] == 0) & (df['Parch'] == 0), 1, 0)

df['Age'] = df['Age'].fillna(28.80)
df.isna().sum()
df = df.select_dtypes(exclude=['object'])

df = df.drop(["PassengerId", "Survived", "Family_Members"], axis=1)

train_x = df
#Test

df_test['Sex'] = df_test['Sex'].astype('category').cat.codes



df_test['Embarked'] = df_test["Embarked"].replace('s', 'S')

df_test['Embarked'] = df_test["Embarked"].replace('c', 'C')

df_test['Embarked'] = df_test["Embarked"].replace('q', 'Q')

df_test['Embarked'] = df_test['Embarked'].astype('category').cat.codes

df_test['Title'] = df_test['Title'].astype('category').cat.codes

df_test['Fare'] = df_test['Fare'].fillna(30)

df_test['Cabin'] = df_test['Cabin'].fillna("")

df_test['Family_Members'] = df_test['Parch'] + df_test['SibSp'] 

df_test['Family_Numerous'] = ((df_test['Family_Members'] > 1) & (df_test['Family_Members'] < 5)).astype(int)

df_test['IsChild'] = (df_test['Age'] < 5.0).astype(int)

df_test['IsAlone'] = np.where((df_test['SibSp'] == 0) & (df_test['Parch'] == 0), 1, 0)

df_test['Age'] = df_test['Age'].fillna(28.80)
df_test.isna().sum()
df_test = df_test.select_dtypes(exclude=['object'])

df_test = df_test.drop(["PassengerId", "Family_Members"], axis=1)

test_x = df_test
sc = StandardScaler()

train_x = pd.DataFrame(sc.fit_transform(train_x.values), index=train_x.index, columns=train_x.columns)

test_x = pd.DataFrame(sc.fit_transform(test_x.values), index=test_x.index, columns=test_x.columns)

train_x.head()
model = Sequential()

model.add(Dense(30, 

                activation='relu',  

                input_dim=df.shape[1],

                kernel_initializer='uniform'))

model.add(Dropout(0.3))

model.add(Dense(60,

                kernel_initializer='uniform',

                activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(150,

                kernel_initializer='uniform',

                activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1,

                kernel_initializer='uniform',

                activation='sigmoid'))

model.summary()



sgd = SGD(lr = 0.03, momentum = 0.95)

adam  = Adam(lr=0.0007, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compiling our model

model.compile(optimizer = adam, 

                   loss = 'binary_crossentropy', 

                   metrics = ['accuracy'])

#optimizers list

#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
history = model.fit(train_x, train_y, validation_split=0.25, epochs=150, batch_size=30, verbose=1, shuffle=False)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('Epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
my_model = XGBClassifier()

my_model.fit(train_x, train_y, verbose=False)
# XGBoost

predictions = my_model.predict(test_x)
# Keras NN



#predictions = model.predict(test_x)

#predictions = (predictions > 0.5).astype('int8')

#print(predictions[:7])
df_final = pd.read_csv('../input/titanic/'+files[1])

df_final['Survived'] = predictions.astype('int8')

print(df_final.head(10))



ones = list(df_final[df_final['Survived'] == 1]['PassengerId'])

print('', len(ones),'people survived out of a total of', df.shape[0], 'being the', len(ones)/df.shape[0]*100,'% of the dataset')
sns.FacetGrid(df_train, col='Survived',row='Pclass').map(sns.distplot,'Fare')

"""

sns.catplot("Family_Numerous", data=df_train, aspect=2.0, kind='count',

                       hue='Survived')

"""

sns.catplot("Pclass", data=df_train, aspect=1.0, kind='count',

                       hue='Survived')



sns.catplot("Sex", data=df_train, aspect=1.0, kind='count',

                       hue='Survived')



sns.catplot("Embarked", data=df_train, aspect=1.0, kind='count',

                       hue='Survived')



sns.FacetGrid(df_train, col='Survived',row='Sex').map(sns.distplot,'Age')



df_surv_male = []

df_final.to_csv('submission.csv', index=False)