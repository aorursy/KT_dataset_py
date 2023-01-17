# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/learn-together/train.csv")

df.head()
df.describe()
df.Cover_Type.value_counts()
# https://www.kaggle.com/kwabenantim/forest-cover-feature-engineering

def add_features(X_):

    X = X_.copy()



    X['Hydro_Elevation_diff'] = (X['Elevation'] - 

                                 X['Vertical_Distance_To_Hydrology'])



    X['Hydro_Euclidean'] = (X['Horizontal_Distance_To_Hydrology']**2 +

                            X['Vertical_Distance_To_Hydrology']**2).apply(np.sqrt)



    X['Hydro_Fire_sum'] = (X['Horizontal_Distance_To_Hydrology'] + 

                           X['Horizontal_Distance_To_Fire_Points'])



    X['Hydro_Fire_diff'] = (X['Horizontal_Distance_To_Hydrology'] - 

                            X['Horizontal_Distance_To_Fire_Points']).abs()



    X['Hydro_Road_sum'] = (X['Horizontal_Distance_To_Hydrology'] +

                           X['Horizontal_Distance_To_Roadways'])



    X['Hydro_Road_diff'] = (X['Horizontal_Distance_To_Hydrology'] -

                            X['Horizontal_Distance_To_Roadways']).abs()



    X['Road_Fire_sum'] = (X['Horizontal_Distance_To_Roadways'] + 

                          X['Horizontal_Distance_To_Fire_Points'])



    X['Road_Fire_diff'] = (X['Horizontal_Distance_To_Roadways'] - 

                           X['Horizontal_Distance_To_Fire_Points']).abs()

    

    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?

    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 

                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 

                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 

                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    

    # Compute Soil_Type number from Soil_Type binary columns

    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    

    # Replace Soil_Type number with "stoneyness" value

    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)

    

    rocks = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1,

             1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 

             0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 

             0, 1, 1, 1, 1, 1, 1, 0, 0, 1]

    

    X['Rocks'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    X['Rocks'] = X['Rocks'].replace(range(1, 41), rocks)

    

    return X
df = add_features(df)

df.head()
df.iloc[:, 1:11]

df.iloc[:, 56:66]

df_to_scal = pd.concat([df.iloc[:, 1:11], df.iloc[:, 56:66]], axis=1)
from sklearn.preprocessing import RobustScaler

sc = RobustScaler()



sc.fit(df_to_scal)

X = sc.transform(df_to_scal)



headers = df_to_scal.columns

X_scal = pd.DataFrame(X, columns=headers)

X_scal.head()



train_scal = pd.concat([X_scal, df.iloc[:, 11:56]], axis=1)

train_scal.head()



from sklearn.model_selection import train_test_split

y = train_scal.loc[:,'Cover_Type']

X = train_scal.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1, random_state=0)
X_train
import keras

print(keras.__version__)
from keras import models

from keras import layers

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(1 * 64,)))

network.add(layers.Dense(512, activation='relu', input_shape=(1 * 512,)))

#network.add(layers.Dense(512, activation='relu', input_shape=(1 * 512,)))

#network.add(layers.Dense(512, activation='relu', input_shape=(1 * 512,)))

#network.add(layers.Dense(32, activation='relu', input_shape=(1 * 512,)))

network.add(layers.Dense(8, activation='softmax'))
network.compile(optimizer='rmsprop',

loss='categorical_crossentropy',

metrics=['accuracy'])
from keras import optimizers

network.compile(optimizer=optimizers.RMSprop(lr=0.001),

loss='mse',

metrics=['accuracy'])
from keras.utils import to_categorical

train_labels = to_categorical(y_train)

test_labels = to_categorical(y_test)
network.fit(X_train, train_labels, epochs=50, batch_size=128)
test_loss, test_acc = network.evaluate(X_test, test_labels)

print('test_acc:', test_acc)



#0.8558201059252286
df = pd.read_csv("/kaggle/input/learn-together/test.csv")

df.head()
df.iloc[:, 55:65]
df = add_features(df)



df.iloc[:, 1:11]

df.iloc[:, 55:65]

df_to_scal = pd.concat([df.iloc[:, 1:11], df.iloc[:, 55:65]], axis=1)



X = sc.transform(df_to_scal)



headers = df_to_scal.columns

X_scal = pd.DataFrame(X, columns=headers)

X_scal.head()



test_scal = pd.concat([X_scal, df.iloc[:, 11:55]], axis=1)

test_scal.head()
test_pred = network.predict_classes(test_scal)
sample_submission = pd.read_csv('../input/learn-together/sample_submission.csv')
sample_submission['Cover_Type'] = test_pred

sample_submission.to_csv('submission.csv', index=False)