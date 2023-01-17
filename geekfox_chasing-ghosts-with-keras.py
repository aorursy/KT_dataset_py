# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential



from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
dummy_color = pd.get_dummies(train_df['color'].astype('category'))

dummy_color = dummy_color.add_prefix('{}#'.format('color'))

train_df.drop('color',

          axis = 1,

          inplace = True)

train_df = train_df.join(dummy_color)

train_ids = train_df['id']

train_labels = train_df['type']

train_df.drop('id',

          axis = 1,

          inplace = True)

train_df.drop('type',

          axis = 1,

          inplace = True)
train_values = train_df.values

print(train_values)

print(train_labels)
encoder = LabelEncoder()

encoder.fit(train_labels)

encoded_Y = encoder.transform(train_labels)

# convert integers to dummy variables (i.e. one hot encoded)

train_classes = np_utils.to_categorical(encoded_Y)

print(train_classes)
def base_model():

    # create model

    model = Sequential()

    model.add(Dense(256, input_dim=10, init='normal', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(128, init='normal', activation='relu'))

    model.add(Dropout(0.2))

    

    model.add(Dense(3, init='normal', activation='sigmoid'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
estimator = KerasClassifier(build_fn=base_model, nb_epoch=30, batch_size=5, verbose=0)

kfold = KFold(n_splits=2, shuffle=True)

split = ShuffleSplit(n_splits=2, train_size=0.8)

estimator.fit(train_values, train_classes)

results = cross_val_score(estimator, train_values, train_classes, cv=split)

print("Base model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
test_ids = test_df['id']

test_dummy_color = pd.get_dummies(test_df['color'].astype('category'))

test_dummy_color = test_dummy_color.add_prefix('{}#'.format('color'))

test_df.drop('color',

          axis = 1,

          inplace = True)

test_df = test_df.join(dummy_color)

test_df.drop('id',

          axis = 1,

          inplace = True)



test_values = test_df.values

print(test_values)
pred = estimator.predict(test_values, batch_size=5)
predict_data = pd.DataFrame(test_ids).join(pd.DataFrame(pred))

predict_data = predict_data.replace({0:{0: 'Ghost',

                                 1: 'Goblin',

                                 2: 'Ghoul'}})

predict_data.columns = ['id', 'type']

predict_data
predict_data.to_csv("submission.csv", index=False)