import math

import numpy as np

import pandas as pd

from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential

from sklearn.preprocessing import StandardScaler



np.random.seed(7)



root_path = '../input'





def get_data(filepath):

    df = pd.read_csv(filepath)

    return get_data_sets(df)





def get_data_sets(df):

    df['Sex'] = df['Sex'].apply(lambda s: 0 if s == 'male' else 1)

    df['Age'] = df['Age'].apply(lambda a: df['Age'].median() if math.isnan(a) else a)

    features = ['Pclass', 'Sex', 'Age', 'SibSp']

    x = StandardScaler().fit_transform(df[features].values)

    y = []

    try:

        y = pd.get_dummies(df['Survived']).values

    finally:

        return x, y





def get_model(x):

    m = Sequential()

    m.add(Dense(input_dim=x.shape[1], output_dim=50, activation='relu'))

    m.add(Dropout(0.3))

    m.add(Dense(output_dim=50, activation='relu'))

    m.add(Dropout(0.3))

    m.add(Dense(output_dim=2, activation='softmax'))

    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return m



X, Y = get_data(root_path + '/train.csv')

X_test, Y_test = get_data(root_path + '/test.csv')



model = get_model(X)

model.fit(X, Y, nb_epoch=10, batch_size=10, verbose=1)



predictions = model.predict(X_test, batch_size=10)

survived = [int(round(p)) for p in predictions[:, 1]]



df = pd.read_csv(root_path + '/test.csv')

passenger_ids = df['PassengerId']

submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': survived})

submission.to_csv("StackingSubmission.csv", index=False)