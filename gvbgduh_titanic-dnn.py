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



df_train = pd.read_csv('../input/train.csv')

df_train.info()

df_test = pd.read_csv('../input/test.csv')

df_test.info()
def prepare_data(df_raw, test=False):

    """

    Preprocess data

    """

    df = df_raw.copy()

    # Categorize Pclass

    df.Pclass = df.Pclass.astype('category')

    print('Pclass categories: ', df.Pclass.cat.categories)

    # Name

    titles = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False).astype('category')

    cats = df['Title'].value_counts()[:9]  # [df['Title'].value_counts() > 1]

    df['Title'] = pd.Categorical(titles, categories=[x for x, y in cats.items()])

    print('Title categories: ', df.Title.cat.categories)

    # Categorize Sex

    df.Sex = df.Sex.astype('category')

    print('Sex categories: ', df.Sex.cat.categories)

    # print('Sex categories: ', dict(enumerate(df_train.Sex.cat.categories)))

    # Fill NAN for Age using mean

    df.Age.fillna(df.Age.mean(), inplace=True)

    # Categorize SibSp

    df.SibSp = df.SibSp.astype('category')

    print('SibSp categories: ', df.SibSp.cat.categories)

    # Categorize Parch

    df.Parch = pd.Categorical(df.Parch, categories=range(10))

    print('Parch categories: ', df.Parch.cat.categories)

    # Ticket

    tick = df_train.Ticket.str.extract('(\d+)', expand=False)

    tick_start_with = tick.apply(lambda x: str(x)[:1])

    tick_num_lenght = tick.apply(lambda x: len(str(x)))

    df['TickStartWith'] = tick_start_with.astype('category')

    df['TickNumLength'] = tick_num_lenght.astype('category')

    print('TickStartWith categories: ', df.TickStartWith.cat.categories)

    print('TickNumLength categories: ', df.TickNumLength.cat.categories)

    # Fare

    df.Fare.fillna(df.Fare.mean(), inplace=True)

    # Cabin

    cab_char = df_train.Cabin.str.extract('([A-Za-z]{1})', expand=False)

    df['CabChar'] = cab_char.astype('category')

    print('CabChar categories: ', df.CabChar.cat.categories)

    # Categorize Embarked

    df.Embarked = df.Embarked.astype('category')

    df.Embarked.fillna(df.Embarked.value_counts().idxmax(), inplace=True)

    print('Embarked categories: ', df.Embarked.cat.categories)

    

    if test:

        Y_train = df.PassengerId

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    else:

        Y_train = df.Survived

        df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    df.info()

    return Y_train, df



def dummify(some_df):

    df = some_df.copy()

    df_d = pd.get_dummies(df, columns=['Pclass', 'Sex', 'SibSp', 

                                       'Parch', 'Embarked', 'Title', 

                                       'TickStartWith', 'TickNumLength', 'CabChar'])

    df_d.info()

    return df_d



Y_train, X_train = prepare_data(df_train)

Y_test, X_test = prepare_data(df_test, test=True)

# print('Dummifying...')

# X_d_train = dummify(X_train)

# X_d_test = dummify(X_test)
from keras.layers import Dense, Dropout

from keras.utils import to_categorical, normalize

from keras.models import Sequential

from sklearn.model_selection import train_test_split





x_tr, x_t, y_tr, y_t = train_test_split(X_train, Y_train, test_size=0.3, random_state=17)



def shape_data(some_df, norm=True):

    if norm:

        # age = normalize(X_train.Age.values.reshape(some_df.shape[0],1))

        # fare = normalize(X_train.Fare.values.reshape(some_df.shape[0],1))

        age = some_df.Age.values.reshape(some_df.shape[0],1)

        fare = some_df.Fare.values.reshape(some_df.shape[0],1)

        m_age, s_age = age.mean(), age.std()

        m_fare, s_fare = fare.mean(), fare.std()

        age = (age - m_age) / s_age

        fare = (fare - m_fare) / s_fare

    else:

        age = some_df.Age.values.reshape(some_df.shape[0],1)

        fare = some_df.Fare.values.reshape(some_df.shape[0],1)

    cat_sex = to_categorical(some_df.Sex.cat.codes)

    pclass = to_categorical(some_df.Pclass.cat.codes)

    sibsp = to_categorical(some_df.SibSp.cat.codes)

    parch = to_categorical(some_df.Parch.cat.codes, num_classes=10)

    embarked = to_categorical(some_df.Embarked.cat.codes)

    title = to_categorical(some_df.Title.cat.codes)

    TickStartWith = to_categorical(some_df.TickStartWith.cat.codes)

    TickNumLength = to_categorical(some_df.TickNumLength.cat.codes)

    CabChar = to_categorical(some_df.CabChar.cat.codes, num_classes=8)

    return np.concatenate([age, fare, cat_sex, pclass, 

                           sibsp, parch, embarked, title,

                           TickStartWith, TickNumLength, CabChar], axis=1)



x, y = shape_data(x_tr, norm=True), y_tr.values.reshape(y_tr.shape[0], 1)

x_test, y_test = shape_data(x_t, norm=True), y_t.values.reshape(y_t.shape[0], 1)

print('x.shape: ', x.shape)

print('y.shape: ', y.shape)

print('x_test.shape: ', x.shape)

print('y_test.shape: ', y.shape)



model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=x.shape[1]))

model.add(Dropout(0.5))

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.7))

model.add(Dense(units=32, activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(units=1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x, y, epochs=700, batch_size=1024)



loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1024)

# classes = model.predict(x_test, batch_size=40)

print(loss_and_metrics)

# print(classes, y_test)

# print(classes.shape)

# cl = pd.Series(list(classes))

# yt = pd.Series(list(y_test))

# ddd = pd.DataFrame(data={'classes': list(classes), 'y': list(y_test)}, index=x_t.index)

# clss = classes >= 0.5

# ddd = pd.concat([y_tr, pd.Series(classes.reshape(-1)), pd.Series(clss.reshape(-1))], axis=1)

# print(ddd)



print('==========================================')



# Y_train, X_train

# X_test, Y_test



# Let's try train model with full avail data and try to predict

# x_train, y_train = shape_data(X_train, norm=True), Y_train.values.reshape(Y_train.shape[0], 1)

# x_pred = shape_data(X_test, norm=True)



# model.fit(x_train, y_train, epochs=43, batch_size=1000)



# classes = model.predict(x_pred, batch_size=40)

# # print(classes)

# pred = pd.Series(classes.reshape(-1), name='Survived')

# pred = pred.apply(lambda x: int(x >= 0.45))

# ans = pd.concat([Y_test, pred], axis=1)

# ans = ans.set_index('PassengerId')

# ans.to_csv('ans_dnn.csv')

# print(ans)

res = """

For 1 HL net

For Age, Fare, Sex

Epoch 25/25

623/623 [==============================] - 0s 50us/step - loss: 0.4663 - acc: 0.8010

268/268 [==============================] - 0s 1ms/step

[0.52375767124232964, 0.75746268745678569]



+ Pclass

Epoch 25/25

623/623 [==============================] - 0s 50us/step - loss: 0.4105 - acc: 0.8170

268/268 [==============================] - 0s 1ms/step

[0.47460541529441946, 0.7574626865671642]



+ 2 Layers without Pclass

Epoch 25/25

623/623 [==============================] - 0s 57us/step - loss: 0.4443 - acc: 0.8090

268/268 [==============================] - 0s 1ms/step

[0.50794072382485689, 0.772388058811871]



+ Pclass

Epoch 25/25

623/623 [==============================] - 0s 59us/step - loss: 0.3729 - acc: 0.8363

268/268 [==============================] - 0s 1ms/step

[0.46900954531199895, 0.78358208955223885]



+ SibSp

Epoch 25/25

623/623 [==============================] - 0s 57us/step - loss: 0.3461 - acc: 0.8604

268/268 [==============================] - 0s 2ms/step

[0.47837811797412472, 0.77611940387469625]

# Note: Looks like var is increasing



+ Embarked

Epoch 25/25

623/623 [==============================] - 0s 61us/step - loss: 0.3206 - acc: 0.8700

268/268 [==============================] - 0s 2ms/step

[0.52090917772321554, 0.76119403074036784]

# Note: Looks like var is increasing even more



+ Title

Epoch 25/25

623/623 [==============================] - 0s 61us/step - loss: 0.2913 - acc: 0.8764

268/268 [==============================] - 0s 2ms/step

[0.54417172207761166, 0.78358209044186033]



+ TickStartWith

Epoch 25/25

623/623 [==============================] - 0s 59us/step - loss: 0.2600 - acc: 0.8812

268/268 [==============================] - 0s 2ms/step

[0.56602454808220937, 0.75746268745678569]



+ TickNumLength

Epoch 25/25

623/623 [==============================] - 0s 59us/step - loss: 0.2399 - acc: 0.9053

268/268 [==============================] - 0s 2ms/step

[0.57680059902703584, 0.78358208955223885]



+ CabChar

Epoch 25/25

623/623 [==============================] - 0s 63us/step - loss: 0.1974 - acc: 0.9230

268/268 [==============================] - 1s 2ms/step

[0.55896452558574394, 0.79477611940298509]



+ Parch

Epoch 25/25

623/623 [==============================] - 0s 63us/step - loss: 0.1904 - acc: 0.9213

268/268 [==============================] - 1s 2ms/step

[0.60053732146078076, 0.78731343283582089]



+ Reg: Dropout + 1 more L

Dropout: 0.6

Epoch 43/43

623/623 [==============================] - 0s 92us/step - loss: 0.3944 - acc: 0.8427

268/268 [==============================] - 1s 3ms/step

[0.44395681637436596, 0.82089552238805974]



Dropout: 0.7 (-1)

Epoch 43/43

623/623 [==============================] - 0s 81us/step - loss: 0.3145 - acc: 0.8876

268/268 [==============================] - 1s 3ms/step

[0.48667496709681268, 0.82089552238805974]



Some very rough tunning

Epoch 43/43

623/623 [==============================] - 0s 88us/step - loss: 0.4723 - acc: 0.8363

268/268 [==============================] - 1s 5ms/step

[0.44081404582778022, 0.82089552416730283]





"""