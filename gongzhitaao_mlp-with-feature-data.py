import time



import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.callbacks import ModelCheckpoint





validation_split = 0.7



print('Loading training data')

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print('Encode species')

le = preprocessing.LabelEncoder()

df_train['label'] = le.fit_transform(df_train['species'].values)



if validation_split:

    print('Split training data set')

    df = df_train

    grp = df.groupby('label')

    df_train = grp.apply(lambda x: x.sample(frac=0.7))

    df_train.reset_index(level='label', drop=True, inplace=True)

    df_val = df.drop(df_train.index.values)

    df_train.reset_index(inplace=True, drop=True)

    df_val.reset_index(inplace=True, drop=True)



lb = preprocessing.LabelBinarizer()

y_train = lb.fit_transform(df_train['species'].values)

if validation_split:

    y_val = lb.transform(df_val['species'].values)



print('Extract margin feature')

s1 = preprocessing.StandardScaler()

X1_train = s1.fit_transform(df_train.loc[:,'margin1':'margin64'])

X1_test = s1.transform(df_test.loc[:,'margin1':'margin64'])

if validation_split:

    X1_val = s1.transform(df_val.loc[:,'margin1':'margin64'])



print('Extract shape feature')

s2 = preprocessing.StandardScaler()

X2_train = s2.fit_transform(df_train.loc[:,'shape1':'shape64'])

X2_test = s2.transform(df_test.loc[:,'shape1':'shape64'])

if validation_split:

    X2_val = s2.transform(df_val.loc[:,'shape1':'shape64'])



print('Extract texture feature')

s3 = preprocessing.StandardScaler()

X3_train = s3.fit_transform(df_train.loc[:,'texture1':'texture64'])

X3_test = s3.transform(df_test.loc[:,'texture1':'texture64'])

if validation_split:

    X3_val = s3.transform(df_val.loc[:,'texture1':'texture64'])



X_train = np.hstack((X1_train, X2_train, X3_train))

X_test = np.hstack((X1_test, X2_test, X3_test))

if validation_split:

    X_val = np.hstack((X1_val, X2_val, X3_val))



print('Build MLP classifier')

model = Sequential()

model.add(Dense(100, input_dim=X_test.shape[1]))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(len(lb.classes_)))

model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',

              metrics=['accuracy'])



start = time.time()

print('Training...')

model.fit(X_train, y_train, nb_epoch=1000, batch_size=32, verbose=False)

model.save('model.h5')

print('Elapsed {0:.2f}s'.format(time.time()-start))



print('Predicting...')

if validation_split:

    score = model.evaluate(X_val, y_val, batch_size=16)

    print('\nloss {0:.4f}, acc {1:.4f}'.format(score[0], score[1]))

else:

    ybar = model.predict_proba(X_test)

    df_out = pd.DataFrame(ybar)

    df_out.columns = lb.classes_

    df_out.insert(0, 'id', df_test['id'])



    print('Saving result')

    df_out.to_csv('res2.csv', index=False)