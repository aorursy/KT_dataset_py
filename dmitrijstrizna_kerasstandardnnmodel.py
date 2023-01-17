import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



np.random.seed(33)



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



# Feature engineering

# 1. Sex to 0 and 1

m = {'m' : 1, 'f' : 0}

train['Sex'] = train['Sex'].str[0].map(m)

test['Sex'] = test['Sex'].str[0].map(m)



# 2. Adding predicate (Miss/Mr/etc) feature

p_tr = train['Name'].str.split(',').map(lambda x: x[1]).str.strip().str.split('.').map(lambda x: x[0])



mapdict = {}

for i,v in enumerate(list(p_tr.unique())):

    mapdict[v] = i

train = pd.concat([train,p_tr.map(mapdict).rename('P')], axis=1)



# Need to use same mapdict

p_ts = test['Name'].str.split(',').map(lambda x: x[1]).str.strip().str.split('.').map(lambda x: x[0])

test = pd.concat([test,p_ts.map(mapdict).rename('P')], axis=1)



# 4. Kids / Adults

m = {'child' : 0, 'adult' : 1}

train["AgeCat"]= pd.cut(train["Age"], bins=[0,15,max(train["Age"]+1)], labels=['child','adult']).map(m).fillna(1).astype(int)

test["AgeCat"]= pd.cut(test["Age"], bins=[0,15,max(test["Age"]+1)], labels=['child','adult']).map(m).fillna(1).astype(int)





# Preparing data to feed into NN

Y = train["Survived"]



#features = ["Pclass", "Sex", "Fare", "Age", "SibSp", "Parch", "P", "AgeCat"]

features = ["Pclass", "Sex", "Fare", "SibSp", "Parch", "P", "AgeCat"]



X = train[features]

X_test = test[features]



def normalize_dataframe(X):

    #mean = np.array(df.describe().T['mean'])

    #STD = np.array(df.describe().T['std'])

        

    mean = np.nanmean(X, axis=0)

    

    # Fill nan values with mean value

    inds = np.where(np.isnan(X))

    Xa = np.array(X)

    Xa[inds] = np.take(mean, inds[1])

    

    std = np.std(Xa, axis=0)

    

    norm_X = (Xa-mean)/std

    

    return norm_X



norm_X = np.array(normalize_dataframe(X))

norm_X_test = np.array(normalize_dataframe(X_test))
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent")

X_nona = imp.fit_transform(X)

X_test_nona = imp.transform(X_test)
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import BinaryCrossentropy
def model(input_shape,

         hidden_init = 'lecun_normal',

         hidden_activation = 'selu'):

    X_input = Input(shape=input_shape)

    X = Dense(64, kernel_initializer=hidden_init)(X_input)

    X = BatchNormalization()(X)

    X = Activation(hidden_activation)(X)

    #X = Dropout(0.7)(X)

    X = Dense(64,kernel_initializer=hidden_init)(X)

    #X = BatchNormalization()(X)

    X = Activation(hidden_activation)(X)

    #X = Dropout(0.7)(X)

    X = Dense(64, kernel_initializer=hidden_init)(X)

    X = BatchNormalization()(X)

    X = Activation(hidden_activation)(X)

    X = Dropout(0.7)(X)

    X = Dense(10, kernel_initializer=hidden_init)(X)

    X = Dense(1, kernel_initializer=hidden_init, activation = 'sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model



model = model(X.shape[1])

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss=BinaryCrossentropy(from_logits=True),

                metrics=['accuracy'])



# Feed Input() and outputs= data

import tensorflow.keras.backend as K

K.clear_session()

import tensorflow as tf

tf.random.set_seed(33)

#model.fit(norm_X, Y, epochs=50, batch_size=30)

model.fit(X_nona, Y, epochs=50, batch_size=30)



# Feed Input() data

#predictions = model.predict(norm_X_test)

predictions = model.predict(X_test_nona)

predictions = np.greater(predictions,0.5).astype(int)
# Write Output

prS = pd.Series(predictions.reshape(predictions.shape[0]))

output = pd.concat([test['PassengerId'],prS], axis=1)

output.columns=['PassengerId','Survived']

output.head()

output.to_csv('KerasNN_submission.csv', index=False)

print("Your submission was successfully saved!")