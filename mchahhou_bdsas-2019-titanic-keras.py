import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler



from keras.layers import Input, Dense, Activation, Embedding, concatenate, Flatten, Dropout

from keras.regularizers import l1, l2

from keras.models import Model



from keras.optimizers import SGD



from keras.callbacks import EarlyStopping, ModelCheckpoint

df = pd.read_csv('../input/train.csv')
df.head()
df = df[ ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked'] ]

df['Age'].fillna(df.Age.mean(), inplace=True)

df['Fare'].fillna(df.Fare.mean(), inplace=True)

df['Sex'] = LabelEncoder().fit_transform(  df['Sex'] )

df['Embarked'] = LabelEncoder().fit_transform(  df['Embarked'].fillna('unknown') )





for f in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare']:

    df[f] = StandardScaler().fit_transform( df[f].values.reshape(-1,1)  )



df.isnull().sum()
train = df[:700].reset_index(drop=True)

test  = df[700:].reset_index(drop=True)

print(train.shape, test.shape)

train_y = train.Survived.values

train_X = train.drop('Survived', axis=1)

test_y = test.Survived.values

test_X = test.drop('Survived', axis=1)

print(train_X.shape, test_X.shape)





def prepare_data(df):

    X = dict()

    X['embedding'] = df['Embarked'].values

    X['numeric'] = df.drop('Embarked', axis=1).values

    return X



train_keras = prepare_data(train_X)

test_keras = prepare_data(test_X)



    
def create_model():

    np.random.seed(0)

    

    inpt_num = Input(shape=(6,) ,name='numeric' )

    inpt_emb = Input(shape=(1,) ,name='embedding' )

    

    emb = Embedding( input_dim=df['Embarked'].max()+1, output_dim=5)(inpt_emb)

    emb = Flatten()(emb)

    x = concatenate( [inpt_num, emb] )

    

    x = Dense(100, kernel_regularizer=l2(1e-5)) (x)

    x = Activation('relu') (x)

    x = Dropout(.2) (x)

    

    x = Dense(1) (x)

    x = Activation('sigmoid') (x)

    

    

    

    model = Model([inpt_num, inpt_emb], x)

    

    return model
model = create_model()

model.summary()
es = EarlyStopping(monitor='val_loss',mode='min', patience=5)

mc = ModelCheckpoint( filepath='./weights.h5', monitor='val_loss',mode='min', save_best_only=True )



model = create_model()



model.compile( optimizer=SGD(), loss='mse', metrics=['accuracy'] )



model.fit( x=train_keras, y=train_y, batch_size=32, epochs=10000000, verbose=1, callbacks=[es, mc],

          validation_data=(test_keras, test_y), shuffle=True )

p  = model.predict( test_keras )

from sklearn.metrics import mean_squared_error, accuracy_score

print ( mean_squared_error( test_y, p), accuracy_score( test_y, p>.5) )



#700/700 [==============================] - 0s 52us/step - loss: 0.1333 - acc: 0.8114 - val_loss: 0.1081 - val_acc: 0.8586

#0.10784520996085677 0.8586387434554974
model.load_weights('./weights.h5')

p  = model.predict( test_keras )

from sklearn.metrics import mean_squared_error, accuracy_score

print ( mean_squared_error( test_y, p), accuracy_score( test_y, p>.5) )



#0.1078 4520996085677 sans checkpoint

#0.1078 1066959635068 avec checkpoint