import pandas as pd

import os as os

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data['data_set']='training'

test_data['data_set']='test'

df = pd.concat([train_data, test_data]).reset_index().drop(['index'], axis=1)

df = df.drop('Ticket', axis='columns')
import numpy as np

# extract Salutation from name



def parse_name(dataf):

    dataf['Salutation'] = [b.split('.',1)[0] for a,b in [row.split(',',1) for row in dataf['Name']]]

    return dataf

df = parse_name(df)

df = df.drop('Name', axis='columns')



# split salutation into 4 categories

def parse_salutation(row):

    titles = {' Mr': 1, ' Capt': 1,' Col': 1,' Dr': 1,' Rev': 1,' Major': 1,' Master': 1,

              ' Ms': 2,' Mlle': 2,' Mme': 2,' Mrs': 2,' Miss': 2,

             ' Jonkheer': 3, ' Sir': 3,' Don': 3,

              ' Lady': 4,' the Countess': 4, 'Dona':4

             }

    for i in range(0,5):

        row['title_'+str(i)] = 0

    if pd.notna(row['Salutation'] ):

        title_column = 'title_'+str(titles.get(row['Salutation'],0))

        row[title_column] = 1 # paint the correct column

    else : # if it is NaN

        row['title_0']=1

    return row   



df = df.apply(parse_salutation, axis =1).drop(['Salutation','title_0'],axis=1)



#proceed to split `male` from `female` in `sex` column.  We can make this *dichotomous* and assign `1` for `female` and `0` for `male` 

def parse_sex(dataf):

    dataf['Sex'] = np.where(dataf['Sex'] == 'female', 1,0)

    return dataf

df = parse_sex(df)



# extract cabin information (1st letter) and split into categories

def keep_cabin_first_letter(row):

    decks = {'A': 1, 'B': 2, 'C': 3, 'D':4, 'E':5, 'F':6, 'G':7}

    for i in range(0,len(decks)+1):

        row['Cabin_'+str(i)] = 0

    if pd.notna(row['Cabin'] ):

        deck = row['Cabin'][0].upper() ## converting to UPPERCLASS so that character case does not create new category

        cabin_column = 'Cabin_'+str(decks.get(deck,0))

        row[cabin_column] = 1 # paint the correct column

    else : # if it is NaN

        row['Cabin_0']=1

    return row   



df = df.apply(keep_cabin_first_letter, axis =1).drop(['Cabin', 'Cabin_0'], axis = 1)



# Embarked : this feature had some `null` values in it.  

# There are only 2 `null` values, so it does not make sense to create a separate category for these.  

# We convert these `null` values to the majority.

def embarkation_point(row):

    points = {'Q': 1, 'S': 2, 'C': 3}

    for i in range(0,len(points)+1):

        row['emb_'+str(i)] = 0

    if pd.notna(row['Embarked'] ):

        point = row['Embarked'][0].upper() ## converting to UPPERCLASS so that character case does not create new category

        emb_column = 'emb_'+str(points.get(point,0))

        row[emb_column] = 1 # paint the correct column

    else : # if it is NaN

        row['emb_0']=1

    return row   



df = df.apply(embarkation_point, axis =1).drop(['Embarked', 'emb_0'],axis=1)
import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats

f, ax = plt.subplots(1, 2, figsize=(16,8))

sns.distplot(df['Age'], norm_hist=True,ax=ax[0])

sns.distplot(df['Fare'], norm_hist=True, ax=ax[1])

# populate and standardize Age

# sample random numbers from a normal distribution to fill up the missing age

age_dist = [[0,0],

            [df['Age'][df.Pclass==1].mean(), df['Age'][df.Pclass==1].std()],

           [ df['Age'][df.Pclass==2].mean(), df['Age'][df.Pclass==2].std() ],

           [ df['Age'][df.Pclass==3].mean(), df['Age'][df.Pclass==3].std() ]]

def fill_age_na(row):

    if pd.isna(row['Age'] ):

        row['Age'] = np.abs(np.random.normal(loc=age_dist[row['Pclass']][0], scale=age_dist[row['Pclass']][1]))

    return row   



df = df.apply(fill_age_na, axis = 1)



# populate and standardize Fare

fare_dist = [[0,0],

            [df['Fare'][df.Pclass==1].mean(), df['Fare'][df.Pclass==1].std()],

           [ df['Fare'][df.Pclass==2].mean(), df['Fare'][df.Pclass==2].std() ],

           [ df['Fare'][df.Pclass==3].mean(), df['Fare'][df.Pclass==3].std() ]]

def fill_age_na(row):

    if pd.isna(row['Fare'] ):

        row['Fare'] = np.abs(np.random.normal(loc=fare_dist[row['Pclass']][0], scale=fare_dist[row['Pclass']][1]))

    if row['Fare'] == 0.0:

        row['Fare'] = np.abs(np.random.normal(loc=fare_dist[row['Pclass']][0], scale=fare_dist[row['Pclass']][1]))

    return row   



df = df.apply(fill_age_na, axis = 1)



# standardize values

df['Age'] = (df.Age - df.Age.mean()) / (df.Age.std())

df['Fare'] = (df.Fare - df.Fare.mean())/(df.Fare.std())
def code_categorical(dataf):

    dataf = pd.concat([dataf,pd.get_dummies(dataf['Pclass'], prefix='class',dummy_na=False)],axis=1).drop(['Pclass'],axis=1)

    return dataf

df = code_categorical(df)
def plot_fit_hist(hist, print_validation = True) :

    # summarize history for accuracy

    plt.plot(hist.history['accuracy'])

    if print_validation : plt.plot(hist.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    if print_validation: plt.legend(['train', 'test'], loc='upper left')

    else : plt.legend(['train'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(hist.history['loss'])

    if print_validation : plt.plot(hist.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    if print_validation: plt.legend(['train', 'test'], loc='upper left')

    else : plt.legend(['train'], loc='upper left')

    plt.show()

    return



train_data = df[df.data_set == 'training'].drop('data_set', axis='columns')

train = train_data.iloc[:,1:].values

np.random.shuffle(train)

Y, X = train[:,0],train[:,1:]

print(f'Y = {Y.shape}, X={X.shape}')

import tensorflow as tf

import tensorflow.keras as keras

from keras.layers import Input, Dense, BatchNormalization, Conv1D, Concatenate,Flatten, Lambda, Add

from keras.layers import Dropout

from keras.models import Model

from keras.optimizers import Adam, SGD



train_data = df[df.data_set == 'training'].drop( 'data_set', axis='columns')

train = train_data.iloc[:,1:].values

np.random.shuffle(train)

print(f'Y = {Y.shape}, X={X.shape}')



np.random.shuffle(train)

Y, X = train[:,0],train[:,1:]

act_1 = 'sigmoid'

act = 'tanh'

act_3 = 'relu'

act_4 = 'selu'

act_5 = 'softmax'

pad = 'causal'

k_size = 3

def TitanicModel(input_shape):

    X_input = Input(input_shape)

    X = Lambda(lambda x: tf.expand_dims(x, axis=-1))(X_input)

    X = BatchNormalization()(X)

    X = Dropout(0.2)(X)

    X = Conv1D(filters=25,kernel_size=1,padding=pad, activation=act)(X)

    X = Conv1D(filters=25,kernel_size=k_size,padding=pad, activation=act)(X)

    X = BatchNormalization()(X)

    X = Dropout(0.2)(X)

    X = Conv1D(filters=25,kernel_size=k_size,padding=pad, activation=act)(X)

    X = BatchNormalization()(X)

    X = Dropout(0.2)(X)

    X = Conv1D(filters=25,kernel_size=k_size,padding=pad, activation=act)(X)

    X = BatchNormalization()(X)

    X = Dropout(0.2)(X)

    X = Flatten()(X)

    X = Dense(30) (X)

    X = Dense(1, activation='sigmoid', name='fc')(X)



    # Create model.

    model = Model(inputs = X_input, outputs = X, name='TitanicModel')

    return model

tf.keras.backend.clear_session() # clear session

titanic = TitanicModel(X.shape[1:]) # create model

optimizer = Adam() # use Adam optimizer



#compile the model

titanic.compile(optimizer = optimizer, 

                   loss = 'binary_crossentropy', 

                   metrics = ['accuracy'])

#fit the model

history = titanic.fit(

    x = X,

    y = Y,

    batch_size= int(len(X)/2), # int(len(X)/2), #int(len(X)/4),

    validation_split=0.1,

    verbose = 0,

    epochs=500,

)



plot_fit_hist(history) # plot loss & accuracy versus epoch to see if we are learning and if learning is stable

print(history.history['accuracy'][-1])
history = titanic.fit(

    x = X,

    y = Y,

    batch_size=64,

    # validation_split=0.1,

    verbose = 0,

    epochs=1000)

plot_fit_hist(history, False)



test = df[df.data_set == 'test'].drop(['Survived','data_set'], axis='columns')

test_data = test.iloc[:,1:].values

print(test_data.shape)

predictions = titanic.predict(test_data)

predictions = np.squeeze(predictions)

predictions = 1*(predictions >= 0.5)

output = pd.DataFrame({'PassengerId': test.PassengerId.astype(int), 'Survived': predictions})

output.to_csv('titanic_submission.csv', index=False)