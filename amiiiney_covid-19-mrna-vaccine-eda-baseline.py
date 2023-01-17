import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore')



# READ DATASETS

train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')



#### HELPER FUNCTIONS

def plotd(f1,f2):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15,5))

    #1 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((1,2),(0,0))

    plt.hist(a[f1], bins=4, color='black',alpha=0.5)

    plt.title(f'{f1}',weight='bold', fontsize=18)

    #first row sec col

    ax1 = plt.subplot2grid((1,2),(0,1))

    plt.hist(a[f2], bins=4, color='crimson',alpha=0.5)

    plt.title(f'{f2}',weight='bold', fontsize=18)

 

    plt.show()



def plotc(f1,f2):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15,5))

    #1 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((1,2),(0,0))

    plt.hist(a[f1], bins=7, color='black',alpha=0.7)

    plt.title(f'{f1}',weight='bold', fontsize=18)

    #first row sec col

    ax1 = plt.subplot2grid((1,2),(0,1))

    plt.hist(a[f2], bins=5, color='crimson',alpha=0.7)

    plt.title(f'{f2}',weight='bold', fontsize=18)

    plt.xticks(weight='bold')

    plt.show()

    

def ploth(data, w=15, h=9):

    plt.figure(figsize=(w,h))

    sns.heatmap(data.corr(), cmap='hot', annot=True)

    plt.title('Correlation between the features', fontsize=18, weight='bold')

    plt.xticks(weight='bold')

    plt.yticks(weight='bold')

    return plt.show()
train.head()
def length(feature):

    column= train[[feature]]

    column['length']= column[feature].apply(len)

    return column.head()



length('sequence')
length('reactivity')
train_data = []

for mol_id in train['id'].unique():

    sample_data = train.loc[train['id'] == mol_id]

    for i in range(68):

        sample_tuple = (sample_data['id'].values[0], sample_data['sequence'].values[0][i],

                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],

                        sample_data['reactivity'].values[0][i], sample_data['reactivity_error'].values[0][i],

                        sample_data['deg_Mg_pH10'].values[0][i], sample_data['deg_error_Mg_pH10'].values[0][i],

                        sample_data['deg_pH10'].values[0][i], sample_data['deg_error_pH10'].values[0][i],

                        sample_data['deg_Mg_50C'].values[0][i], sample_data['deg_error_Mg_50C'].values[0][i],

                        sample_data['deg_50C'].values[0][i], sample_data['deg_error_50C'].values[0][i])

        train_data.append(sample_tuple)
a = pd.DataFrame(train_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type', 'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10',

                                  'deg_pH10', 'deg_error_pH10', 'deg_Mg_50C', 'deg_error_Mg_50C', 'deg_50C', 'deg_error_50C'])

a.head()
plotd('reactivity', 'reactivity_error')
plotd('deg_50C','deg_Mg_50C')
plotd('deg_pH10','deg_Mg_pH10')
plotc('predicted_loop_type', 'structure')
sns.countplot(a['sequence'], palette='terrain', alpha=0.8)

plt.title('Nucleotides count per sequence', weight='bold', fontsize=12)

plt.show()
b=a[['reactivity', 'deg_Mg_pH10',

                                  'deg_pH10', 'deg_Mg_50C', 'deg_50C']]



ploth(b, 10, 4)
c= a[['id', 'sequence', 'structure', 'predicted_loop_type']]

c= pd.get_dummies(c, columns=['sequence', 'structure', 'predicted_loop_type'])

ploth(c)
#Filter public and private sets

public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



#trunc= trunc.apply(lambda x: x.str.slice(0, 68))

#trunc2= trunc2.apply(lambda x: x.str.slice(0, 91))

#trunc=public_df[['sequence']]

#trunc2=private_df[['sequence']]

#public_df['sequence']=trunc['sequence']

#private_df['sequence']=trunc2['sequence']



#PUBLIC SET

public_data = []

for mol_id in public_df['id'].unique():

    sample_data = public_df.loc[public_df['id'] == mol_id]

    for i in range(68):

        sample_tuple = (sample_data['id'].values[0] + '_' + str(i),

                        sample_data['sequence'].values[0][i],

                        sample_data['structure'].values[0][i], 

                        sample_data['predicted_loop_type'].values[0][i],

                        )

        public_data.append(sample_tuple)



pudf_=pd.DataFrame(public_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type'])

        



#PRIVATE SET

private_data = []

for mol_id in private_df['id'].unique():

    sample_data = private_df.loc[private_df['id'] == mol_id]

    for i in range(91):

        sample_tuple = (sample_data['id'].values[0] + '_' + str(i),

                        sample_data['sequence'].values[0][i],

                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],

                        )

        private_data.append(sample_tuple)

        

prdf_=pd.DataFrame(private_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type'])





#ENCODE CATEGORICAL FEATURES IN PUBLIC AND PRIVATE SETS

#X2= pd.get_dummies(pudf, columns=['sequence', 'structure', 'predicted_loop_type'])

#X3= pd.get_dummies(prdf, columns=['sequence', 'structure', 'predicted_loop_type'])





#DROP ID

#X2= X2.drop('id', axis=1)

#X3= X3.drop('id', axis=1)

#X=c.drop('id', axis=1)
from sklearn.preprocessing import OrdinalEncoder

df = OrdinalEncoder(dtype="int").fit_transform(a[['id', 'sequence', 'structure', 'predicted_loop_type']])

df=pd.DataFrame(df, columns=['id', 'sequence', 'structure', 'predicted_loop_type'] )

#validation

valid_x= df[df['id']==2399]

dense_cols=['sequence', 'structure', 'predicted_loop_type']





pudf = OrdinalEncoder(dtype="int").fit_transform(pudf_[['id', 'sequence', 'structure', 'predicted_loop_type']])

prdf = OrdinalEncoder(dtype="int").fit_transform(prdf_[['id', 'sequence', 'structure', 'predicted_loop_type']])



pudf=pd.DataFrame(pudf, columns=['id', 'sequence', 'structure', 'predicted_loop_type'] )

prdf=pd.DataFrame(prdf, columns=['id', 'sequence', 'structure', 'predicted_loop_type'] )



def make_X(df):

    X = {"dense1": df[dense_cols].to_numpy()}

    for i, v in enumerate(dense_cols):

        X[v] = df[[v]].to_numpy()

    return X



X2= make_X(pudf)

X3= make_X(prdf)



df=make_X(df)



de = np.split(b, b.shape[1], axis=1)



valid_x=make_X(valid_x)



valid_y=b[['reactivity']].tail(68)

valid_y = np.split(valid_y, valid_y.shape[1], axis=1)
import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential

from keras.utils import np_utils

import keras.models as models

import keras.layers as layers

from keras import regularizers

import numpy.random as nr

import keras

from keras.layers import Dropout, BatchNormalization

import keras.layers as layers

import tensorflow as tf

import tensorflow.keras as keras



from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten

from tensorflow.keras.models import Model

import gc

import os

from tqdm.notebook import tqdm
def create_model():

    tf.random.set_seed(173)



    tf.keras.backend.clear_session()

    gc.collect()



    # Dense input

    dense_input = Input(shape=(len(dense_cols), ), name='dense1')



    # Embedding input

    sequence = Input(shape=(1,), name='sequence')

    structure = Input(shape=(1,), name='structure')

    loop = Input(shape=(1,), name='predicted_loop_type')

    

    sequence_emb = Flatten()(Embedding(4, 1)(sequence))

    structure_emb = Flatten()(Embedding(3, 1)(structure))

    loop_emb = Flatten()(Embedding(7, 1)(loop))

    



    # Combine dense and embedding parts and add dense layers. Exit on linear scale.

    x = concatenate([dense_input, sequence_emb, structure_emb, loop_emb])

    x = Dense(512, activation="tanh")(x)

    x=  Dropout(0.2)(x)

    x = Dense(256, activation="tanh")(x)

    x=  Dropout(0.2)(x)

    x = Dense(128, activation="tanh")(x)

    x=  Dropout(0.2)(x)

    x = Dense(64, activation="tanh")(x)

    x=  Dropout(0.2)(x)

    x = Dense(16, activation="tanh")(x)

    outputs = Dense(5, name='output')(x)



    inputs = {"dense1": dense_input, "seq": sequence, "structure": structure, "loop": loop}



    # Connect input and output

    model = Model(inputs, outputs)



    model.compile(loss=keras.losses.mean_squared_error,

                  metrics=["mse"],

                  optimizer=keras.optimizers.Adam())

    return model
model = create_model()

history = model.fit(df, 

                    de,

                    batch_size=64,

                    epochs=50

                    #,shuffle=True

                    ,callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), tf.keras.callbacks.ModelCheckpoint('model.h5')]

                    #,validation_data=(valid_x, valid_y))

                   ,validation_split=0.33)
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

train_loss = history.history['loss']

test_loss = history.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'crimson', label = 'Test loss')

plt.plot(x, train_loss,color='black', label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Loss vs. Epoch',weight='bold', fontsize=18)





plt.show()
#PREDICT PUBLIC AND PRIVATE SETS

public_preds = model.predict(X2)

private_preds = model.predict(X3)
#MULTIOUTPUTREGRESSOR FOR MULTILABEL TASK

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import GradientBoostingRegressor

#model= MultiOutputRegressor(GradientBoostingRegressor(random_state=42)).fit(X, b)



#PREDICT PUBLIC SET

#public_preds=model.predict(X2)



#PREDICT PRIVATE SET

#private_preds=model.predict(X3)



#DATAFRAMES OF PREDS

pu_predictions= pd.DataFrame(public_preds, columns=b.columns)

pr_predictions= pd.DataFrame(private_preds, columns=b.columns)
#ADD id_seqpos to merge later with the submission file

pu_predictions['id_seqpos']= pudf_['id']

pr_predictions['id_seqpos']= prdf_['id']

#CONCAT PUBLIC AND PRIVATE SET

final= pd.concat([pu_predictions, pr_predictions])

#MERGE WITH SUB FILE

sub1=sub.merge(final, on='id_seqpos', how='left')

#DROP AND RENAME SUB COLUMNS

sub1= sub1.drop(['reactivity_x', 'deg_Mg_pH10_x', 'deg_pH10_x', 'deg_Mg_50C_x', 'deg_50C_x'], axis=1)

sub1= sub1.rename(columns= {'reactivity_y':'reactivity','deg_Mg_pH10_y':'deg_Mg_pH10',

                     'deg_pH10_y':'deg_pH10', 'deg_Mg_50C_y':'deg_Mg_50C',

                     'deg_50C_y':'deg_50C'})

#FILL NA WITH 0 (NONSCORED SEQUENCES)

submission= sub1.fillna(0)
submission.head()
submission.to_csv('submission.csv', index=False)