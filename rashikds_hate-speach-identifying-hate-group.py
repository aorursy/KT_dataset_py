# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from keras.models import Sequential

from keras.preprocessing import sequence

from keras.models import Model, Input

from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout, Activation

from keras.preprocessing.text import Tokenizer

from keras.optimizers import Adam
#test=pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/train.csv')



#X_train = train["comment_text"].values





#X_test = test["comment_text"].values

train.tail()
def try_underSampling(train):

    

    #Prepare for under sampling 

    religion = train[(train.religion == 1)]

    print("target religion : ",religion.shape)

    ethnicity = train[(train.ethnicity == 1)]

    print("ethnicity: ",ethnicity.shape)

    sexualOrientation = train[(train.sexualOrientation == 1)]

    print("sexualOrientation: ",sexualOrientation.shape)

    gender = train[(train.sex == 1)]

    print("gender: ",gender.shape)

    no_hatespeech = train[(train.religion == 0) | (train.ethnicity == 0) | (train.sexualOrientation == 0) | (train.sex == 0)]

    print("No hatespeech : ",no_hatespeech.shape)

    all_together = train[(train.religion == 1) | (train.ethnicity == 1) | (train.sexualOrientation == 1) | (train.sex == 1)]



    religion = religion.sample(1189, replace=True)

    ethnicity = ethnicity.sample(1189, replace=True)

    sexualOrientation = sexualOrientation.sample(1189, replace=True)

    gender = gender.sample(1189, replace=True)

    no_hatespeech = no_hatespeech.sample(1189, replace=True)

    all_together = all_together.sample(1189, replace=True)



    print("New target religion : ",religion.shape)

    print("New ethnicity: ",ethnicity.shape)

    print("New sexualOrientation: ",sexualOrientation.shape)

    print("New gender: ",gender.shape)

    print("New No hatespeech : ",no_hatespeech.shape)

    print("All : ",all_together.shape)

    

    train_set = pd.concat([religion, ethnicity, sexualOrientation, gender, no_hatespeech, all_together])

    train_set = train_set[['id', 'target', 'comment_text', 'religion','ethnicity','sexualOrientation', 'sex']]



    return train_set
def one_hot_encode_by_label(dataset, new_col, col_list, threshold_val=0.1):

    

    for col in col_list:

        dataset[new_col] = (

            dataset[col] > threshold_val

        )*1.0  

    

    return dataset



#

religions = [

    'atheist',

    'buddhist',

    'christian',

    'hindu',

    'jewish',

    'muslim',

    'other_religion'

]

train = one_hot_encode_by_label(train, 'religion', religions)



#

ethnicity = [

    'asian',

    'black',

    'latino',

    'white',

    'other_race_or_ethnicity'

]

train = one_hot_encode_by_label(train, 'ethnicity', ethnicity)



# 

sexualOrientation = [

    'bisexual',

    'heterosexual',

    'homosexual_gay_or_lesbian',

    'transgender',

    'other_sexual_orientation'    

]

train = one_hot_encode_by_label(train, 'sexualOrientation', sexualOrientation)



#

sex = [

    'male',

    'female',

    'other_gender' 

]



train = one_hot_encode_by_label(train, 'sex', sex)



train=train[(train.religion == 1) | (train.ethnicity == 1) | (train.sexualOrientation == 1) | (train.sex == 1)]

print("Take everythin with hatespeech: ", train.shape)



#train[ train['religion'] == 1 ].tail()

train[['id', 'target', 'comment_text', 'religion','ethnicity','sexualOrientation', 'sex']]



train = try_underSampling(train)



train.tail()
X = train['comment_text']

y = train[['religion','ethnicity','sexualOrientation', 'sex']]





X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

y_train = y_train[[ 'religion','ethnicity','sexualOrientation', 'sex']].values

y_test = y_test[[ 'religion','ethnicity','sexualOrientation', 'sex']].values
max_features = 20000  # number of words we want to keep

maxlen = 200  # max length of the comments in the model

batch_size = 64  # batch size for the model

embedding_dims = 30  # dimension of the hidden variable, i.e. the embedding dimension

epochs = 10

tok = Tokenizer(num_words=max_features)

tok.fit_on_texts(list(X_train) + list(X_test))

x_train = tok.texts_to_sequences(X_train)

x_test = tok.texts_to_sequences(X_test)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')

print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))



print(x_train)

y_train=y_train[:1209265,:]

y_test=y_test[:1209265,:]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print(x_train)
def create_model(max_features, embedding_dims):

    print('Build model...')

    model = Sequential()



    # we start off with an efficient embedding layer which maps

    # our vocab indices into embedding_dims dimensions

    model.add(Embedding(max_features,

                        embedding_dims,

                        input_length=maxlen))

    model.add(Dropout(0.2))



    # we add a Convolution1D, which will learn filters

    # word group filters of size filter_length:

    model.add(Conv1D(filters=3,

                     kernel_size=100,

                     padding='valid',

                     activation='relu',

                     strides=1))

    # we use max pooling:

    model.add(GlobalMaxPooling1D())



    # We add a vanilla hidden layer:

    model.add(Dense(100))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))



    # We project onto a single unit output layer, and squash it with a sigmoid:

    model.add(Dense(4))

    model.add(Activation('sigmoid'))

    

    return model
def create_model_org(max_features, embedding_dims):

    comment_input = Input((maxlen,))



    # we start off with an efficient embedding layer which maps

    # our vocab indices into embedding_dims dimensions

    comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen, 

                            embeddings_initializer="uniform")(comment_input)

    

    # we add a GlobalMaxPooling1D, which will extract features from the embeddings

    # of all words in the comment

    pooling = GlobalMaxPooling1D()(comment_emb)

    

    # We add a hidden layer:

    h=Dense(5, activation='relu')(pooling)

    #model.add(Activation('relu'))



    # We project onto a three-unit output layer, and squash it with a sigmoid:

    output = Dense(4, activation='sigmoid')(h)

    return Model(inputs=comment_input, outputs=output)
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model = create_model_org(max_features, embedding_dims)

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy',f1])

model.summary()
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
def get_confusion_matrix(x_test,y_true):

    y_pred = model.predict(x_test)

    #print(y_pred)

    y_pred = (y_pred > 0.5) #greater than 0.50 on scale 0 to 1

    #print(y_pred)



    from sklearn.metrics import classification_report

    print(classification_report(y_true, y_pred))
print(model.metrics_names)

print(model.evaluate(x_test,y_test))

y_pred = model.predict(x_test)
get_confusion_matrix(x_test,y_test)