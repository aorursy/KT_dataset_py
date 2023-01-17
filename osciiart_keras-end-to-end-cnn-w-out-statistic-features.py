import numpy as np

import pandas as pd
### Step 1: load data

train = pd.read_csv('../input/training_variants')

test1 = pd.read_csv('../input/test_variants')

test2 = pd.read_csv('../input/stage2_test_variants.csv')



trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', 

                     header=None, skiprows=1, names=["ID","Text"])

test1x = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', 

                    header=None, skiprows=1, names=["ID","Text"])

testx2 = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', 

                     header=None, skiprows=1, names=["ID","Text"], encoding='utf-8')



test_solution = pd.read_csv('../input/stage1_solution_filtered.csv')
# merge test 1 and test 1 solution

test_idx = sorted(list(test_solution['ID'].unique()))

test_filterd = test1.loc[test_idx]



test_y = test_solution.iloc[:,1:].as_matrix()

test_y = np.argmax(test_y, axis=1)

test_y = test_y+1

test_filterd['Class'] = test_y



testx_filterd = test1x.loc[test_idx]
# merge variants and text

train = pd.merge(train, trainx, how='left', on='ID').fillna('')

test_filterd = pd.merge(test_filterd, testx_filterd, how='left', on='ID').fillna('')

test2 = pd.merge(test2, testx2, how='left', on='ID').fillna('')



pid = test2['ID'].values
# merge training and test data

train_test1 = pd.concat((train, test_filterd), axis=0, ignore_index=True)

y = train_test1['Class'].values # yを分離

train_test1 = train_test1.drop(['Class'], axis=1)



df_all = pd.concat((train_test1, test2), axis=0, ignore_index=True)

df_all.shape # should be (4675, 4)
### Step 2: Text Tokenize

from keras.preprocessing.text import Tokenizer



# tokenize gene in char level

gene_tokenizer = Tokenizer(char_level=True)

print("tokenizer learning...")

gene_tokenizer.fit_on_texts(texts=df_all['Gene'])

print("word count", len(gene_tokenizer.word_counts)) # 37
gene_token_list = gene_tokenizer.texts_to_sequences(df_all['Gene'])

gene_token = np.zeros([len(gene_token_list), 9], dtype=np.uint8)

for k, v in enumerate(gene_token_list):

    gene_token[k,:len(v)] = np.array(v)

for i in range(5):

    print(df_all['Gene'][i], gene_token[i])
# variation tokenize in char level

vari_tokenizer = Tokenizer(char_level=True)

print("tokenizer learning...")

vari_tokenizer.fit_on_texts(texts=df_all['Variation'])

print("word count", len(vari_tokenizer.word_counts)) # 65
vari_token_list = vari_tokenizer.texts_to_sequences(df_all['Variation'])

vari_token = np.zeros([len(vari_token_list), 55], dtype=np.uint8)

for k, v in enumerate(vari_token_list):

    vari_token[k,:len(v)] = np.array(v)

for i in range(5):

    print(df_all['Variation'][i], vari_token[i])
# text tokenize in word level. this process spends a few minutes.

text_tokenizer = Tokenizer()

print("tokenizer learning...")

text_tokenizer.fit_on_texts(texts=df_all['Text'])

print("word count", len(text_tokenizer.word_counts)) # 196704
text_token_list = text_tokenizer.texts_to_sequences(df_all['Text']) #this process spends a few minutes.

print(text_token_list[0])
# split data into training and test

gene_token_train = gene_token[:train_test1.shape[0]]

gene_token_test = gene_token[train_test1.shape[0]:]



vari_token_train = vari_token[:train_test1.shape[0]]

vari_token_test = vari_token[train_test1.shape[0]:]



text_token_train = text_token_list[:train_test1.shape[0]]

text_token_test = text_token_list[train_test1.shape[0]:]
# make y into one-hot

y = y -1

encoded_y = np.eye(9)[y]
### Step 3: training

# build model

from keras.models import Model

from keras.layers import Input, Embedding, Dense, Activation, Dropout, Reshape, Flatten

from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Concatenate, Add, Multiply

from keras.optimizers import Adam, SGD



def ConvGlu_block(input_tensor, nb_filter, kernel_size=7, strides=2):

    x = Conv1D(nb_filter, kernel_size, padding='same', strides=strides)(input_tensor)

    x = BatchNormalization()(x)

    gate = Conv1D(nb_filter, kernel_size, padding='same', strides=strides)(input_tensor)

    gate = BatchNormalization()(gate)

    gate = Activation('sigmoid')(gate)

    x = Multiply()([x, gate])

    x = Dropout(0.5)(x)

    shortcut = Conv1D(nb_filter, 1, padding='same', strides=strides)(input_tensor)

    shortcut = BatchNormalization()(shortcut)



    x = Add()([x, shortcut])



    return x



def CNN(k=9,

        embed_size=128,

        length=[9,55, 2048],

        boc=196704,

        ):



    input_gene = Input(shape=(length[0],))

    x = Embedding(37+1, embed_size)(input_gene)

    x = Reshape((length[0], embed_size))(x)

    x = ConvGlu_block(input_tensor=x, nb_filter=128, kernel_size=7, strides=1)

    x = MaxPooling1D(pool_size=9)(x)

    feature_gene = Flatten()(x)



    input_vari = Input(shape=(length[1],))

    x = Embedding(65+1, embed_size)(input_vari)

    x = Reshape((length[1], embed_size))(x)

    x = ConvGlu_block(input_tensor=x, nb_filter=128, kernel_size=7, strides=1)

    x = MaxPooling1D(pool_size=55)(x)

    feature_vari = Flatten()(x)



    input_text = Input(shape=(length[2],))

    x = Embedding(boc, embed_size)(input_text)

    x = Reshape((length[2], embed_size))(x)

    x = ConvGlu_block(input_tensor=x, nb_filter=128, kernel_size=7, strides=2)

    x = ConvGlu_block(input_tensor=x, nb_filter=256, kernel_size=7, strides=2)

    x = ConvGlu_block(input_tensor=x, nb_filter=512, kernel_size=7, strides=2)

    x = ConvGlu_block(input_tensor=x, nb_filter=512, kernel_size=7, strides=2)

    x = ConvGlu_block(input_tensor=x, nb_filter=512, kernel_size=7, strides=2)



    x = MaxPooling1D(pool_size=length[2]//2**5)(x)



    feature_text = Flatten()(x)



    gate = Dense(256)(feature_text)

    gate = BatchNormalization()(gate)

    gate = Activation('sigmoid')(gate)



    linear = Concatenate()([feature_gene, feature_vari])

    gated = Multiply()([linear, gate])

    x = Dense(1024)(gated)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1024)(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1024)(x)

    x = Activation('relu')(x)

    x = Dropout(0.2)(x)

    y = Dense(k, activation='softmax')(x)



    model = Model(inputs=[input_gene,

                          input_vari,

                          input_text

                          ],

                  outputs=y)

    opt = SGD(decay=1e-6, momentum=0.1, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



#commented for Kaggle Limits

len_text = 256  #Change to 2048, 256 for Kaggle Limits. 

boc = 1000 # Change to 196704, 1000 for Kaggle Limits. this is number of words to use.

model = CNN(length=[9,55,len_text], boc=boc)
# define batch generator

def batch_generator(gene,vari, text, y, batch_size, shuffle=True, len_text=2048, boc=1000):

    batch_index = 0

    n = y.shape[0]

    while 1:

        if batch_index == 0:

            index_array = np.arange(n)

            if shuffle:

                index_array = np.random.permutation(n)



        current_index = (batch_index * batch_size) % n

        if n >= current_index + batch_size:

            current_batch_size = batch_size

            batch_index += 1

        else:

            current_batch_size = n - current_index

            batch_index = 0



        batch_text = np.zeros([current_batch_size, len_text], np.uint32)

        index_array_batch = index_array[current_index: current_index + current_batch_size]

        for i in range(current_batch_size):

            text_i = text[index_array_batch[i]]

            text_i = np.array(text_i, dtype=np.uint32)

            text_i = text_i[text_i<boc]

            if text_i.shape[0] <= len_text:

                batch_text[i,:text_i.shape[0]] = text_i

            else:

                if shuffle:

                    start = np.random.randint(0, text_i.shape[0] - len_text)

                else:

                    start = 0

                text_crop = text_i[start:start+len_text]

                batch_text[i] = text_crop



        batch_gene = gene[index_array[current_index: current_index + current_batch_size]]

        batch_vari = vari[index_array[current_index: current_index + current_batch_size]]

        batch_y = y[index_array[current_index: current_index + current_batch_size]]



        yield [batch_gene, batch_vari, batch_text], batch_y
import math

# parameters

num_epoch = 5 #Change to 100, 5 for Kaggle Limits. 

batch_size = 16

learning_rate = 0.001

nb_val = 256



nb_sample = y.shape[0]

nb_train = nb_sample - nb_val

nb_train_step = math.ceil(nb_train / batch_size)

nb_val_step = math.ceil(nb_val / batch_size)
# split data into training and validation

np.random.seed(42)

perm = np.arange(nb_sample)

np.random.shuffle(perm)

idx_val, idx_train = perm[:nb_val], perm[nb_val:]

text_train = []

for i in idx_train:

    text_train.append(text_token_train[i])

text_val = []

for i in idx_val:

    text_val.append(text_token_train[i])
# build batch generator

gen = batch_generator(gene=gene_token_train[idx_train],

                      vari=vari_token_train[idx_train,],

                      text=text_train,

                      y=encoded_y[idx_train],

                      batch_size=batch_size,

                      shuffle=True,

                      len_text=len_text,

                      boc=boc)



gen_val = batch_generator(gene=gene_token_train[idx_val],

                          vari=vari_token_train[idx_val],

                          text=text_val,

                          y=encoded_y[idx_val],

                          batch_size=batch_size,

                          shuffle=False,

                          len_text=len_text,

                          boc=boc)
# training

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



save_checkpoint = ModelCheckpoint(filepath='best_weight.hdf5', monitor='val_loss', save_best_only=True)

lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=learning_rate * 0.00001)

early_stopping = EarlyStopping(monitor='val_loss',patience=16, verbose=1, min_delta=1e-4, mode='min')

Callbacks = [save_checkpoint, lerning_rate_schedular, early_stopping]



model.fit_generator(gen,

                    steps_per_epoch=nb_train_step,

                    epochs=num_epoch,

                    validation_data=gen_val,

                    validation_steps=nb_val_step,

                    callbacks=Callbacks)
### Step 4: prediction

# build batch generator

def test_batch_generator(gene, vari, text, batch_size, shuffle=True, len_text=2048, boc=1000):

    batch_index = 0

    n = gene.shape[0]

    while 1:

        if batch_index == 0:

            index_array = np.arange(n)

            if shuffle:

                index_array = np.random.permutation(n)



        current_index = (batch_index * batch_size) % n

        if n >= current_index + batch_size:

            current_batch_size = batch_size

            batch_index += 1

        else:

            current_batch_size = n - current_index

            batch_index = 0



        batch_text = np.zeros([current_batch_size, len_text], np.uint32)

        index_array_batch = index_array[current_index: current_index + current_batch_size]

        for i in range(current_batch_size):

            text_i = text[index_array_batch[i]]

            text_i = np.array(text_i, dtype=np.uint32)

            text_i = text_i[text_i<boc]

            if text_i.shape[0] <= len_text:

                batch_text[i,:text_i.shape[0]] = text_i

            else:

                if shuffle:

                    start = np.random.randint(0, text_i.shape[0] - len_text)

                else:

                    start = 0

                text_crop = text_i[start:start+len_text]

                batch_text[i] = text_crop



        batch_gene = gene[index_array[current_index: current_index + current_batch_size]]

        batch_vari = vari[index_array[current_index: current_index + current_batch_size]]



        yield [batch_gene, batch_vari, batch_text]

        

        

gen_test = test_batch_generator(gene=gene_token_test,

                                vari=vari_token_test,

                                text=text_token_test,

                                batch_size=batch_size,

                                shuffle=False,

                                len_text=len_text,

                                boc=boc,

                                )
# predict

model.load_weights('best_weight.hdf5')

nb_test = gene_token_test.shape[0]

nb_test_step = math.ceil(nb_test / batch_size)

y_pred = np.empty([nb_test, 9], dtype=np.float32)

print('predicting...')

for i in range(nb_test_step):

    batch = next(gen_test)

    predict = model.predict(batch)

    if i != nb_test_step - 1:

        y_pred[i * batch_size:(i + 1) * batch_size] = predict

    else:

        y_pred[i * batch_size:] = predict

print('done.')
# make submission

submission = pd.DataFrame(y_pred, columns=

                          ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'])

submission['ID'] = np.arange(y_pred.shape[0]) + 1

submission.to_csv("submission_CNN.csv", index=False)