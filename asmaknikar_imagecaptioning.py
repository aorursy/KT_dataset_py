# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()

import pandas as pd

from tqdm.notebook import tqdm

import string

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image

from tensorflow.keras.utils import to_categorical

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

start_token = 'starttoken'

end_token = 'endtoken'

# pad_token = '<pad>'
caption_data = pd.read_csv("/kaggle/input/image-captioning-assignment-data/captions.txt",sep="\t",header=None,names=["image","caption"])

gp34_images = pd.read_csv("/kaggle/input/image-captioning-assignment-data/image_names.txt",header=None).to_numpy()
caption_data
table = str.maketrans('', '', string.punctuation)

for index,caption_row in tqdm(caption_data.iterrows()):

    caption = caption_row['caption']

    words = caption.split()

    # convert to lower case

    words = [word.lower() for word in words]

    words = [w.translate(table) for w in words]

    words = [word for word in words if len(word)>1]

    words = [word for word in words if word.isalpha()]

    caption_data.iloc[index].caption = ' '.join(words)

#     print(words,end='\r')
grp34_data = caption_data[caption_data.image.str.split('#',expand=True)[0].isin(gp34_images[:,0])].reset_index(drop=True)

grp34_data[["image","cap_ind"]] = grp34_data.image.str.split('#',expand=True)
image_names = np.unique(grp34_data.image)

print("Number of images",image_names.shape)

train_images,test_images = train_test_split(image_names, test_size=0.2, random_state=42)

# test_images,val_images = train_test_split(test_images, test_size=(1/5), random_state=42)

print("Train",train_images.shape)

print("Test",test_images.shape)

# print("Val",val_images.shape)
grp34_data
dictionary = set()#Better this than using np unique since a set is faster

for cap_words in tqdm(caption_data.caption.str.split().to_numpy()):

    dictionary.update(cap_words)
f = open("/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt", encoding="utf-8")

dic_em = {}

# dic_em[pad_token] = np.zeros(200)

dic_em[start_token] = np.zeros(200)

dic_em[end_token] = np.zeros(200)

# count = 0

pbar = tqdm(total=4*10e4)

while True:

    pbar.update(1)

    line = f.readline() 

    if not line: 

        break

    values = line.split()

    word = values[0]

    if(word in dictionary):

        dic_em[word] = np.asarray(values[1:], dtype='float32')

        dictionary.remove(word)

# #     embeddings_index[word] = coefs

f.close()

pbar.close()

# print('Found %s word vectors.' % len(embeddings_index))



for rem_word in tqdm(dictionary):

    dic_em[rem_word] = np.ones(200)
dic_em_csv = pd.DataFrame.from_dict(dic_em,orient='index')

dic_em_csv['ix'] = np.arange(0,dic_em_csv.shape[0])

# dic_em_csv.to_csv('glove_mapping_dl_assignment.csv')

# embedding_weights = dic_em_csv.iloc[:,:-1].to_numpy()
max_caption_len = max(len(s.split()) for s in grp34_data.caption)

max_caption_len = 37

max_caption_len
tokenizer = Tokenizer()

tokenizer.fit_on_texts([start_token,end_token])

tokenizer.fit_on_texts(grp34_data.caption)
vocab_length = len(tokenizer.word_index)+1

embedding_weights = np.zeros((vocab_length,200))

for word in tokenizer.word_index:

    try:

        embedding_weights[tokenizer.word_index[word]] = dic_em_csv.loc[word][:200]

    except Exception as e:

        print(word,e)

        pass
images = np.unique(grp34_data.image)



image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+images[0])

image = image.resize((224,224),resample=Image.BILINEAR)

grp34_data[grp34_data.image == images[0]]
# image_names = np.unique(grp34_data.image)

# # X1,X2,y=list(),list(),list()

# X1,X2,y=np.empty(shape=[0,224,224,3]),list(),list()

# for image_name in tqdm(image_names):

#     image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

#     image = np.array(image.resize((224,224),resample=Image.BILINEAR))

#     image = np.expand_dims(image,0)

#     captions = grp34_data[grp34_data.image == image_name].caption

#     captions = tokenizer.texts_to_sequences(captions)

#     captions = [[tokenizer.word_index[start_token]]+caption+[tokenizer.word_index[end_token]] for caption in captions]



#     for caption in captions:

#         for inx in range(1,len(caption)):

#             in_seq,word = caption[:inx],caption[inx]

#             X2.append(in_seq)

# #             X1.append(image)

#             X1 = np.vstack([X1,image])

#             y.append(word)

# #                 X2 = np.vstack([X2,np.array(in_seq)])

#                 print(image.shape)

#                 y = np.vstack([y,word])

#     if(n==batch_size):

#         X2 = pad_sequences(X2,40,padding='post')

#         y = to_categorical(y,num_classes=vocab_length)

# #             X1 = X1

#         yield([[np.array(X1),np.array(X2)],np.array(y)])

# #             X1,X2,y=list(),list(),list()

# #             X1,X2,y=np.empty(shape=[0,224,224,3]),list(),list()

#         X1,X2,y=list(),list(),list()

#         n=0

        



np.random.rand(40).shape
t3 = np.empty(shape=[0,224,224,3])

np.vstack([t3,np.random.rand(1,224,224,3)]).shape
def generator_2(image_set,batch_size=1):

#     image_names = np.unique(grp34_data.image)

    n =0

    X1,X2,y=list(),list(),list()

    while (True):

        for image_name in image_set:

    #         print(image_name)

            n+=1

    #         X1,X2,y=np.empty(shape=[0,224,224,3]),list(),list()

            image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

            image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255

    #         image = np.expand_dims(image,0)

            captions = grp34_data[grp34_data.image == image_name].caption

            captions = tokenizer.texts_to_sequences(captions)

            captions = [[tokenizer.word_index[start_token]]+caption+[tokenizer.word_index[end_token]] for caption in captions]

            for caption in captions:

                X1.append(image)

                X2.append(caption)

#                 for inx in range(1,len(caption)):

#                     in_seq,word = caption[:inx],caption[inx]

#                     X2.append(in_seq)

#                     X1.append(image)

#                     y.append(word)

            if(n==batch_size):

                X2 = pad_sequences(X2,max_caption_len,padding='post')

                y = to_categorical(X2,num_classes=vocab_length)

    #             X1 = X1

                yield([np.array(X1),np.array(X2)],y)

                X1,X2,y=list(),list(),list()

                n=0
gen = generator_2(train_images,1)

t3 = next(gen)

print("image:",t3[0][0].shape)

print("cap_in",t3[0][1].shape)

print("y",t3[1].shape)

t3[0][1][2]
def generator(image_set,batch_size=1):

#     image_names = np.unique(grp34_data.image)

    n =0

    X1,X2,y=list(),list(),list()

    while (True):

        for image_name in image_set:

    #         print(image_name)

            n+=1

    #         X1,X2,y=np.empty(shape=[0,224,224,3]),list(),list()

            image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

            image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255

    #         image = np.expand_dims(image,0)

            captions = grp34_data[grp34_data.image == image_name].caption

            captions = tokenizer.texts_to_sequences(captions)

            captions = [[tokenizer.word_index[start_token]]+caption+[tokenizer.word_index[end_token]] for caption in captions]

    #         captions = pad_sequences(captions,40,padding='post')

            for caption in captions:

#                 print(caption)

                for inx in range(1,len(caption)):

                    in_seq,word = caption[:inx],caption[inx]

#                     print(tokenizer.index_word[word],in_seq)

                    X2.append(in_seq)

                    X1.append(image)

                    y.append(word)

    #                 X2 = np.vstack([X2,np.array(in_seq)])

    #                 print(image.shape)

    #                 X1 = np.vstack([X1,image])

    #                 y = np.vstack([y,word])

            if(n==batch_size):

                X2 = pad_sequences(X2,max_caption_len,padding='post')

                y = to_categorical(y,num_classes=vocab_length)

    #             X1 = X1

                yield([np.array(X1),np.array(X2)],np.array(y))

    #             yield(np.array(X1),np.array(y))

    #             X1,X2,y=list(),list(),list()

    #             X1,X2,y=np.empty(shape=[0,224,224,3]),list(),list()

                X1,X2,y=list(),list(),list()

                n=0
gen = generator(train_images,1)
t2 = next(gen)
t2[1].shape
print("image:",t2[0][0].shape)

print("cap_in",t2[0][1].shape)

print("y",t2[1].shape)
tokenizer.sequences_to_texts(t2[0][1])
from matplotlib import pyplot as plt

plt.imshow(t2[0][0][1])
# from tensorflow.keras import Sequential

# from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D,AveragePooling2D,Flatten, InputLayer, BatchNormalization,LSTM,Embedding,Concatenate,RepeatVector

# from keras.layers.merge import add

# from tensorflow.keras.models import Model

# from tensorflow.keras.utils import plot_model

# import tensorflow as tf

# from tensorflow.keras.losses import sparse_categorical_crossentropy

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



from keras import Sequential

from keras.layers import Input, Dense, Dropout, Conv2D,AveragePooling2D,Flatten, InputLayer, BatchNormalization,LSTM,Embedding,Concatenate,RepeatVector,Layer,Add,SimpleRNN

from keras.layers.merge import add

from keras.models import Model

from keras.utils import plot_model

import tensorflow as tf

from keras.losses import sparse_categorical_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

from tensorflow.keras.losses import CategoricalCrossentropy

from keras.applications import VGG16

class NetVLAD(Layer):

    def __init__(self,filters,**kwargs):

        super(NetVLAD,self).__init__(**kwargs)

        self.filters = filters

        

    def build(self,input_shape):

        self.depth = input_shape[-1]

        self.conv = Conv2D(filters=self.filters,kernel_size=1,strides=(1,1),use_bias=False,padding='valid',kernel_initializer='uniform')

        self.C = self.add_weight(name='cluster_center',shape=(1,1,1,self.depth,self.filters),initializer='uniform',trainable=True)



    def call(self,X):

        A = self.conv(X)

        A = tf.nn.softmax(A)

        V = tf.expand_dims(X,-1)-self.C

        V = tf.expand_dims(A,-2)*V

        V = tf.reduce_sum(V,axis=[1,2])

        V = tf.transpose(V,perm=[0,2,1])

        return(V)

    

    def get_config(self):

        config = super().get_config().copy()

        config.update({

            'filters': self.filters,

        })

        return config
image_feature = Sequential()

image_feature.add(Input(shape = (224,224,3)))

# image_feature.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same'))

# image_feature.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# image_feature.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same'))

# image_feature.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# image_feature.add(Dropout(0.1))

vgg16 = VGG16(include_top=False,weights='imagenet')

vgg16.trainable=False

image_feature.add(vgg16)

image_feature.add(NetVLAD(64))

image_feature.add(Flatten())

image_feature.add(Dense(units=256, activation='relu'))

# image_feature.add(RepeatVector(max_caption_len))



image_feature.summary()
# image_feature.compile(loss='categorical_crossentropy', optimizer='adam')

# image_feature.fit_generator(generator(1), epochs=1, steps_per_epoch=500, verbose=1)
# # caption_model_1 = Sequential()

# # caption_model_1.add(Input(shape=(max_caption_len)))

# # caption_model_1.add(Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,trainable=False,weights=[embedding_weights]))

# # caption_model_1.add(SimpleRNN(256,unroll=True))

# # # caption_model_1.add(Dense(256))

# # caption_model_1.summary()





# dec_input_1 = Input(shape=(max_caption_len,))

# dec_inp_h_1 = Input(shape = (256,))

# dec_inp_c_1 = Input(shape = (256,))

# dec_em_1 = Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,

#                    trainable=False,weights=[embedding_weights])(dec_input_1)

# dec_rnn_1 = LSTM(256,unroll=True,return_state=True)

# dec_output_1,h_1,c_1 = dec_rnn_1(dec_em_1,initial_state=[dec_inp_h_1,dec_inp_c_1])

# states = [h_1,c_1]







# decoder_1 = Add()([image_feature.output,dec_output_1])

# # decoder_1 = Add()([image_feature.output,caption_model_1.output])

# # decoder_1 = add([image_feature.output,caption_model_1.output])

# # decoder_1 = LSTM(32)(decoder_1)

# decoder_1 = Dense(256,activation='relu')(decoder_1)

# decoder_1 = Dense(vocab_length,activation='relu')(decoder_1)



# # model_1 = Model(inputs=[image_feature.input,caption_model_1.input],outputs=decoder_1)

# model_1 = Model(inputs=[image_feature.input,dec_input_1,dec_inp_h_1,dec_inp_c_1],outputs=[decoder_1,h_1,c_1])



# print(model_1.summary)

# plot_model(model_1,to_file='/kaggle/working/model_1.png', show_shapes=True)
# # image: (51, 224, 224, 3)

# # cap_in (51, 37)

# # y (51, 6361)



# # image: (5, 224, 224, 3)

# # cap_in (5, 37)



# @tf.function

# def train_step(img_tensor,seq_inp,targ):

#     loss = 0



#     # initializing the hidden state for each batch

#     # because the captions are not related from image to image

#     h = tf.zeros((BATCH_SIZE,256))

#     c = tf.zeros((BATCH_SIZE,256))



#     dec_input = seq_inp[:,0]



#     with tf.GradientTape() as tape:

#         features = image_feature(img_tensor)



#         for i in range(1, seq_inp.shape[1]):

#             # passing the features through the decoder

#             predictions, h, c = decoder(features,dec_input,h,c )



#             loss += CategoricalCrossentropy(seq_inp[:, i], predictions)



#             # using teacher forcing

#             dec_input = tf.expand_dims(seq_inp[:, i], 1)



#     total_loss = (loss / int(seq_inp.shape[1]))



#     trainable_variables = encoder.trainable_variables + decoder.trainable_variables



#     gradients = tape.gradient(loss, trainable_variables)



#     optimizer.apply_gradients(zip(gradients, trainable_variables))



#     return loss, total_loss
# caption_model_1 = Sequential()

# caption_model_1.add(Input(shape=(max_caption_len)))

# caption_model_1.add(Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,trainable=False,weights=[embedding_weights]))

# caption_model_1.add(SimpleRNN(256,unroll=True))

# # caption_model_1.add(Dense(256))

# caption_model_1.summary()





dec_input_1 = Input(shape=(None,))

dec_em_1 = Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,

                   trainable=False,weights=[embedding_weights])(dec_input_1)

dec_rnn_1 = LSTM(256,return_state=True,return_sequences=True)

dec_output_1,h_1,c_1 = dec_rnn_1(dec_em_1)

states = [h_1,c_1]







decoder_1 = Add()([RepeatVector(max_caption_len)(image_feature.output),dec_output_1])

# decoder_1 = Add()([image_feature.output,caption_model_1.output])

# decoder_1 = LSTM(32)(decoder_1)

dense_1_1 = Dense(256,activation='relu')

decoder_1 = dense_1_1(decoder_1)

dense_2_1 = Dense(vocab_length,activation='softmax')

decoder_1 = dense_2_1(decoder_1)



# model_1 = Model(inputs=[image_feature.input,caption_model_1.input],outputs=decoder_1)

model_1 = Model(inputs=[image_feature.input,dec_input_1],outputs=decoder_1)



print(model_1.summary)

plot_model(model_1,to_file='/kaggle/working/model_1.png', show_shapes=True)
# gen = generator(1)

batch_size_images = 4

mcp_save = ModelCheckpoint('/kaggle/working/mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')

opt = Adam(learning_rate=1e-4)

model_1.compile(loss='categorical_crossentropy', optimizer=opt)

his = model_1.fit(generator_2(train_images,batch_size_images), epochs=12,

#                 validation_data = generator(val_images,batch_size_images),

                steps_per_epoch=(len(train_images)//batch_size_images),

#                 validation_steps=(len(val_images)//batch_size_images),

#                 steps_per_epoch=5,

#                 validation_steps = 2,

                callbacks=[mcp_save],

                verbose=1)
his.history
from matplotlib import pyplot as plt

plt.plot(his.history['loss'])

# plt.plot(his.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.savefig('/kaggle/working/test-2.png', bbox_inches='tight')

plt.show()

# def predict_for_image_1(image_name):

#     in_text = [start_token_t]

#     image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

#     image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255

# #     plt.imshow(image)

# #     plt.show;

#     X1_test =image

#     X1_test = np.expand_dims(X1_test,axis=0)

#     for i in range(40):

#         X2_test = np.array(pad_sequences([in_text],max_caption_len,padding='post'))

# #         X2_test = np.expand_dims(X2_test,axis=0)

# #         print(X1_test.shape,X2_test.shape)

#         y_pred = model_1.predict([X1_test,X2_test])

#         y_pred_t = np.argmax(y_pred)

#         if(y_pred_t==end_token_t):

#             break

#         in_text+= [np.argmax(y_pred)]

# #     print(tokenizer.sequences_to_texts([in_text]))

#     return(in_text[1:])
# y_pred_all =list()

# start_token_t = tokenizer.word_index[start_token]

# end_token_t = tokenizer.word_index[end_token]

# t=0

# for image_name in tqdm(test_images):

#     t+=1

# #     print(image_name)

#     in_text = predict_for_image_1(image_name)

# #     print(tokenizer.sequences_to_texts([in_text]))

    

#     y_pred_all.append(in_text)

# #     break

# #     if(t==4):

# #         break

# y_pred_all = tokenizer.sequences_to_texts(y_pred_all)
dec_inf_features = Input(shape=(256,))

dec_inp_h_1 = Input(shape = (256,))

dec_inp_c_1 = Input(shape = (256,))

dec_inf_states = [dec_inp_h_1,dec_inp_c_1]

dec_inf_em = Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,

                   trainable=False,weights=[embedding_weights])(dec_input_1)



dec_output_inf_1,h_1,c_1 = dec_rnn_1(dec_inf_em,initial_state=dec_inf_states)



out_1 = Add()([RepeatVector(1)(dec_inf_features),dec_output_inf_1])

print(out_1.shape)

decoder_inf_1 = dense_1_1(out_1)

decoder_inf_1 = dense_2_1(decoder_inf_1)



# model_1 = Model(inputs=[image_feature.input,caption_model_1.input],outputs=decoder_1)

model_inference_1 = Model(inputs=[dec_inf_features,dec_input_1,dec_inp_h_1,dec_inp_c_1],outputs=[decoder_inf_1,h_1,c_1])



print(model_inference_1.summary)

plot_model(model_inference_1,to_file='/kaggle/working/model_1.png', show_shapes=True)
start_token_t = tokenizer.word_index['starttoken']

end_token_t = tokenizer.word_index['endtoken']

y_pred_all = list()

for (i,image_name) in enumerate(tqdm(test_images)):

    image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

    image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255



    X1_test = np.array([image])

    features = image_feature.predict(X1_test)

    inp_word = np.array([[start_token_t]])

    y_pred_ix = []

    h = tf.zeros((1,256))

    c = tf.zeros((1,256))

    try:

        while True:

            pred_word,h,c = model_inference_1.predict([features,inp_word,h,c])

            pred_index = np.argmax(pred_word, axis = -1)[0][0]

            if(pred_index == end_token_t or len(y_pred_ix)>max_caption_len):

                break

            y_pred_ix.append(pred_index)

    #         inp_word = np.array([[pred_index]])

            inp_word = np.append(inp_word,[y_pred_ix],axis=1)

    #         print(inp_word,y_pred_ix)

            curr_state = [h,c]

    #     print(tokenizer.sequences_to_texts([y_pred_ix]))

    except:

        y_pred_ix = []

        print(image_name)

    y_pred_all.append(y_pred_ix)

    

y_pred_all_txt = tokenizer.sequences_to_texts(y_pred_all)
from nltk.translate.bleu_score import sentence_bleu





df = pd.DataFrame({'x':test_images, 'y_pred':y_pred_all_txt})

df.to_csv('/kaggle/working/test_res-model1-lstm-.csv')



y_true_da = []

for image in df.x:

    y_true_da.append(grp34_data[grp34_data.image == image].caption.values)

df['y_actual'] = y_true_da



df['BLEUone'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(1, 0, 0, 0)), axis=1)

df['BLEUtwo'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.5, 0.5, 0, 0)), axis=1)

df['BLEUthr'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.33, 0.33, 0.33, 0)), axis=1)

df['BLEUfou'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.25, 0.25, 0.25, 0.25)), axis=1)



df.to_csv('/kaggle/working/test_res-model1-lstm-BLEU.csv')

pd.DataFrame({'x':test_images, 'y':y_pred_all}).to_csv('/kaggle/working/test_res-model1-30-11-45.csv')
# caption_model_2 = Sequential()

# caption_model_2.add(Input(shape=(max_caption_len)))

# caption_model_2.add(Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,trainable=False,weights=[embedding_weights]))

# caption_model_2.add(LSTM(256))

# # caption_model_2.add(Dense(256))





# caption_model_2.summary()



# decoder_2 = Add()([image_feature.output,caption_model_2.output])

# # decoder_2 = add([image_feature.output,caption_model_2.output])

# # decoder_2 = LSTM(32)(decoder_2)

# decoder_2 = Dense(256,activation='relu')(decoder_2)

# decoder_2 = Dense(vocab_length,activation='relu')(decoder_2)



# model_2 = Model(inputs=[image_feature.input,caption_model_2.input],outputs=decoder_2)



# print(model_2.summary)

# plot_model(model_2,to_file='/kaggle/working/model_2.png', show_shapes=True)
# embedding_size=200 
# # gen = generator(1)

# batch_size_images = 4

# mcp_save = ModelCheckpoint('/kaggle/working/md2_wts.hdf5', save_best_only=True, monitor='loss', mode='min')

# opt = Adam(learning_rate=5*1e-3)

# model_2.compile(loss=CategoricalCrossentropy(from_logits=True), optimizer=opt)

# his = model_2.fit(generator(train_images,batch_size_images), epochs=6,

# #                 validation_data = generator(val_images,batch_size_images),

#                 steps_per_epoch=(len(train_images)//batch_size_images),

# #                 validation_steps=(len(val_images)//batch_size_images),

# #                 steps_per_epoch=5,kernel7b44974945

# #                 validation_steps = 2,

#                 callbacks=[mcp_save],

#                 verbose=1)
# caption_model_2 = Sequential()

# caption_model_2.add(Input(shape=(max_caption_len)))

# caption_model_2.add(Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,trainable=False,weights=[embedding_weights]))

# caption_model_2.add(SimpleRNN(256,unroll=True))

# # caption_model_2.add(Dense(256))

# caption_model_2.summary()





dec_input_2 = Input(shape=(None,))

dec_em_2 = Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,

                   trainable=False,weights=[embedding_weights])(dec_input_2)

dec_rnn_2 = SimpleRNN(256,return_state=True,return_sequences=True)

dec_output_2,h_2 = dec_rnn_2(dec_em_2)

# states = [h_2,c_2]







decoder_2 = Add()([RepeatVector(max_caption_len)(image_feature.output),dec_output_2])

# decoder_2 = Add()([image_feature.output,caption_model_2.output])

# decoder_2 = LSTM(32)(decoder_2)

dense_1_2 = Dense(256,activation='relu')

decoder_2 = dense_1_2(decoder_2)

dense_2_2 = Dense(vocab_length,activation='softmax')

decoder_2 = dense_2_2(decoder_2)



# model_2 = Model(inputs=[image_feature.input,caption_model_2.input],outputs=decoder_2)

model_2 = Model(inputs=[image_feature.input,dec_input_2],outputs=decoder_2)



print(model_2.summary)

plot_model(model_2,to_file='/kaggle/working/model_2.png', show_shapes=True)
# gen = generator(1)

batch_size_images = 4

mcp_save = ModelCheckpoint('/kaggle/working/mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')

opt = Adam(learning_rate=1e-4)

model_2.compile(loss='categorical_crossentropy', optimizer=opt)

his = model_2.fit(generator_2(train_images,batch_size_images), epochs=12,

#                 validation_data = generator(val_images,batch_size_images),

                steps_per_epoch=(len(train_images)//batch_size_images),

#                 validation_steps=(len(val_images)//batch_size_images),

#                 steps_per_epoch=5,kernel7b44974945

#                 validation_steps = 2,

                callbacks=[mcp_save],

                verbose=1)
from matplotlib import pyplot as plt

plt.plot(his.history['loss'])

# plt.plot(his.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.savefig('/kaggle/working/test-2.png', bbox_inches='tight')

plt.show()

# def predict_for_image_2(image_name):

#     in_text = [start_token_t]

#     image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

#     image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255

# #     plt.imshow(image)

# #     plt.show;

#     X1_test =image

#     X1_test = np.expand_dims(X1_test,axis=0)

#     for i in range(40):

#         X2_test = np.array(pad_sequences([in_text],max_caption_len,padding='post'))

# #         X2_test = np.expand_dims(X2_test,axis=0)

# #         print(X1_test.shape,X2_test.shape)

#         y_pred = model_2.predict([X1_test,X2_test])

#         y_pred_t = np.argmax(y_pred)

#         if(y_pred_t==end_token_t):

#             break

#         in_text+= [np.argmax(y_pred)]

# #     print(tokenizer.sequences_to_texts([in_text]))

#     return(in_text)
# y_pred_all =list()

# start_token_t = tokenizer.word_index[start_token]

# end_token_t = tokenizer.word_index[end_token]

# t=0

# for image_name in tqdm(test_images):

#     t+=1

# #     print(image_name)

#     in_text = predict_for_image_2(image_name)

# #     print(in_text)

    

#     y_pred_all.append(in_text)

# #     break

# #     if(t==4):

# #         break

# y_pred_all = tokenizer.sequences_to_texts(y_pred_all)
dec_inf_features = Input(shape=(256,))

dec_inp_h_2 = Input(shape = (256,))

# dec_inp_c_2 = Input(shape = (256,))

# dec_inf_states = [dec_inp_h_2,dec_inp_c_2]

dec_inf_em = Embedding(input_dim=vocab_length, output_dim=200, input_length=max_caption_len,

                   trainable=False,weights=[embedding_weights])(dec_input_2)



dec_output_inf_2,h_2 = dec_rnn_2(dec_inf_em,initial_state=dec_inp_h_2)



out_2 = Add()([RepeatVector(1)(dec_inf_features),dec_output_inf_2])

print(out_2.shape)

decoder_inf_2 = dense_1_2(out_2)

decoder_inf_2 = dense_2_2(decoder_inf_2)



# model_2 = Model(inputs=[image_feature.input,caption_model_2.input],outputs=decoder_2)

model_inference_2 = Model(inputs=[dec_inf_features,dec_input_2,dec_inp_h_2],outputs=[decoder_inf_2,h_2])



print(model_inference_2.summary)

plot_model(model_inference_2,to_file='/kaggle/working/model_2.png', show_shapes=True)
start_token_t = tokenizer.word_index['starttoken']

end_token_t = tokenizer.word_index['endtoken']

y_pred_all = list()

for (i,image_name) in enumerate(tqdm(test_images)):

    image = Image.open('/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'+image_name)

    image = np.array(image.resize((224,224),resample=Image.BILINEAR))/255



    X1_test = np.array([image])

    features = image_feature.predict(X1_test)

    inp_word = np.array([[start_token_t]])

    y_pred_ix = []

    h = tf.zeros((1,256))

#     c = tf.zeros((1,256))

    try:

        while True:

            pred_word,h = model_inference_2.predict([features,inp_word,h])

            pred_index = np.argmax(pred_word, axis = -1)[0][0]

            if(pred_index == end_token_t or len(y_pred_ix)>max_caption_len):

                break

            y_pred_ix.append(pred_index)

    #         inp_word = np.array([[pred_index]])

            inp_word = np.append(inp_word,[y_pred_ix],axis=1)

    #         curr_state = [h,c]

#     print(tokenizer.sequences_to_texts([y_pred_ix]))

    except:

        y_pred_ix = []

        print(image_name)

    y_pred_all.append(y_pred_ix)

y_pred_all_txt = tokenizer.sequences_to_texts(y_pred_all)
from nltk.translate.bleu_score import sentence_bleu





df = pd.DataFrame({'x':test_images, 'y_pred':y_pred_all_txt})

df.to_csv('/kaggle/working/test_res-model2-simplernn.csv')



y_true_da = []

for image in df.x:

    y_true_da.append(grp34_data[grp34_data.image == image].caption.values)

df['y_actual'] = y_true_da



df['BLEUone'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(1, 0, 0, 0)), axis=1)

df['BLEUtwo'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.5, 0.5, 0, 0)), axis=1)

df['BLEUthr'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.33, 0.33, 0.33, 0)), axis=1)

df['BLEUfou'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.25, 0.25, 0.25, 0.25)), axis=1)



df.to_csv('/kaggle/working/test_res-model2-simplernn-BLEU.csv')
