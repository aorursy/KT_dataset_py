# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import pickle

from tensorflow.keras.applications.inception_v3 import InceptionV3 , preprocess_input

from tensorflow.keras.preprocessing.image import load_img , img_to_array

from tensorflow.keras.models import Model

from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
model = InceptionV3(include_top=True, weights='imagenet')

model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

#model.summary()
src_dir = '../input/flickr8k_dataset/Flicker8k_Dataset/'

imgs = os.listdir(src_dir)

features = dict()

for im in tqdm_notebook(imgs):

    image = load_img(src_dir + imgs[0], target_size=(299, 299))

    image = img_to_array(image)

    image = np.expand_dims(image , axis=0)

    image = preprocess_input(image)

    feature = model.predict(image)

    file_name = im.split('.')[0]

    features[file_name] = feature[0]
with open(r"image_features.pickle",'wb') as file:

    pickle.dump(features,file)

! du -h image_features.pickle  # Verify 
%reset
import string
def load_doc(file):

    file = open(file,'r')

    texts = file.read()

    file.close()

    return texts



def prep_description(doc):

    stock = dict()

    for line in doc.split('\n'):

        tokens = line.split()

        if len(tokens) < 2:

            continue

        image_id, image_desc = tokens[0], tokens[1:]

        image_id = image_id.split('.')[0]

        image_desc = ' '.join(image_desc)

        if image_id not in stock:

            stock[image_id] = list()

        stock[image_id].append(image_desc)

    return stock



def clean_desc(desc):

    table = str.maketrans('', '', string.punctuation)

    for key , desc_list in desc.items():

        for i in range(len(desc_list)):

            des = desc_list[i]

            des = des.split()

            des = [w.lower() for w in des]

            des = [w.translate(table) for w in des]

            des = [w for w in des if len(w) >1]

            des = [w for w in des if w.isalpha()]

            desc_list[i] = ' '.join(des)

def collect_vocab(desc):

    voc = set()

    for key,desc_list in desc.items():

        for line in desc_list:

            voc.update(line.split())

    return voc



def save_descriptions(descriptions, filename):

    lines = list()

    for key, desc_list in descriptions.items():

        for desc in desc_list:

            lines.append(key + ' ' + desc)

    data = '\n'.join(lines)

    file = open(filename, 'w')

    file.write(data)

    file.close()
doc = load_doc('/kaggle/input/flickr8k_text/Flickr8k.token.txt')

descriptions = prep_description(doc)

print('Total no of descriptions: {}'.format(len(descriptions)))

clean_desc(descriptions)

vocabulary = collect_vocab(descriptions)

print('Length of volabulary: {}'.format(len(vocabulary)))
save_descriptions(descriptions, '/kaggle/working/descriptions.txt')
import pickle

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import Sequence

from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import add

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.python.keras.optimizers import Adam

from nltk.translate.bleu_score import corpus_bleu



from matplotlib import pyplot as plt

from random import choice, sample
def load_doc(file):

    file = open(file,'r')

    texts = file.read()

    file.close()

    return texts



def load_set(filename):

    doc = load_doc(filename)

    dataset = list()

    for line in doc.split('\n'):

        if len(line) < 1:

            continue

        pic_id = line.split('.')[0]

        dataset.append(pic_id)

    return set(dataset)



def load_description(filename,dataset):

    doc = load_doc(filename)

    descriptions = dict()

    

    for line in doc.split('\n'):

        tokens = line.split()

        image_id , image_desc = tokens[0] ,tokens[1:]

        if image_id in dataset:

            

            if image_id not in descriptions:

                descriptions[image_id] = list()

            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            

            descriptions[image_id].append(desc)

    return descriptions



def load_img_features(filename,image_ids):

    with open(filename,'rb') as f:

        data = pickle.load(f)

    features = {img_id:data[img_id] for img_id in image_ids}

    return features



def to_lines(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc
# loading training data

train_ids = load_set('/kaggle/input/flickr8k_text/Flickr_8k.trainImages.txt')

train_descriptions = load_description('descriptions.txt',train_ids)

train_img_features = load_img_features('image_features.pickle',train_ids)

print('Length of Train ids: {}'.format(len(train_ids)))

print('Length of Train descriptions: {}'.format(len(train_descriptions)))

print('Length of Train image features: {}'.format(len(train_img_features)))
val_ids = load_set('/kaggle/input/flickr8k_text/Flickr_8k.devImages.txt')

val_descriptions = load_description('descriptions.txt',val_ids)

val_img_features = load_img_features('image_features.pickle',val_ids)

print('Length of validation ids: {}'.format(len(val_ids)))

print('Length of validation descriptions: {}'.format(len(val_descriptions)))

print('Length of validation image features: {}'.format(len(val_img_features)))
def create_tokenizer(descriptions):

    desc_in_lines = to_lines(descriptions)

    tokenizer = Tokenizer(num_words = 4000 , oov_token='<ukn>')

    tokenizer.fit_on_texts(desc_in_lines)

    return tokenizer



def max_length(descriptions):

    lines = to_lines(descriptions)

    return max([len(lin.split()) for lin in lines])
tokenizer = create_tokenizer(train_descriptions)

vocab_size = 4000

print('Total different kind of words in our dataset: {}'.format(vocab_size))

max_words = max_length(train_descriptions)

print('longest description in training set contains {} words'.format(max_words))
class DataGenerator(Sequence):

    'Generates data for Keras'

    def __init__(self,tokenizer,max_length,descriptions,image_features,vocab_size,batch_size=5,shuffle=True):

        'Initialization'

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.descriptions = descriptions

        self.image_features = image_features

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.vocab_size = vocab_size

        self.list_IDs = list(image_features.keys())

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X1,X2,y = self.__data_generation(list_IDs_temp)

        return [X1,X2], y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.image_features))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X1, X2, y = list(), list(), list()

        # Generate data

        for data_id in list_IDs_temp:

            img_vec = self.image_features[data_id]

            desc_list = self.descriptions[data_id]

            for desc in desc_list:

                seq = self.tokenizer.texts_to_sequences([desc])[0]

                for i in range(1,len(seq)):

                    in_seq, out_seq = seq[:i] , seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]

                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                    

                    X1.append(img_vec)

                    X2.append(in_seq)

                    y.append(out_seq)

            

        

        return np.array(X1), np.array(X2), np.array(y)
train_generator = DataGenerator(tokenizer,max_words,train_descriptions,train_img_features,vocab_size,batch_size=100,shuffle=True)

train_steps = train_generator.__len__()

train_steps
word_to_int = tokenizer.word_index

int_to_word = {val:key for key,val in word_to_int.items()}

#int_to_word
# k  = train_generator.__getitem__(7)

# f = list(k[0][1][6])

# kw = [int_to_word[j] for j in f if j != 0]

# kw
val_generator = DataGenerator(tokenizer,max_words,val_descriptions,val_img_features,vocab_size,batch_size=100,shuffle=True)

val_steps = val_generator.__len__()

val_steps
def merge_model(vocab_size,max_length):

    # feature layers

    inputs1 = Input(shape=(2048,))

    fe1 = Dropout(0.5)(inputs1)

    fe2 = Dense(256,activation='relu')(fe1)

    # sequence layers

    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size , 256,mask_zero=True)(inputs2)

    se2 = Dropout(0.5)(se1)

    se3 = LSTM(256)(se2)

    # decoder layers

    decoder1 = add([fe2,se3])

    decoder2 = Dense(256,activation='relu')(decoder1)

    outputs = Dense(vocab_size , activation='softmax')(decoder2)

    # Grouping layer to a model

    model = Model(inputs = [inputs1, inputs2], outputs=outputs)

    

    return model    
model = merge_model(vocab_size,max_words)

model.summary()
model.compile(optimizer='rmsprop' , loss='categorical_crossentropy',metrics=['accuracy'] )

model.metrics_names
filename = 'model-weights.h5'

mc = ModelCheckpoint(mode='max', filepath=filename , monitor='val_acc',save_best_only='True', verbose=1,save_weights_only=True)

es = EarlyStopping(mode='max', monitor='val_acc', patience=5, verbose=1)

callbacks = [ mc, es]
history = model.fit_generator(train_generator,steps_per_epoch=train_steps,epochs=7,callbacks=callbacks,

                    validation_data=val_generator,validation_steps=val_steps)
def generate_caption(model, tokenizer, image, max_length):

    

    prediction = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([prediction])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        softmax_prob = model.predict([[image], sequence] , verbose = 0)

        int_token = np.argmax(softmax_prob)

        

        word = int_to_word[int_token]

        

        if word is None:

            raise exception("integer token Not Found")

            

        prediction += ' ' + word

        if word == 'endseq':

            break

    

    return prediction
test = list(load_set('/kaggle/input/flickr8k_text/Flickr_8k.testImages.txt'))

print('Test Dataset: %d' % len(test))

test_descriptions = load_description('descriptions.txt', test)

print('Descriptions: test=%d' % len(test_descriptions))

test_features = load_img_features('image_features.pickle', test)

print('images: test=%d' % len(test_features))
img_id = choice(test)

print(img_id)

pred = generate_caption(model, tokenizer, test_features[img_id], max_words)

img = plt.imread('/kaggle/input/flickr8k_dataset/Flicker8k_Dataset/' + img_id +'.jpg')

plt.figure(figsize=(8,8))

plt.title(pred)

plt.imshow(img)


def evaluate_model(model, descriptions, images, tokenizer, max_length):

    

    predictions = list()

    actuals = list()

    

    for key, desc_list in descriptions.items():

        

        pred = generate_caption(model, tokenizer, images[key], max_length)

        references = [d.split() for d in desc_list]

        actuals.append(references)

        predictions.append(pred.split())

        

    print('BLEU-1: %f' % corpus_bleu(actuals, predictions, weights=(1.0, 0, 0, 0)))

    print('BLEU-2: %f' % corpus_bleu(actuals, predictions, weights=(0.5, 0.5, 0, 0)))

    print('BLEU-3: %f' % corpus_bleu(actuals, predictions, weights=(0.3, 0.3, 0.3, 0)))

    print('BLEU-4: %f' % corpus_bleu(actuals, predictions, weights=(0.25, 0.25, 0.25, 0.25)))      
evaluate_model(model,test_descriptions,test_features,tokenizer,max_words)