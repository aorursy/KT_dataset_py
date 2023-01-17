import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import keras as K
%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from os import listdir
from collections import Counter
def load_doc(filename):
    with open(filename) as file:
        text = file.readlines()
        return text
    
def image_to_captions(text):
    hash_map = {}
    for line in text:
        token = line.split()
        image_id = token[0].split('.')[0] # separating with '.' to extract image id (removing .jpg)
        image_caption = ' '.join(token[1: ])
        
        if(image_id not in hash_map):
            hash_map[image_id] = [image_caption]
        else:
            hash_map[image_id].append(image_caption)
        
    return hash_map

def preprocess(map_img_to_captions):
    preprocessed_captions = []
    for key in map_img_to_captions.keys():
        for idx in range(len(map_img_to_captions[key])):
            tokens = map_img_to_captions[key][idx].split()
            tokens = [token.lower() for token in tokens if len(token)>1 if token.isalpha()]
            map_img_to_captions[key][idx] = ' '.join(tokens)
            
    return map_img_to_captions

def create_vocabulary(preprocessed_map):
    vocabulary = set()
    for img_captions in preprocessed_map.values(): # list of 5 captions for each image
        for caption in img_captions:
            for token in caption.split():
                vocabulary.add(token)    
    return vocabulary

def save_captions(preprocessed_map,filename):
    data = []
    for image_id,image_captions in preprocessed_map.items():
        for caption in image_captions:
            data.append(image_id + ' ' + caption + '\n')
            
    with open(filename,'w') as file:
        for line in data:
            file.write(line)
            
def img_id_train(filename):
    with open(filename) as file:
        data = file.readlines()
        train_img_name = []
        for img_id in data:
            train_img_name.append(img_id.split('.')[0])
    return train_img_name 

def load_captions(filename, img_name):
    doc = load_doc(filename) 
    train_captions = {}    
    
    for line in doc:
        tokens = line.split()
        image_id, image_caption = tokens[0], tokens[1:]

        if(image_id in img_name):
            if(image_id not in train_captions):
                train_captions[image_id] = []
            
            modified_caption = 'startseq ' + ' '.join(image_caption) + ' endseq'
            train_captions[image_id].append(modified_caption)
    
    return train_captions

def preprocess_image(img_path):
    img = image.load_img(img_path,target_size=(299,299)) 
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img

def images_name(path):
    img_name = set([path+image for image in listdir(path)])
    return img_name

# Function to encode given image into a vector of size (2048, )
def encode_image(image, model):
    image = preprocess_image(image)
    feature_vector = model.predict(image)
    feature_vector = feature_vector.reshape(feature_vector.shape[1], ) # reshape from (1, 2048) to (2048, )
    return feature_vector

def encode_image_feature(image_file_name, model):
    start_time = time()
    encoding_features = {}
    for idx,img in enumerate(image_file_name):
        encoding_features[img] = encode_image(img, model_new)
        if( (idx+1)%500 == 0):
            print('images encoded ',idx+1)        
    print("** Time taken for encoding images {} seconds **".format(time()-start_time))
    return encoding_features

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            temp='../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'
            
            photo = photos[temp+key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0
                
def max_len_caption(all_train_captions):   
    max_len = 0
    for caption in all_train_captions:
        max_len = max(max_len,len(caption.split()))
    print('Maximum length of caption= ',max_len)
    return max_len

def show_history_metrics(history, metrics_name=None):
    if metrics_name==None:
        print("未指定性能指标")
    else:
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history[metrics_name])
        plt.plot(history.history['val_'+metrics_name])
        plt.title('Model '+metrics_name)
        plt.ylabel(metrics_name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()

def greedySearch(photo, model, max_length_caption, wordtoix, ixtoword):
    in_text = 'startseq'
    for i in range(max_length_caption):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length_caption)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def define_model(max_length, vocab_size, input_shape=(2048,)):
    inputs1 = Input(shape=input_shape) 
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def Predict_test_caption(test_image_id, model, show_predict=True, CNN_units_num=2048):
    print("image_id:"+test_image_id)
    test_image_filename = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'+test_image_id+'.jpg'
    image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
    pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
    true_captions = preprocessed_map[test_image_id]
    if show_predict:
        x=plt.imread(test_image_filename)
        plt.imshow(x)
        plt.show()
        print("Predict:\n"+pred_caption)
        print("True:")
        print(*preprocessed_map[test_image_id],sep='\n')
    return pred_caption, true_captions

def compute_BLEU(pred_caption, true_captions, show_bleu=True): 
    bleu = [0.0, 0.0, 0.0, 0.0]
    references = [true_captions[0].split(),true_captions[1].split(),true_captions[2].split(),true_captions[3].split(),true_captions[4].split()]
    hypothesis = pred_caption.split()
    smooth = SmoothingFunction()
    bleu[0] = sentence_bleu(references, hypothesis, weights=(1.0, 0, 0, 0), smoothing_function=smooth.method1)
    bleu[1] = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    bleu[2] = sentence_bleu(references, hypothesis, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth.method1)
    bleu[3] = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    if show_bleu:
        print('BLEU-1: %f' % bleu[0])
        print('BLEU-2: %f' % bleu[1])
        print('BLEU-3: %f' % bleu[2])
        print('BLEU-4: %f' % bleu[3])
        
    return bleu

def evaluate_BLEU(encoding_features, images_name, model, show_results=True, CNN_units_num=2048):
    mean_bleu = np.zeros(4)
    for test_id in iter(images_name):
        test_image_filename = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'+test_id+'.jpg'
        image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
        pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
        true_captions = preprocessed_map[test_id]
         
        bleu = compute_BLEU(pred_caption, true_captions, show_bleu=False)
        bleu_temp = np.array(bleu)
        mean_bleu = mean_bleu + bleu_temp
    
    mean_bleu = mean_bleu/len(images_name)
    if show_results:
        print('MEAN_BLEU-1: %f' % mean_bleu[0])
        print('MEAN_BLEU-2: %f' % mean_bleu[1])
        print('MEAN_BLEU-3: %f' % mean_bleu[2])
        print('MEAN_BLEU-4: %f' % mean_bleu[3])
    return mean_bleu    

def evaluate_corpursBLEU(encoding_features, images_name, model, show_results=True, CNN_units_num=2048):
    REFERS = [[]]*len(images_name)
    HYPOTS = [[]]*len(images_name)
    i = 0
    for test_id in iter(images_name):
        test_image_filename = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'+test_id+'.jpg'
        image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
        pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
        true_captions = preprocessed_map[test_id]
        
        references = [true_captions[0].split(),true_captions[1].split(),true_captions[2].split(),true_captions[3].split(),true_captions[4].split()]
        hypothesis = pred_caption.split()
  
        REFERS[i] = references
        HYPOTS[i] = hypothesis
        
        i = i+1
        
    C_bleu = [0.0, 0.0, 0.0, 0.0]
    smooth = SmoothingFunction()
    C_bleu[0] = corpus_bleu(REFERS, HYPOTS, weights=(1.0, 0, 0, 0), smoothing_function=smooth.method1)
    C_bleu[1] = corpus_bleu(REFERS, HYPOTS, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    C_bleu[2] = corpus_bleu(REFERS, HYPOTS, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth.method1)
    C_bleu[3] = corpus_bleu(REFERS, HYPOTS, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    
    if show_results:
        print('Corpus_bleu-1: %f' % C_bleu[0])
        print('Corpus_bleu-2: %f' % C_bleu[1])
        print('Corpus_bleu-3: %f' % C_bleu[2])
        print('Corpus_bleu-4: %f' % C_bleu[3])
    return C_bleu    
# 1.读取文件
filename = "../input/flicker8k-dataset/Flickr8k_text/Flickr8k.token.txt"
text = load_doc(filename)
for line in text[:5]:
    print(line,end='')
# 2.从数据集读取所有caption
map_img_to_captions = image_to_captions(text)
map_img_to_captions['1000268201_693b08cb0e']
# 3.预处理所有caption
preprocessed_map = preprocess(map_img_to_captions)
preprocessed_map['1000268201_693b08cb0e']
save_captions(preprocessed_map,'preprocessed_captions.txt')
vocabulary = create_vocabulary(preprocessed_map)
print('Vocabulary size',len(vocabulary))
# 4.将所有captions划分为train,valid,test
TRAIN_IMAGE_TEXT = '../input/flicker8k-dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
VALID_IMAGE_TEXT = '../input/flicker8k-dataset/Flickr8k_text/Flickr_8k.devImages.txt'
TEST_IMAGE_TEXT = '../input/flicker8k-dataset/Flickr8k_text/Flickr_8k.testImages.txt'

train_img_name = img_id_train(TRAIN_IMAGE_TEXT)
valid_img_name = img_id_train(VALID_IMAGE_TEXT)
test_img_name  = img_id_train(TEST_IMAGE_TEXT)

train_captions = load_captions('preprocessed_captions.txt', train_img_name)
valid_captions = load_captions('preprocessed_captions.txt', valid_img_name)
test_captions = load_captions('preprocessed_captions.txt', test_img_name)
plt.subplot(131)
plt.imshow(K.preprocessing.image.load_img('../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/1000268201_693b08cb0e.jpg',target_size=(299,299)))
print(len(train_captions))
print(train_captions['1000268201_693b08cb0e'][0])

plt.subplot(132)
plt.imshow(K.preprocessing.image.load_img('../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/2635164923_2a774f7854.jpg',target_size=(299,299)))
print(len(valid_captions))
print(valid_captions['2635164923_2a774f7854'][0])

plt.subplot(133)
plt.imshow(K.preprocessing.image.load_img('../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/256085101_2c2617c5d0.jpg',target_size=(299,299)))
print(len(test_captions))
print(test_captions['256085101_2c2617c5d0'][0])
# 5.读取CNN模型用于抽取特征
model = InceptionV3(weights='imagenet')
model_new = Model(inputs=model.input, outputs=model.layers[-2].output) # outputs=(second last layer output)
plot_model(model_new, to_file='model_new_CNN.png', show_shapes=True)
# 6.输入图片路径，抽取特征
path = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'
all_images_name = images_name(path)
train_image_full_name = [path+img+'.jpg' for img in train_img_name]
valid_image_full_name = [path+img+'.jpg' for img in valid_img_name]
test_image_full_name = [path+img+'.jpg' for img in test_img_name]

encoding_train = encode_image_feature(train_image_full_name, model_new)
encoding_valid = encode_image_feature(valid_image_full_name, model_new)
encoding_test = encode_image_feature(test_image_full_name, model_new)
with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    dump(encoding_train, encoded_pickle)

with open("encoded_valid_images.pkl", "wb") as encoded_pickle:
    dump(encoding_valid, encoded_pickle)    
    
with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    dump(encoding_test, encoded_pickle)
train_features = load(open("encoded_train_images.pkl", "rb"))
valid_features = load(open("encoded_valid_images.pkl", "rb"))
test_features = load(open("encoded_test_images.pkl", "rb"))
# 7.split caption创建词汇集(每个词汇唯一)
all_train_captions = []
for captions in train_captions.values():
    for caption in captions:
        all_train_captions.append(caption)
        
all_valid_captions = []
for captions in valid_captions.values():
    for caption in captions:
        all_valid_captions.append(caption)
        
all_test_captions = []
for captions in test_captions.values():
    for caption in captions:
        all_test_captions.append(caption)

corpus = []
for caption in all_train_captions:
    for token in caption.split():
        corpus.append(token)

hash_map = Counter(corpus)

vocab = []
for token,count in hash_map.items():
    if(count>=1):
        vocab.append(token)
# 8.读取预训练Embedding权重，创建Embedding矩阵

word_to_index = {}
index_to_word = {}
    
for idx,token in enumerate(vocab):
    word_to_index[token] = idx+1
    index_to_word[idx+1] = token

vocab_size = len(index_to_word) + 1

max_length_caption = max_len_caption(all_train_captions)

embeddings_index = {}
f = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_to_index.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
        
print("Embedding_matrix.shape = ",embedding_matrix.shape)
# 9.创建image_captioning模型

model_ic = define_model(max_length_caption, vocab_size)
model_ic.summary()
plot_model(model_ic, to_file='model_ic.png', show_shapes=True)
print(model_ic.layers[2].input)
print(model_ic.layers[2].output)
model_ic.layers[2].set_weights([embedding_matrix])
model_ic.layers[2].trainable = False
LOSS = 'categorical_crossentropy'
OPTIM = 'adam'
METRICS = [K.metrics.CosineSimilarity(),
           K.metrics.Precision()]
model_ic.compile(loss=LOSS, optimizer=OPTIM, metrics=METRICS)
BATCH_SIZE = 5
EPOCHS = 20
STEPS = len(train_captions)//BATCH_SIZE
VALID_STEPS = len(valid_captions)//BATCH_SIZE

print(STEPS)
print(VALID_STEPS)
# 10.创建数据生成器
train_generator = data_generator(train_captions, train_features, word_to_index, max_length_caption, BATCH_SIZE)
valid_generator = data_generator(valid_captions, valid_features, word_to_index, max_length_caption, BATCH_SIZE)
test_generator = data_generator(test_captions, test_features, word_to_index, max_length_caption, BATCH_SIZE)
# 11.训练模型
history = model_ic.fit_generator(train_generator, 
                    epochs=EPOCHS, 
                    steps_per_epoch=STEPS, 
                    verbose=1,
                    validation_data=valid_generator,
                    validation_steps=VALID_STEPS)
model_ic.save_weights('model_ic_weights.h5')
model_ic.load_weights('model_ic_weights.h5')
# 12.输出性能随epoch变化曲线
show_history_metrics(history, 'loss')
show_history_metrics(history, 'precision_1')
show_history_metrics(history, 'cosine_similarity')
# 13.keras内置评估
scores = model_ic.evaluate(test_generator, steps=VALID_STEPS)
print(scores)
print("Loss:      %.5f" % scores[0])
print("Precision: %.5f" % scores[2])
with open("encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)
# 14.模型平均sentence_BLEU评估
Mean_bleu = evaluate_BLEU(encoding_test, test_img_name, model_ic)
C_bleu = evaluate_corpursBLEU(encoding_test, test_img_name, model_ic)
test_img_name
# 15.预测结果预览,输入图像id
pc, tc = Predict_test_caption('3605676864_0fb491267e', model_ic)
bleu = compute_BLEU(pc, tc)