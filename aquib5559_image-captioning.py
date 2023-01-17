import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import re
import string
import nltk
from nltk.corpus import stopwords
import cv2
def readTextFile(path):
    with open(path) as f:
        captions = f.read()
    return captions
captions = readTextFile("../input/flicker8k-image-captioning/Flickr8k_text/Flickr8k.token.txt")
captions = captions.split('\n')[:-1]
first,second  = captions[10].split('\t')
print(first)
print(second)
descriptions = {}
for x in captions:
    first,second = x.split('\t')
    image_name = first.split(".")[0]
    # if the image id is already present or not
    if descriptions.get(image_name) is None:
        descriptions[image_name] = []
    descriptions[image_name].append(second)
descriptions["1002674143_1b742ab4b8"]
img_path = "../input/flicker8k-image-captioning/Flickr8k_Dataset/Flicker8k_Dataset/"
img = cv2.imread(img_path + "1002674143_1b742ab4b8.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()
def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]", " ", sentence)
    sentence = sentence.split()
    sentence = [s for s in sentence if len(s)>1]
    sentence = " ".join(sentence)
    return sentence
clean_text("A small girl in the grass plays with fingerpaints in front of a white canvas with a rainbow on it . #12")
for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])
descriptions["1000268201_693b08cb0e"]
vocab = set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]
    
print("Vocab Size : %d"% len(vocab))
total_words = []
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]
print("Total Words" , len(total_words))

import collections

counter = collections.Counter(total_words)
freq_cnt = dict(counter)
print(len(freq_cnt.keys()))
# Sort this dictionary according to the freq count
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

# Filter
threshold = 10
sorted_freq_cnt  = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]


print(len(total_words))
train_file_data = readTextFile("../input/flicker8k-image-captioning/Flickr8k_text/Flickr_8k.trainImages.txt")
test_file_data = readTextFile("../input/flicker8k-image-captioning/Flickr8k_text/Flickr_8k.testImages.txt")
train = [row.split(".")[0] for row in train_file_data.split("\n")[:-1]]
test = [row.split(".")[0] for row in test_file_data.split("\n")[:-1]]
train[:5]
# Prepare Description for the Training Data
# Tweak - Add <s> and <e> token to our training data

train_descriptions = {}
for img_id in train:
    train_descriptions[img_id] = []
    for cap in descriptions[img_id]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[img_id].append(cap_to_append)
train_descriptions["1000268201_693b08cb0e"]
from tensorflow.keras.applications.xception import Xception,preprocess_input,decode_predictions
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
model = Xception(weights='imagenet', input_shape=(299,299,3))
model.summary()
model_new = Model(model.input,model.layers[-2].output)
def preprocess_img(img):
    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1))
    return feature_vector
encode_img(img_path + "1000268201_693b08cb0e.jpg")
encoding_train = {}
for img_id in train:
    PATH = img_path + img_id + ".jpg"
    encoding_train[img_id] = encode_img(PATH)
encoding_test = {}
for img_id in test:
    PATH = img_path + img_id + ".jpg"
    encoding_test[img_id] = encode_img(PATH)
import pickle

with open("encoded_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)

with open("encoded_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)
# Vocab
len(total_words)
word_to_idx = {}
idx_to_word = {}
for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word
# Two special words
idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847
vocab_size = len(word_to_idx) + 1
print("Vocab Size",vocab_size)
max_len = 0 
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len = max(max_len,len(cap.split()))
        
print(max_len)
from tensorflow.keras.utils import to_categorical
def data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size):
    X1,X2, y = [],[],[]
    
    n =0
    while True:
        for key,desc_list in train_descriptions.items():
            n += 1
            
            photo = encoding_train[key]
            for desc in desc_list:
                
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    
                    #0 denote padding word
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorical([yi],num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                    
                if n==batch_size:
                    yield ([np.array(X1),np.array(X2)],np.array(y))
                    X1,X2,y = [],[],[]
                    n = 0
f = open("../input/glove6b50dtxt/glove.6B.50d.txt",encoding='utf8')
embedding_index = {}
for line in f :
    values = line.split()
    word = values[0]
    word_embedding = np.array(values[1:],dtype = 'float')
    embedding_index[word] = word_embedding
embedding_index['machine']
def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)
        
        if embedding_vector is not None:
            matrix[idx] = embedding_vector
    return matrix
embedding_matrix = get_embedding_matrix()
embedding_matrix.shape
input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256,activation='relu')(inp_img1)

#Captions as Input

input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim = vocab_size,output_dim=50,mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)
from tensorflow.keras.layers import Add
decoder1 = Add()([inp_img2,inp_cap3])
decoder2 = Dense(256,activation='relu')(decoder1)
outputs = Dense(vocab_size,activation='softmax')(decoder2)
model = Model(inputs=[input_img_features,input_captions],outputs=outputs)
model.summary()
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy',optimizer='adam')
epochs = 20
batch_size = 3
steps = len(train_descriptions)// batch_size
#number_pics_per_batch
generator = data_generator(train_descriptions,encoding_train,word_to_idx,
                                   max_len,batch_size)
model.fit_generator(generator,epochs=20,steps_per_epoch=steps)
model.save('model.h5')
def predict_caption(img):
    in_text = 'startseq'
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([img,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += (' ' +  word)
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption
plt.style.use("seaborn")
for i in range(15):
    idx = np.random.randint(0,1000)
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo_2048 = encoding_test[img_name].reshape((1,2048))
    
    i = plt.imread("../input/flicker8k-image-captioning/Flickr8k_Dataset/Flicker8k_Dataset/"+img_name+".jpg")
    
    caption = predict_caption(photo_2048)
    #print(caption)
    
    plt.title(caption)
    plt.imshow(i)
    plt.axis("off")
    plt.show()
