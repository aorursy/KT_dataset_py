import numpy as np

import pandas as pd

import os

import keras

import itertools



from Levenshtein import distance as levenshtein_distance

from sklearn.metrics.pairwise import pairwise_distances

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import re

from nltk.stem.porter import PorterStemmer



from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import regularizers

from sklearn.model_selection import train_test_split



import cv2

import matplotlib.pyplot as plt

import seaborn as sns



from dataclasses import dataclass
os.chdir("../input/asl-alphabet")

print(os.listdir("."))
# Strange folderstructure 

train_dir = 'asl_alphabet_train/asl_alphabet_train'

test_dir = 'asl_alphabet_test/asl_alphabet_test'
@dataclass

class ASL_Character_Images:

    asl_character: str

    images: []

        

def load_image(file_path):

    size_img = 64,64 

    image = cv2.imread(file_path)

    resize_img = cv2.resize(image, size_img)

    return cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)

    

def load_images(path):

    print("LOADING DATA FROM : ",end = "")

    image_collection = []



    for folder in os.listdir(path):

        print(folder, end = ' | ')



        asl_character_folder = os.path.join(train_dir,folder)

        asl_character = os.path.basename(asl_character_folder)

        

        image_list = []

        asl_character_images = ASL_Character_Images(asl_character,image_list)

        image_collection.append(asl_character_images)

        

        for file in os.listdir(asl_character_folder):

            image_path = os.path.join(asl_character_folder,file)

            asl_img = load_image(image_path)

            image_list.append(asl_img)

    

        image_list = np.array(image_list)

        image_list = image_list.astype('float32')/255.0

 

    return image_collection



image_train_collection = load_images(train_dir)
def generate_unique_labels(dataset):

    unique_image_collection = {}



    for asl_character_image in dataset:

        key = asl_character_image.asl_character



        if key not in unique_image_collection:

            if len(asl_character_image.images) >1:

                value = asl_character_image.images[1]

                unique_image_collection[key] = value

            else:

                value = asl_character_image.images

                key = key[0:1]

                unique_image_collection[key] = value



    return unique_image_collection



def print_unique_labels(dataset):

    print(dataset.keys())

    

unique_image_collection_train = generate_unique_labels(image_train_collection)

print_unique_labels(unique_image_collection_train)
fig = plt.figure(figsize = (15,15))

row = 5

col = 6



def plot_images(fig, image, label, row, col, index):

    fig.add_subplot(row, col, index)

    plt.axis('off')

    plt.imshow(image)

    plt.title(label)



def plot_image_sample():

    cnt = 1

    for label in unique_image_collection_train:

        plot_images(fig, unique_image_collection_train[label], label, row, col, cnt)

        cnt +=1



plot_image_sample()
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,

                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,

                   'Z':25,'space':26,'del':27,'nothing':28}



labels_inv_dict = {v: k for k, v in labels_dict.items()}
def prepare_data_keras():

    images = []

    labels = []

    

    for asl_character_image in image_train_collection:

        asl_character = asl_character_image.asl_character

        image = asl_character_image.images

        label = [labels_dict[asl_character] for _ in itertools.repeat(None, len(image))]



        images.extend(image)

        labels.extend(label)

        

    images = np.array(images)

    images = images.astype('float32')/255.0

    

    labels = keras.utils.to_categorical(labels)

    

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.05)



    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)

    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)

    

    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = prepare_data_keras()
def create_model():

    

    model = Sequential()

    

    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))

    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))

    model.add(MaxPool2D(pool_size = [3,3]))

    

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))

    model.add(Dense(29, activation = 'softmax'))

    

    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    model.summary()

    

    return model



def fit_model():

    model_hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 10, validation_split = 0.1)

    return model_hist 



model = create_model()

curr_model_hist = fit_model()
plt.plot(curr_model_hist.history['accuracy'])

plt.plot(curr_model_hist.history['val_accuracy'])

plt.legend(['train', 'test'], loc='lower right')

plt.title('accuracy plot - train vs test')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()



plt.plot(curr_model_hist.history['loss'])

plt.plot(curr_model_hist.history['val_loss'])

plt.legend(['training loss', 'validation loss'], loc = 'upper right')

plt.title('loss plot - training vs vaidation')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
evaluate_metrics = model.evaluate(X_test, Y_test)

print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
def load_test_images(path):

    print("LOADING TEST DATA FROM")

    image_collection = []



    for asl_character in os.listdir(path):

        image_path = os.path.join(path,asl_character)

        asl_img = load_image(image_path)

                    

        image_list = []

        image_list.append(asl_img)



        image_list = np.array(image_list)

        image_list = image_list.astype('float32')/255.0

    

        asl_character_images = ASL_Character_Images(asl_character,image_list)

        image_collection.append(asl_character_images)

        

    return image_collection



image_test_collection = load_test_images(test_dir)
predictions = {}



for asl_character_image in image_test_collection:

    asl_character = asl_character_image.asl_character

    image = asl_character_image.images[0]

    prediction = model.predict_classes(image.reshape(1,64,64,3))[0]

    

    predictions[asl_character] = {

        "prd" : labels_inv_dict[prediction],

        "img" : image

    }
predfigure = plt.figure(figsize = (13,13))



def plot_image(fig, image, label, predictions_label, row, col, index):

    fig.add_subplot(row, col, index)

    plt.axis('off')

    plt.imshow(image)

    title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label

    plt.title(title)

    return



row = 5

col = 6

cnt = 0



for asl_character in predictions:

    cnt += 1

    image = predictions[asl_character]['img']

    predictions_label = predictions[asl_character]['prd']

    label = asl_character

    

    plot_image(predfigure, image, label, predictions_label, row, col, cnt)

plt.show()
model.save('/kaggle/working/ASL_prediction.h5')
unique_image_collection_test = generate_unique_labels(image_test_collection)
def get_images_for_word(word):

    images_list = []

    for character in word.upper():

        image = unique_image_collection_test[character][0]

        images_list.append(image)

        

    return images_list
def predict_character(image):

    prediction = model.predict_classes(image.reshape(1,64,64,3))[0]

    return labels_inv_dict[prediction]



def predict_word(images):

    word = []

    for image in images:

        word.append(predict_character(image))

        

    return ''.join(word)



def print_images(images):

    fig = plt.figure(figsize = (15,15))

    row = 1

    col = len(images)

    index = 0

    

    for image in images:

        index += 1

        plot_images(fig, image, '', row, col, index)

            

images = get_images_for_word('pseudopseudohypoparathyroidism')

print_images(images)



natural_word = predict_word(images).lower()

print(natural_word)
from skimage import io



def predict_url_image(url):

    img = io.imread(url)



    size_img = 64,64 

    resize_img = cv2.resize(img, size_img)

    img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)

    print(predict_word([img]))
# F

predict_url_image("https://www.signingsavvy.com/images/words/alphabet/2/f1.jpg")

    

# O

predict_url_image("https://www.signingsavvy.com/images/words/alphabet/2/o1.jpg")



# O

predict_url_image("https://www.signingsavvy.com/images/words/alphabet/2/o1.jpg")



# D

predict_url_image("https://www.signingsavvy.com/images/words/alphabet/2/d1.jpg")
os.chdir("/kaggle/working")

print(os.listdir("."))
# Download common words and sharespeare

!wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt 

!wget https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
common_english_words = []

with open("/kaggle/working/20k.txt") as f:

    content = f.readlines()



common_english_words = [x.strip() for x in content] 
with open("/kaggle/working/t8.shakespeare.txt") as f:

    content = f.read()



shakespeare_texts = content.split("\n\n")
porter_stemmer = PorterStemmer()



def stemming_tokenizer(str_input):

    words = re.findall(r"(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))", str_input)

    words = [porter_stemmer.stem(word.lower()) for word in words]

    return words
from functools import reduce

from operator import add



txt = [stemming_tokenizer(paragraph) for paragraph in shakespeare_texts]

txt_flat = reduce(add, txt)

df = pd.DataFrame(txt_flat,columns=['word'])



df_count = df.groupby(['word']).size().reset_index(name='count')

df_count['freq'] = df_count['count']/df.size
def apply_levenshtein_distance_on_list(str_1, common_words):

    col_names =  ['score','word']

    my_df  = pd.DataFrame(columns = col_names)

    

    for word in common_words:

        score = levenshtein_distance(str_1, word)

        my_df = my_df.append({'score' : score , 'word' : word} , ignore_index=True)



    return my_df
def predict_word_improve(pre_predict_word):

    common_word_score = apply_levenshtein_distance_on_list(pre_predict_word,common_english_words)

    min_score = common_word_score['score'].min()

    canidates = common_word_score[common_word_score['score'] == min_score]['word']

    

    if min_score >= len(pre_predict_word):

        raise ValueError 

        

    return pd.merge(canidates,df_count,on='word').head(1)['word']
#print(predict_word_improve('gool'))

#print(predict_word_improve('foog'))

print(predict_word_improve('fccd'))
def split_video():

    images_list = []

    

    return images_list

    

def receive():

    pass



def publish():

    pass