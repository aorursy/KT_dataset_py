from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import string
from keras.applications.resnet50 import ResNet50
from pickle import dump
from pickle import load
from IPython.display import Image
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.layers.merge import add, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# Defining the Dataset path
image_dataset_path = '../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset'
caption_dataset_path = '../input/flickr8k-imageswithcaptions/Flickr8k_text/Flickr8k.token.txt'
# Visulaizing the Image Data
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/1000268201_693b08cb0e.jpg')
# load the caption file & read it
def load_caption_file(path):
    
    # dictionary to store captions
    captions_dict = {}
    
    # iterate through the file
    for caption in open(path):
        
        # Splitting the name of image file and image caption
        tokens = caption.split()
        caption_id, caption_text = tokens[0].split('.')[0], tokens[1:]
        caption_text = ' '.join(caption_text)
        
        # save it in the captions dictionary
        if caption_id not in captions_dict:
            captions_dict[caption_id] = caption_text
        
    return captions_dict

# call the function
captions_dict = load_caption_file(caption_dataset_path)
# dictionary to store the cleaned captions
new_captions_dict = {}

# prepare translation table for removing punctuation.
table = str.maketrans('', '', string.punctuation)

# loop through the dictionary
for caption_id, caption_text in captions_dict.items():
    # tokenize the caption_text
    caption_text = caption_text.split()
    # convert it into lower case
    caption_text = [token.lower() for token in caption_text]
    # remove punctuation from each token
    caption_text = [token.translate(table) for token in caption_text]
    # remove all the single letter tokens like 'a', 's'
    caption_text = [token for token in caption_text if len(token)>1]
    # store the cleaned captions
    new_captions_dict[caption_id] = 'startseq ' + ' '.join(caption_text) + ' endseq'
    
# delete unwanted captions which do not match any images
del captions_dict
print('"' + list(new_captions_dict.keys())[0] + '"' + ' : ' + new_captions_dict[list(new_captions_dict.keys())[0]])
caption_images_list = []

image_index = list(new_captions_dict.keys())

caption_images_list = [ image.split('.')[0] for image in os.listdir(image_dataset_path) if image.split('.')[0] in image_index ]
caption_images_list[0]
# Total images along with captions
len(caption_images_list)
train_validate_images = caption_images_list[0:8081]  
test_images = caption_images_list[8081:8091]
test_images
# extract features from each photo in the directory
def extract_features1(directory, image_keys):
    # load the model
    model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    
    # model summary
    print(model.summary())
    
    # extract features from each photo
    features = dict()
    
    for name in image_keys:
        
        # load an image from file
        filename = directory + '/' + name + '.jpg'
        
        # load the image and convert it into size accepted by ResNet model
        image = load_img(filename, target_size=(224, 224))
        
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        
        # reshape data for the model.
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        # prepare the image for the ResNet model
        image = preprocess_input(image)
        
        # get features
        feature = model.predict(image, verbose=0)
        
        # get image id
        image_id = name.split('.')[0]
        
        # store feature
        features[image_id] = feature
         

    return features
# Note: This section takes time as it is processing the entire dataset.
train_validate_features1 = extract_features1(image_dataset_path, train_validate_images)
# Printing the features extracted from a sample image.
print("{} : {}".format(list(train_validate_features1.keys())[0], train_validate_features1[list(train_validate_features1.keys())[0]] ))
len(train_validate_features1)
# Converting the features into pickle format for later uses.
dump(train_validate_features1, open('./train_validate_features1.pkl', 'wb'))
# Loading the features from pickle format.
train_validate_features1 = load(open('./train_validate_features1.pkl', 'rb'))
# make a dictionary of image with caption for train_validate_images
train_validate_image_caption = {}

for image, caption in new_captions_dict.items():
    
    # check whether the image is available in both train_validate_images list and train_validate_features dictionary
    if image in train_validate_images and image in list(train_validate_features1.keys()):
        
         train_validate_image_caption.update({image : caption})

len(train_validate_image_caption)
list(train_validate_image_caption.values())[2]
Image(image_dataset_path+'/'+list(train_validate_image_caption.keys())[2]+'.jpg')
# initialise tokenizer
tokenizer = Tokenizer()

# create word count dictionary on the captions list
tokenizer.fit_on_texts(list(train_validate_image_caption.values()))

# Creating the vocabulary
vocab_len = len(tokenizer.word_index) + 1

# Store the length of the maximum sentence
max_len = max(len(train_validate_image_caption[image].split()) for image in train_validate_image_caption)

print("vocab_len ", vocab_len)
print("max_len ", max_len)

def prepare_data(image_keys):
    
    # x1 will store the image feature, x2 will store one sequence and y will store the next sequence
    x1, x2, y = [], [], []

    # iterate through all the images 
    for image in image_keys:

        # store the caption of that image
        caption = train_validate_image_caption[image]

        # split the image into tokens
        caption = caption.split()

        # generate integer sequences of the
        seq = tokenizer.texts_to_sequences([caption])[0]

        length = len(seq)

        for i in range(1, length):

            x2_seq, y_seq = seq[:i] , seq[i]  

            # pad the sequences
            x2_seq = pad_sequences([x2_seq], maxlen = max_len)[0]


            # encode the output sequence                
            y_seq = to_categorical([y_seq], num_classes = vocab_len)[0]

            x1.append( train_validate_features1[image][0] )

            x2.append(x2_seq)

            y.append(y_seq)
               
    return np.array(x1), np.array(x2), np.array(y)
train_x1, train_x2, train_y = prepare_data( train_validate_images[0:7081] )
validate_x1, validate_x2, validate_y = prepare_data( train_validate_images[7081:8081] )
len(train_x1)
embedding_size = 128
image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(Dropout(0.5))
image_model.add(RepeatVector(max_len))

image_model.summary()
language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_len, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256,return_sequences=True))
language_model.add(Dropout(0.5))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()
# Final Model
conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, dropout=0.5, recurrent_dropout=0.5,return_sequences=True)(conca)
x = LSTM(512, dropout=0.5, recurrent_dropout=0.5,return_sequences=False)(x)
x = Dense(vocab_len)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()
# define checkpoint callback to save the best model only
filepath = './model.h5'

callbacks = [ ModelCheckpoint(filepath= filepath, verbose = 2,save_best_only=True, monitor='val_loss', mode='min') ]

print("shape of train_x1 ", train_x1.shape)
print("shape of train_x2 ", train_x2.shape)
print("shape of train_y ", train_y.shape)
print()
print("shape of validate_x1 ", validate_x1.shape)
print("shape of validate_x2 ", validate_x2.shape)
print("shape of validate_y ", validate_y.shape)
BATCH_SIZE = 512

# Define training epochs
EPOCHS = 100

history = model.fit([train_x1, train_x2],  
                    train_y,              
                    verbose = 1,            
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = callbacks, 
                    validation_data=([validate_x1, validate_x2], validate_y)) 
# plot the training artifacts

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()
# extract features from each photo in the directory
def extract_feat(filename):
    # load the model
    model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizr):
    for word, index in tokenizr.word_index.items():
        if index == integer:
            return word
    return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text
# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3561433412_3985208d53.jpg')  

    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','') 
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3561433412_3985208d53.jpg')
# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3229821595_77ace81c6b.jpg')  

    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','')
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3229821595_77ace81c6b.jpg')
# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/554526471_a31f8b74ef.jpg')  

    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','')
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/554526471_a31f8b74ef.jpg')

# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/1057089366_ca83da0877.jpg')  

    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','')
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/1057089366_ca83da0877.jpg')

# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3613667665_1881c689ea.jpg')
    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','')
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3613667665_1881c689ea.jpg')
# generate a description for an image
model = load_model('./model.h5')
tokenizr = Tokenizer()
tokenizr.fit_on_texts([caption for image, caption in new_captions_dict.items() if image in train_validate_images])
max_length = max_len

photo = extract_feat('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3422146099_35ffc8680e.jpg')
    # seed the generation process
in_text = 'startseq'
    # iterate over the whole length of the sequence
for i in range(max_length):
        # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
    yhat = np.argmax(yhat)
        # map integer to word
    word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
    if word is None:
        break
        # append as input for generating the next word
    in_text += ' ' + word
        # stop if we predict the end of the sequence
    if word == 'endseq':
        break
in_text = in_text.replace('startseq','') 
in_text = in_text.replace('endseq','')
print("Predicted caption -> ", in_text)
print()
print('*********************************************************************')
Image('../input/flickr8k-imageswithcaptions/Flickr8k_Dataset/Flicker8k_Dataset/3422146099_35ffc8680e.jpg')