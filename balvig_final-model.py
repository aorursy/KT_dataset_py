from os import listdir
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, add
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from pickle import dump, load
import string
# extract features from each photo in the directory
def extract_features(data_path):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(data_path):
        filename = data_path + "/" + name
        image1= name.split(".")
        image_id=image1[0]
        if not image1[1]=='jpg':
            continue
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)  # this is (224, 224, 3)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])  # this is (1, 224, 224, 3)
        image = preprocess_input(image)
        feature = model.predict(image)
        features[image_id] = feature
        print(">%s" % name)
    return features
#directory='../input/flickr8k/flickr8k/Flicker8k_Dataset'
#features = extract_features(directory)
#print('Extracted Features: %d' % len(features))
# save to file
#dump(features, open('features.pkl', 'wb'))
def load_doc(filepath):
    with open(filepath, 'r') as ifp:
        text = ifp.read()
    return  text

def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id = tokens[0].split('.')[0]
        image_desc = ' '.join(tokens[1:])
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping
filepath = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filepath)
descriptions = load_descriptions(doc)
print("loaded %d descriptions" % len(descriptions))
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i, desc in enumerate(desc_list):
            desc = desc.split()
            desc = [w.lower() for w in desc]
            desc = [w.translate(table) for w in desc]
            desc = [w for w in desc if len(w) > 1]
            desc = [w for w in desc if w.isalpha()]
            descriptions[key][i] = ' '.join(desc)

def to_vocabulary(descriptions):
    words = set()
    for key in descriptions.keys():
        for d in descriptions[key]:
            words.update(d.split())
    return words

def save_descriptions(descriptions, output_filepath):
    with open(output_filepath, 'w') as ofp:
        for key,desc_list in descriptions.items():
            for d in desc_list:
                ofp.write(key + ' ' + d + '\n')
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
print("vocabulary size : %d" % len(vocabulary))
save_descriptions(descriptions, 'cleaned_descriptions.txt')
def load_identifiers(filepath):
    doc = load_doc(filepath)
    ids = set()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        ids.add(line.split('.')[0])
    return ids

def load_clean_descriptions(filepath, ids):
    doc = load_doc(filepath)
    descriptions = {}
    for line in doc.split('\n'):
        tokens = line.split()
        if len(tokens) < 1:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in ids:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filepath, ids):
    features = load(open(filepath, 'rb'))
    return {k:features[k] for k in ids}
filepath = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train_ids = load_identifiers(filepath)
print('Dataset: %d' % len(train_ids))
train_descriptions = load_clean_descriptions('cleaned_descriptions.txt', train_ids)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load_photo_features('../input/features/features.pkl', train_ids)
print('Photos: train=%d' % len(train_features))
def to_lines(descriptions):
    desc_list = []
    for key in descriptions:
        for d in descriptions[key]:
            desc_list.append(d)
    return desc_list

def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(to_lines(descriptions))
    return tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size based on tokenizer from train data: %d' % vocab_size)
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)
max_length = max_length(train_descriptions)
print('Max description Length: %d' % max_length)
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
# dev data
filepath = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'
test_ids = load_identifiers(filepath)
print("Dataset: %d" %  len(test_ids))
test_descriptions = load_clean_descriptions('cleaned_descriptions.txt', test_ids)
print("Descriptions: test = %d" % len(test_descriptions))
test_features = load_photo_features('../input/features/features.pkl', test_ids)
print('Photos: test=%d' % len(test_features))
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
from keras.layers.wrappers import Bidirectional

def define_model(vocab_size, max_length):
    inputs1 = Input(shape = (4096, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    inputs2 = Input(shape = (max_length, ))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(256))(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs = [inputs1, inputs2], outputs = outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    #plot_model(model, to_file = 'res/model.png', show_shapes=True)
    return model


def data_generator(tokenizer, max_length, descriptions, photos):
    while True:
       for key, desc_list in descriptions.items():
           photo = photos[key][0]
           in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
           yield [[in_img, in_seq], out_word]
# fit model
"""
model = define_model(vocab_size, max_length)
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
generator = data_generator(tokenizer, max_length, train_descriptions, train_features)
num_epochs = 21
for i in range(num_epochs):
    model.fit_generator(generator, epochs=1, steps_per_epoch = len(train_descriptions), verbose=1) 
    if(i%5==0):
        model.save('model_' + str(i) + '.h5')
"""
#Image.open('model.png')
#from subprocess import check_output
#print(check_output(["ls","../lib"]).decode("utf8"))

from PIL import Image
from keras.models import load_model
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
generator = data_generator(tokenizer, max_length, train_descriptions, train_features)
model=load_model('../input/bilstm-models/model_20 (1).h5')
model.summary()
#plot_model(model,to_file='model.png')
#Image.open('model.png')
num_epochs = 11
for i in range(num_epochs):
    model.fit_generator(generator, epochs=1, steps_per_epoch = len(train_descriptions), verbose=1)
    if(i%5==0):
        model.save('model_' + str(i) + '.h5')
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        #print(line)
        #print(tokens)
        # split id from description
        if len(tokens)==0:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
                # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                # store
            descriptions[image_id].append(desc)
    return descriptions

# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
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
        yhat = argmax(yhat)
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

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        #calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# prepare tokenizer on train set

# load training dataset (6K)
filename = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('cleaned_descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('cleaned_descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('../input/features/features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
#filename = '../input/bilstm-models/model_15.h5'
#model = load_model(filename)
# evaluate model
#evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
# load training dataset (6K)
filename = '../input/flickr8k/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('cleaned_descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
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

tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
filename = '../input/bilstm-models/model_20 (1).h5'
model = load_model(filename)
photo = extract_features('../input/images/35506150_cbdb630f4f.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)

photo = extract_features('../input/images/47871819_db55ac4699.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('../input/images/42637987_866635edf6.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('../input/images/54501196_a9ac9d66f2.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('../input/images/55135290_9bed5c4ca3.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('../input/images/41999070_838089137e.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('../input/images/50030244_02cd4de372.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)