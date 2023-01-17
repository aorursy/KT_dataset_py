# from os import listdir
# from pickle import dump
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model

# # extract features from each photo in the directory
# def extract_features(directory):
# 	# load the model
# 	model = VGG16()
# 	# re-structure the model
# 	model.layers.pop()
# 	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
# 	# summarize
# 	print(model.summary())
# 	# extract features from each photo
# 	features = dict()
# 	for name in listdir(directory):
# 		# load an image from file
# 		filename = directory + '/' + name
# 		image = load_img(filename, target_size=(224, 224))
# 		# convert the image pixels to a numpy array
# 		image = img_to_array(image)
# 		# reshape data for the model
# 		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 		# prepare the image for the VGG model
# 		image = preprocess_input(image)
# 		# get features
# 		feature = model.predict(image, verbose=0)
# 		# get image id
# 		image_id = name.split('.')[0]
# 		# store feature
# 		features[image_id] = feature
# 		print('>%s' % name)
# 	return features

# # extract features from all images
# # directory = '/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images'
# # features = extract_features(directory)
# # print('Extracted Features: %d' % len(features))
# # # save to file
# # dump(features, open('/kaggle/features.pkl', 'wb'))
import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
 

filename = '/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
print('Text data loaded !')

# extract descriptions for images

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

# clean descriptions
clean_descriptions(descriptions)
print('Cleaned !')

def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# save descriptions
save_descriptions(descriptions, '/kaggle/descriptions.txt')
print('Done !')

 

from pickle import load

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
		# split id from description
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

print('Done !')
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='/kaggle/model.png', show_shapes=True)
	return model

print('Model Defined')
# from numpy import array
# from pickle import load
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.utils import plot_model
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Embedding
# from keras.layers import Dropout
# from keras.layers.merge import add
# from keras.callbacks import ModelCheckpoint
# def to_lines(descriptions):
# 	all_desc = list()
# 	for key in descriptions.keys():
# 		[all_desc.append(d) for d in descriptions[key]]
# 	return all_desc
 
# # fit a tokenizer given caption descriptions
# def create_tokenizer(descriptions):
# 	lines = to_lines(descriptions)
# 	tokenizer = Tokenizer()
# 	tokenizer.fit_on_texts(lines)
# 	return tokenizer

# def create_sequences(tokenizer, max_length, descriptions, photos):
# 	X1, X2, y = list(), list(), list()
# 	# walk through each image identifier
# 	for key, desc_list in descriptions.items():
# 		# walk through each description for the image
# 		for desc in desc_list:
# 			# encode the sequence
# 			seq = tokenizer.texts_to_sequences([desc])[0]
# 			# split one sequence into multiple X,y pairs
# 			for i in range(1, len(seq)):
# 				# split into input and output pair
# 				in_seq, out_seq = seq[:i], seq[i]
# 				# pad input sequence
# 				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# 				# encode output sequence
# 				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# 				# store
# 				X1.append(photos[key][0])
# 				X2.append(in_seq)
# 				y.append(out_seq)
# 	return array(X1), array(X2), array(y)

# # calculate the length of the description with the most words
# def max_length(descriptions):
# 	lines = to_lines(descriptions)
# 	return max(len(d.split()) for d in lines)


# # load training dataset (6K)
# filename = 'Flickr_TextData/Flickr_8k.trainImages.txt'
# train = load_set(filename)
# print('Dataset: %d' % len(train))
# # descriptions
# train_descriptions = load_clean_descriptions('/kaggle/descriptions.txt', train)
# print('Descriptions: train=%d' % len(train_descriptions))

# # prepare tokenizer
# tokenizer = create_tokenizer(train_descriptions)
# vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary Size: %d' % vocab_size)

# # photo features
# train_features = load_photo_features('/kaggle/features.pkl', train)
# print('Photos: train=%d' % len(train_features))

# print('Vocabulary Size: %d' % vocab_size)
# # determine the maximum sequence length
# max_length = max_length(train_descriptions)
# print('Description Length: %d' % max_length)
# # prepare sequences
# X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# # dev dataset

# # load test set
# filename = 'Flickr_TextData/Flickr_8k.devImages.txt'
# test = load_set(filename)
# print('Dataset: %d' % len(test))
# # descriptions
# test_descriptions = load_clean_descriptions('descriptions.txt', test)
# print('Descriptions: test=%d' % len(test_descriptions))
# # photo features
# test_features = load_photo_features('features.pkl', test)
# print('Photos: test=%d' % len(test_features))
# # prepare sequences
# X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# # define checkpoint callback
# filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# # fit model
# model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

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

# create sequences of images, input sequences and output words for an image for single generator
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
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
    return array(X1), array(X2), array(y)


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]

            
# Create sequences with batch modification

def create_sequences_batch(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions:
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


# data generator with batch modification
def data_generator_batch(descriptions, photos, tokenizer, max_length, batch_size):
    # loop for ever over images
    while 1:
        for i in range(0,len(descriptions)-batch_size,batch_size):
            # retrieve the photo feature
            in_img, in_seq, out_word = create_sequences_batch(tokenizer, max_length, list(descriptions.items())[i:i+batch_size], photos)
            yield [[in_img, in_seq], out_word]




# load training dataset (6K)
filename = '/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('/kaggle/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# photo features
train_features = load_photo_features('/kaggle/input/image-features/features.pkl', train)
print('Photos: train=%d' % len(train_features))

print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# dev dataset

# load test set
filename = '/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('/kaggle/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('/kaggle/input/image-features/features.pkl', test)
print('Photos: test=%d' % len(test_features))

print('\n\n\nDone !')
# Model definition

# model = define_model(vocab_size,max_length)

# print('Done !')
# Start training

# batch_size=30
# epochs = 20
# steps = len(train_descriptions)/batch_size
# for i in range(epochs):
#     # create the data generator
#     generator = data_generator_batch(train_descriptions, train_features, tokenizer, max_length, batch_size)
#     # fit for one epoch
#     model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#     # save model
#     model.save('/kaggle/captioner_' + str(i) + '.h5')
model.save('/kaggle/captioner1'+'.h5')
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

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

#Beam search based description generation
def beam_generate_desc(model, tokenizer, photo, max_length, beam_idx):
    in_text = 'startseq'
    k=beam_idx
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([photo,sequence], verbose=0)
    yhat = yhat[0].argsort()[::-1][:k]
#     yhat = (np.argpartition(yhat,-k)[:k])[0]
    words=[]
    for i in range(k):
        words.append(word_for_id(yhat[i], tokenizer))
    in_text = list(list())
    in_text = [['startseq'] for i in range(k)]
    for i in range(k):
        in_text[i].append(words[i])
    for p in range(max_length-1):
        total_seq = []
        for i in range(k):
#             print(' '.join(in_text[i]))
            if(in_text[i][-1]=='endseq'):
                continue
            sequence = tokenizer.texts_to_sequences([' '.join(in_text[i])])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = argmax(yhat)
            word = word_for_id(yhat, tokenizer)
            in_text[i].append(word)
    return in_text
    

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in list(descriptions.items())[:10]:
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        pil_im = Image.open('/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images/'+str(key)+'.jpg')
        im_array = np.asarray(pil_im)
        plt.imshow(im_array)
        plt.show()
        print(' '.join((yhat.split()[1:-1])))
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    
# evaluate the skill of the model for beam
def evaluate_model_beam(model, descriptions, photos, tokenizer, max_length, beam_idx=3):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in list(descriptions.items())[:25]:
        # generate description
        yhat = beam_generate_desc(model, tokenizer, photos[key], max_length,beam_idx)
        pil_im = Image.open('/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images/'+str(key)+'.jpg')
        im_array = np.asarray(pil_im)
        plt.imshow(im_array)
        plt.show()
        for yhat1 in yhat:
            print(' '.join((yhat1[1:-1])))
            # store actual and predicted
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat1)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    
print('Done !')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
evaluate_model_beam(model, test_descriptions, test_features, tokenizer, max_length, beam_idx=10)
# evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

