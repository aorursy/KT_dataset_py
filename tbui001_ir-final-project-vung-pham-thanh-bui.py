# from nltk.corpus import stopwords,words

# from nltk.tokenize import word_tokenize

# from string import punctuation



# def process_sentence(sentence, tokens_to_remove, English_words, max_tokens=None):

#      words = word_tokenize(sentence) # Tokenize

#      if max_tokens is not None and len(words) < max_tokens:

#         return None

#      else:

#          words = [w.lower() for w in words if not w.isdigit()] # Convert to lowercase and also remove digits

#          filter_words = [w for w in words if w not in tokens_to_remove and w in English_words] # remove tokens + check english words + stem

#          return filter_words
# %%time

# import csv

# max_docs = 30000 # test with this number of docs first. If would like to do for all docs, set this value to None

# review_outfile = 'review_text.txt'



# tokens_to_remove = set(punctuation)

# English_words = set(words.words())





# doc_count = 0



# with open('../input/amazon-review-testset/test.csv','rt',encoding='utf-8') as rf:

#      with open(review_outfile, 'w') as outputfile:

#          reader = csv.reader(rf, delimiter=',')

#          for row in reader:

#              score = int(row[0])-1 #fit into class for easier work later

#              review = process_sentence(row[1], tokens_to_remove, English_words, 100)     

#              if review is not None:

#                  outputfile.writelines(str(score) + ", " + " ".join(review) + '\n') # write the results

#                  doc_count += 1

#                  if  max_docs and doc_count >= max_docs: # if we do define the max_docs

#                      break

#      outputfile.close()

# rf.close()
# # View the file if needed

# from IPython.display import FileLink

# FileLink('review_text.txt')
# from sklearn.model_selection import train_test_split

# import numpy as np

# from numpy import savetxt



# with open("review_text.txt", "rt") as infile:

#      data = infile.read().split('\n')

        

# data = np.array(data)



# data = np.delete(data,-1) 



# train_data,test_data = train_test_split(data,test_size=0.15,random_state = 1) #set the seed to 1 for reproducibility



# #Save data into csv files

# savetxt('train_set.csv', train_data, delimiter=',',fmt='%s')

# savetxt('test_set.csv', test_data, delimiter=',', fmt='%s')
# # View the file if needed

# FileLink('train_set.csv')
# # View the file if needed

# FileLink('test_set.csv')
# # GloVe model import

# import os

# import numpy as np



# embeddings_index = {}

# f = open(os.path.join('../input/glove-global-vectors-for-word-representation', 'glove.6B.200d.txt'))

# for line in f:

#     values = line.split()

#     word = values[0]

#     coefs = np.asarray(values[1:])

#     embeddings_index[word] = coefs

# f.close()



# print('Found %s word vectors.' % len(embeddings_index))
# import pandas as pd





# train_df = pd.read_csv("train_set.csv", header=None)

# train_df.columns = ["label", "text"]

# class_label = train_df['label'].values.astype(str)

# train_document = train_df['text'].values
# print(len(np.amax(train_document)))
# MAX_SEQUENCE_LENGTH = len(np.amax(train_document))

# MAX_NUM_WORDS = 25000 #we set a limitation of word to import to speed up
# # Read data

# from keras.preprocessing.text import Tokenizer



# # finally, vectorize the text samples into a 2D integer tensor

# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# tokenizer.fit_on_texts(train_document)

# sequences = tokenizer.texts_to_sequences(train_document)



# word_index = tokenizer.word_index

# print('Found %s unique tokens.' % len(word_index))
# from keras.preprocessing.sequence import pad_sequences

# from keras.utils import to_categorical



# train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)



# train_labels = to_categorical(np.asarray(class_label))

# print('Shape of data tensor:', train_data.shape)

# print('Shape of label tensor:', train_labels.shape)



# print('Preparing embedding matrix.')
# # Import test data

# test_df = pd.read_csv('test_set.csv', header=None)

# test_df.columns = ['label', 'text']

# test_label = test_df['label'].values

# test_document = test_df['text']

# trans_test =  tokenizer.texts_to_sequences(test_document)
# from keras.preprocessing.sequence import pad_sequences





# test_data = pad_sequences(trans_test, maxlen=MAX_SEQUENCE_LENGTH)
# print(len(test_data))
# # prepare embedding matrix

# num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

# embedding_dim = 200



# embedding_matrix = np.zeros((num_words, embedding_dim))

# for word, i in word_index.items():

#     if i >= MAX_NUM_WORDS:

#         continue

#     embedding_vector = embeddings_index.get(word)

#     if embedding_vector is not None:

#         # words not found in embedding index will be all-zeros.

#         embedding_matrix[i] = embedding_vector

#     else:

#         embedding_matrix[i] = np.random.randn(embedding_dim)
# from keras.models import Model

# from keras.layers import Dense, Dropout, Activation,Flatten

# from keras.layers import Embedding, Concatenate, Input, Reshape

# from keras.layers import Conv2D, MaxPool2D

# from keras.callbacks import EarlyStopping, ModelCheckpoint

# from keras import regularizers

# from keras.initializers import Constant



# num_filters = 150

# filter_size = 2



# start_model = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype='int32')



# embedding_layer = Embedding(num_words, embedding_dim,embeddings_initializer=Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH,trainable=True)(start_model)

# reshape_layer = Reshape((MAX_SEQUENCE_LENGTH, embedding_dim, 1))(embedding_layer)



# Convo2D_1 = Conv2D(num_filters, kernel_size =(3,embedding_dim),activation='relu')(reshape_layer)

# max_pooling_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH-2,1), strides=(filter_size,filter_size),padding='valid')(Convo2D_1)

# # model.add(Dropout(0.1))



# Convo2D_2 = Conv2D(num_filters, kernel_size =(4,embedding_dim),activation='relu')(reshape_layer)

# max_pooling_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH-3,1), strides=(filter_size,filter_size),padding='valid')(Convo2D_2)

# # model.add(Dropout(0.1))



# Convo2D_3 = Conv2D(num_filters, kernel_size =(5,embedding_dim),activation='relu')(reshape_layer)

# max_pooling_3 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH-4,1), strides=(filter_size,filter_size),padding='valid')(Convo2D_3)

# # model.add(Dropout(0.1))



# Concatenate_layer = Concatenate(axis=1)([max_pooling_1,max_pooling_2,max_pooling_3])

# Flatten_layer = Flatten()(Concatenate_layer)

# # model.add(Dense(256,activation = 'relu', name = 'Dense_1'))



# Dropout_layer = Dropout(0.5)(Flatten_layer)

# Output_layer = Dense(units=5, activation='softmax')(Dropout_layer)
# model = Model(inputs=start_model, outputs=Output_layer)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())
# from keras.utils.vis_utils import plot_model



# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# FileLink('model_plot.png')
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy',save_best_only=True, mode='max')



# history = model.fit(train_data, train_labels, batch_size=64, epochs=30,verbose = 1,callbacks=[early_stopping,mc],validation_split=0.2)
# from keras.models import load_model



# saved_model = load_model('best_model.h5')
# import matplotlib.pyplot as plt



# # Plot history: MAE

# plt.plot(history.history['loss'], label='MAE (testing data)')

# plt.plot(history.history['val_loss'], label='MAE (validation data)')

# plt.title('MAE for Document Classification Levels')

# plt.ylabel('MAE value')

# plt.xlabel('Number of epoch')

# plt.legend(loc="upper left")

# plt.show()
# # Plot history: MSE

# plt.plot(history.history['accuracy'], label='MSE (testing data)')

# plt.plot(history.history['val_accuracy'], label='MSE (validation data)')

# plt.title('MSE for Document Classification Levels')

# plt.ylabel('MSE value')

# plt.xlabel('Number of epoch')

# plt.legend(loc="upper left")

# plt.show()
import numpy as np

import pandas as pd
gs_results = pd.read_csv('../input/results-for-viz/BPNN_GridSearch.csv', header=None);
gs_results
grid_search_results = gs_results.values
y = [float(a) for a in grid_search_results[:, 4]]
x = grid_search_results[:,0]
import matplotlib.pyplot as plt
with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 10))

    plt.plot(x, y, '-o', color='steelblue')

    idx = np.argmax(y)

    plt.plot(x[idx], y[idx], 'P', color='red', ms=10)

    plt.xlabel("Simple NN hyper-parameters")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=90)

    plt.show()
with open('../input/results-for-viz/CNN_GridSearch.csv', 'r') as f:

    grid_search_lines = f.readlines()
grid_search_results = np.array([line.replace("\n", "").replace("{","").replace("}","").split(", 'mean accuracy': ") for line in grid_search_lines])
grid_search_results
y = [float(a) for a in grid_search_results[:, 1]]
y
x = grid_search_results[:,0]
x = [elm.split(',')[1] + elm.split(',')[3] + elm.split(',')[4] for elm in x]
x
with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 10))

    plt.plot(x, y, '-o', color='steelblue')

    idx = np.argmax(y)

    plt.plot(x[idx], y[idx], 'P', color='red', ms=10)

    plt.xlabel("CNN1D Simple Model")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=90)

    plt.show()
import pandas as pd

import numpy as np
train_data = pd.read_csv('../input/results-for-viz/train_data.csv', header=None)
test_data = pd.read_csv('../input/results-for-viz/test_data.csv', header=None)
train_label = pd.read_csv('../input/results-for-viz/train_label.csv', header=None)
test_label = pd.read_csv('../input/results-for-viz/test_label.csv', header=None)
train_label.head()
test_label.head()
type(test_label.values)
len(test_label)
# Load model

from keras.models import load_model

model = load_model('../input/results-for-viz/best_model.h5')
saved_model = model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



prediction = saved_model.predict(test_data)

predicted_class = np.argmax(prediction, axis=1)
test_label = test_label.values
from sklearn.metrics import classification_report
print(classification_report(predicted_class, test_label))
# Import test data

test_df = pd.read_csv('../input/results-for-viz/test_set.csv', header=None)

test_df.columns = ['label', 'text']

test_label = test_df['label'].values.astype(str)

test_document = test_df['text']
def confusion_type_for_class(cls, true_label, predicted_label):

    result={}

    result["true_positive"] = [true_label[i] == cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_positive"] = [true_label[i] != cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_negative"] = [true_label[i] == cls and predicted_label[i] != cls for i in range(len(true_label))]

    result["true_negative"] = [true_label[i] != cls and predicted_label[i] != cls for i in range(len(true_label))]

    return result
def visualize_confusion_results(cls, test_df, true_label, predicted_label):

    confusion_result = confusion_type_for_class(cls, true_label, predicted_label)

    for cf in confusion_result.keys():

        texts = test_df[confusion_result[cf]]['text'].values

        if len(texts) == 0:

            import pdb

            pdb.set_trace()

        if len(texts) > 0:

            show_wordcloud(" ".join(texts), f"Test set class {cls}, type {cf}")
import wordcloud

import matplotlib.pyplot as plt

def show_wordcloud(text, title=None):

    # Create and generate a word cloud image:

    wc = wordcloud.WordCloud(background_color='white').generate(text)

    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    if title is not None:

        plt.title(title)

    plt.show()
np.unique(test_label)
predicted_class = [str(c) for c in predicted_class]
for cls in np.unique(test_label):

    visualize_confusion_results(cls, test_df, test_label, predicted_class)
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# prediction = saved_model.predict(test_data)

# predicted_class = np.argmax(prediction, axis=1)
# print(len(predicted_class))

# print(len(test_label))
# # Print out the scores

# precision =precision_score(predicted_class, test_label, average = 'weighted')

# recall = recall_score(predicted_class,test_label, average = 'weighted')

# F1_score = f1_score(predicted_class,test_label, average = 'weighted')

# Accuracy = accuracy_score(predicted_class,test_label)



# # output_string = "{},{:.2f},{:.2f},{:.2f},{:.2f}".format(precision,recall,F1_score,Accuracy)

# # BPNN_Test_array.append(output_string)



# print("Precision score: {:.2f}".format(precision))

# print("Recall score: {:.2f}".format(recall))

# print("F1 Score: {:.2f}".format(F1_score))

# print("Accuracy Score: {:.2f}".format(Accuracy))
# #Save transformed data into csv files

# savetxt('train_data.csv', train_data, delimiter=',',fmt='%s')

# savetxt('test_data.csv', test_data, delimiter=',', fmt='%s')



# #Save transformed label into csv files

# savetxt('train_label.csv', train_labels, delimiter=',',fmt='%s')

# savetxt('test_label.csv', test_label, delimiter=',', fmt='%s')
# FileLink('train_data.csv')
# FileLink('test_data.csv')
# FileLink('train_label.csv')
# FileLink('test_label.csv')
# #Save built model using pickle

# import pickle



# # Save to file in the current working directory

# pkl_filename = "CNNText_model.pkl"

# with open(pkl_filename, 'wb') as file:

#     pickle.dump(model, file)
# FileLink('CNNText_model.pkl')