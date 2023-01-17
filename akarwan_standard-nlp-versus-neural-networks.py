# Import the necessary libraries 

import numpy as np

import pandas as pd # Dataframe Management

import matplotlib.pyplot as plt 

import seaborn as sns # Visualization

from sklearn.model_selection import train_test_split 

import pickle # Model Serialization
# Load Data

df = pd.read_csv('../input/dataset/spam_or_ham.csv', delimiter=',', encoding='latin-1')
# Exploratory Analysis

# View Dataset, top 10 Text Messages

df.head(n=10)
# Check Distribution - Not Balanced Data

sns.countplot(df.Label)

plt.xlabel('Label')

plt.title('Number of ham and spam messages')

# 20% Spam Data
# Word Cloud - Install From Terminal

# conda install -c conda-forge wordcloud (if Anaconda is installed)



import matplotlib.pyplot as plt

from wordcloud import WordCloud 



# spam and ham words

spam_words = ' '.join(list(df[df['Label'] == 'spam']['Text']))

ham_words = ' '.join(list(df[df['Label'] == 'ham']['Text']))



# Create Word Clouds 

spam_wc = WordCloud(width = 512, height = 512, colormap = 'plasma').generate(spam_words)

ham_wc = WordCloud(width = 512, height = 512, colormap = 'ocean').generate(ham_words)



# Plot Word Clouds

# SPAM

plt.figure(figsize = (10,8), facecolor = 'r')

plt.imshow(spam_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()



# HAM 

plt.figure(figsize = (10,8), facecolor = 'g')

plt.imshow(ham_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()



# In Spam Messages word FREE occurs very oftenly

# In Ham Messages words 'OK', 'will', 'got' occur often and corrupted words ('gt' or 'lt')
# Split Data Set into Train and Test

# Train 70%

# Test  30%

# random_state setup means same values every time split is perfomed 

X_train, X_test, Y_train, Y_test = train_test_split(df.Text, df.Label, test_size=0.3, random_state=123)
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB # Naive Bayes

from sklearn.svm import LinearSVC, SVC # Support Vector Machine

from sklearn.ensemble import RandomForestClassifier # Random Forest

import time



# Python Function

def models(list_sentences_train, list_sentences_test, train_labels, test_labels):

    t0 = time.time() # start time

    

    # Pipeline 

    model = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))), 

                      ('tfidf', TfidfTransformer(use_idf=False)),

                      ('clf', MultinomialNB())]) # Naive Bayes

    #                  ('clf', SVC(kernel='linear', probability=True))]) # Linear SVM with probability

    #                  ('clf', RandomForestClassifier())]) # Random Forest



    # Train Model

    model.fit(list_sentences_train, train_labels) 

    

    duration = time.time() - t0 # end time

    print("Training done in %.3fs " % duration)



    # Model Accuracy

    print('Model final score: %.3f' % model.score(list_sentences_test, test_labels))

    return model



# Train, Evaluate and Save Model

model_std_NLP = models(X_train, X_test, Y_train, Y_test)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



from sklearn.metrics import confusion_matrix # Library to Compute Confusion Matrix



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
np.set_printoptions(precision=2)



# Predictions with model

Y_pred = model_std_NLP.predict(X_test)

class_names = np.array(['ham', 'spam'])





# Plot non-normalized confusion matrix

plot_confusion_matrix(Y_test, Y_pred, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
# Save to file in the current working directory

pkl_filename = "pickle_model.pkl"  

with open(pkl_filename, 'wb') as file:  

    pickle.dump(model_std_NLP, file)



# Load from file

with open(pkl_filename, 'rb') as file:  

    pickle_model = pickle.load(file)
test_text_spam = ['Urgent! call 09066350750 from your landline. Your complimentary 4* Ibiza Holiday or 10,000 cash await collection SAE T&Cs PO BOX 434 SK3 8WP 150 ppm 18+ ']

test_text_ham = ['Good. No swimsuit allowed :)']



# Predict Category and Probability

# Spam

print(model_std_NLP.predict(test_text_spam)) 

print(model_std_NLP.predict_proba(test_text_spam)) 



# Ham

print(model_std_NLP.predict(test_text_ham)) 

print(model_std_NLP.predict_proba(test_text_ham)) 



# More Test Examples

# Ham - 0

# Good. No swimsuit allowed :)

# Wish i were with you now!

# Im sorry bout last nite it wasnÃ¥Ãt ur fault it was me, spouse it was pmt or sumthin! U 4give me? I think u shldxxxx



# Spam - 1

# Urgent! call 09066350750 from your landline. Your complimentary 4* Ibiza Holiday or 10,000 cash await collection SAE T&Cs PO BOX 434 SK3 8WP 150 ppm 18+ 

# +123 Congratulations - in this week's competition draw u have won the Ã¥Â£1450 prize to claim just call 09050002311 b4280703. T&Cs/stop SMS 08718727868. Over 18 only 150ppm

# Double mins and txts 4 6months FREE Bluetooth on Orange. Available on Sony, Nokia Motorola phones. Call MobileUpd8 on 08000839402 or call2optout/N9DX
# Test Pickle Model

print(pickle_model.predict(test_text_spam)) # Predict Category

print(pickle_model.predict_proba(test_text_spam)) # Predict Probability
# Import Keras Libraries

import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
le = LabelEncoder()            # Create Label Encoder

Y = le.fit_transform(df.Label) # Set Labels (ham, spam) -> (0, 1)

Y = Y.reshape(-1,1)            # Change (0, 1) -> (-1, 1)
# Split Data Set into Train and Test

# Train 70%

# Test  30%

# random_state setup means same values every time split is perfomed 

X_train, X_test, Y_train, Y_test = train_test_split(df.Text, Y, test_size=0.3, random_state=123)
# Tokenize the data and convert the text to sequences

# Add padding to ensure that all the sequences have the same shape

# There are many ways of taking the max_len and here an arbitrary length of 150 is chosen

max_words = 1000

max_len = 150



# Split Sentences into Tokens 

tok = Tokenizer(num_words = max_words) 

tok.fit_on_texts(X_train) 



# Transform sequence of senteces in sequence of integers

sequences = tok.texts_to_sequences(X_train)

# Create Input to Neural Network

sequences_matrix = sequence.pad_sequences(sequences, maxlen = max_len)
# Raw Textual Data

print(X_train[1])

# After Tokenization

print(sequences[1])

# Matrix with Tokens (Proper Input to Neural Network)

print(sequences_matrix[1])
# Define RNN - Recursive Neural Network

def RNN():

    inputs = Input(name = 'inputs', shape = [max_len])

    layer = Embedding(max_words, 50, input_length = max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256, name = 'FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1, name = 'out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs = inputs, outputs = layer)

    return model
# Create Model

model = RNN()

model.summary() # About RNN ...

model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(), metrics=['accuracy']) # Finalize Neural Network
# Train Neural Network

history = model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,

          validation_split=0.2, callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001)])



# Batch Size ?

# Epochs ?

# Validation Set ?
# Process the test set data

test_sequences = tok.texts_to_sequences(X_test)

# Prepare Test Input for Neural Network

test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen = max_len)



# Evaluate the model on the test set



accr = model.evaluate(test_sequences_matrix, Y_test)



# Check Neural Network Accuracy

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
# Setup precision of floating point number

np.set_printoptions(precision=2)



# Predictions with keras model

Y_pred = model.predict(test_sequences_matrix)

class_names = np.array(['ham', 'spam'])



# Based on Threshold (e.g. 0,9) change output from Neural Network

Y_pred = (Y_pred > 0.05).astype(int)



# Plot non-normalized confusion matrix

plot_confusion_matrix(Y_test, Y_pred, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()