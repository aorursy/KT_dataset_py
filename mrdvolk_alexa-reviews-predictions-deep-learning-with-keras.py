import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk # wonderful tutorial can be found here https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/

%matplotlib inline
df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')
df.head(10)
df.describe()
#omitting unnecessary columns
cdf = df[['rating', 'verified_reviews']]

print(cdf['rating'].value_counts())
cdf = pd.concat([cdf[cdf['rating'] < 5], cdf[cdf['rating'] == 5].sample(frac=1).iloc[:300]])
cdf['rating'].describe()
cdf['rating'].hist(bins=5)
text_body = ''
for row in cdf.iterrows():
    text_body += row[1]['verified_reviews'] + ' '
    
cleaned_text_body = re.sub('[^a-zA-Z]', ' ', text_body)
word_list = nltk.tokenize.word_tokenize(cleaned_text_body.lower())
word_set = set(word_list)
len(word_set)
embeddings = {}
f = open('../input/glove6b100dtxt/glove.6B.100d.txt', 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings[word] = vector
f.close()
def assess_embeddings(set_of_words):
    c = 0
    missing_embeddings = []
    for word in set_of_words:
        if word in embeddings:
            c+=1
        else:
            missing_embeddings.append(word)

    print(c/len(set_of_words)*100, 'percents of words in reviews are covered in embeddings')
    return missing_embeddings

missing_embeddings = assess_embeddings(word_set)    
print(sorted(missing_embeddings))
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts([cleaned_text_body])
print('Words in word_index:', len(tokenizer.word_index))
_ = assess_embeddings(set([kvp for kvp in tokenizer.word_index]))
cdf['cleaned_text'] = cdf['verified_reviews'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
cdf['cleaned_text'] = cdf['cleaned_text'].apply(lambda x: re.sub(' +',' ', x)) #remove consecutive spacing
cdf['sequences'] = cdf['cleaned_text'].apply(lambda x: tokenizer.texts_to_sequences([x])[0])
cdf['sequences'].head(10)
# Need to know max_sequence_length to pad other sequences
max_sequence_length = cdf['sequences'].apply(lambda x: len(x)).max()
cdf['padded_sequences'] = cdf['sequences'].apply(lambda x: pad_sequences([x], max_sequence_length)[0])
print(cdf['padded_sequences'][2])
train = cdf.sample(frac=0.8)
test_and_validation = cdf.loc[~cdf.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)
def get_arrayed_data(df_set):
    setX = np.stack(df_set['padded_sequences'].values, axis=0)
    setY = pd.get_dummies(df_set['rating']).values #using one-hot encoding
    
    return (setX, setY)

trainX, trainY = get_arrayed_data(train)
validationX, validationY = get_arrayed_data(validation)
testX, testY = get_arrayed_data(test)
from keras.layers import Embedding
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    # words that are not in pretrained embedding will be zero vectors.
    if word in embeddings:
        embedding_matrix[i] = embeddings[word]
embedding_layer = Embedding(len(tokenizer.word_index) + 1, 100,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Flatten, Dropout
def simple_reccurent_model(input_shape, output_shape):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.2))
    #model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    return model
model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(trainX, trainY, batch_size=64, epochs=100)
score, accuracy = model.evaluate(validationX, validationY, batch_size=64)
print(accuracy)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
for i, y in enumerate(validationY):
    #p = model.predict(np.array([validationX[i]])).round()[0].astype('int32')
    prediction = model.predict(np.array([validationX[i]])).argmax()
    actual = y.argmax()
    if prediction != actual:
        print("Validation", i)
        print("Predicted review:", prediction + 1, ", actual review:", actual + 1)
        #print(validationX[i])
        text = []
        for word_i in validationX[i]:
            if word_i in reverse_word_map:
                text.append(reverse_word_map[word_i])
        print(' '.join(text))
        print()
def tunable_reccurent_model(input_shape, output_shape, hyperparams):
    model = Sequential()
    model.add(embedding_layer)
    
    for i, lstm_size in enumerate(hyperparams['lstm_sizes']):
        model.add(LSTM(lstm_size, dropout=hyperparams['dp']))
    
    for i, dense_size in enumerate(hyperparams['dense_sizes']):
        model.add(Dense(dense_size, activation=hyperparams['dense_activation']))
        model.add(Dropout(hyperparams['dp']))
    
    model.add(Dense(output_shape, activation='softmax'))
    return model
def evaluate_model(input_shape, output_shape,
                   hyperparams, train_set, validation_set,
                   train_epochs=100):
    model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=hyperparams['optimizer'],
                  metrics=['accuracy'])
    
    model.fit(train_set[0], train_set[1], batch_size=hyperparams['batch_size'], epochs=train_epochs, verbose=0)
    _, train_accuracy = model.evaluate(train_set[0], train_set[1])
    _, validation_accuracy = model.evaluate(validation_set[0], validation_set[1])
    print("Train accuaracy:", train_accuracy, "Validation Accuracy:", validation_accuracy)
    return validation_accuracy
lstm_sizes = [[32], [64], [128], [64, 32]]
dense_sizes = [[32], [64], [128], [64, 32]]
dense_activations = ['relu', 'tanh', 'sigmoid']
dps = [0.1, 0.2, 0.3]
optimizers = ['Adam', 'SGD', 'RMSprop']
epochs = [100, 125, 150]
batch_sizes = [32, 64, 128]

results = []
counter=1
# all hyperparameters here are enumerated not in random order - the least important are closer to outer cycle
for ep in epochs:
    for optimizer in optimizers:
        for dense_activation in dense_activations:
            for batch_size in batch_sizes:
                for dp in dps:
                    for dense_size in dense_sizes:
                        for lstm_size in lstm_sizes:
                            hyperparams = {
                                'lstm_sizes': lstm_size,
                                'dp': dp,
                                'dense_sizes': dense_size,
                                'dense_activation': dense_activation,
                                'optimizer': optimizer,
                                'batch_size': batch_size
                            }
                            #print("Interation", counter)
                            #acc = evaluate_model(trainX.shape[1], trainY.shape[1],
                            #                    hyperparams, (trainX, trainY), (validationX, validationY),
                            #                    ep)
                            #results.append((acc, hyperparams, {'ep': ep, 'batch_size': batch_size}))
                            #counter+=1
                            #print()
                            
import copy
lstm_sizes = [[32], [64], [64, 32]]
dense_sizes = [[32], [64], [64, 32]]
dense_activations = ['relu', 'tanh', 'sigmoid']
dps = [0.1, 0.2, 0.3]
optimizers = ['Adam', 'SGD', 'RMSprop']
epochs = 10
batch_sizes = [32, 64, 128]

hyperparams = {
    'lstm_size': lstm_sizes,
    'dense_size': dense_sizes,
    'dense_activation': dense_activations,
    'dp': dps,
    'optimizer': optimizers,
    'batch_size': batch_sizes
}

default_hyperparams = {
    'lstm_size': [64],
    'dp': 0.2,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 64
}

counter = 1
validation_results = []
for hp_name, hp_list in hyperparams.items():
    accs = []
    for hp_val in hp_list:
        hp = copy.deepcopy(default_hyperparams)
        hp[hp_name] = hp_val
        print("Interation", counter)
        acc = evaluate_model(trainX.shape[1], trainY.shape[1],
                            hp, (trainX, trainY), (validationX, validationY), epochs)
        counter+=1
        accs.append(acc)
        print()
    validation_results.append((hp_name, accs))
fig = plt.figure(figsize=(6, 18))

for i, result in enumerate(validation_results):
    ax = fig.add_subplot(len(validation_results), 1, i+1)
    hp_name = result[0]
    hp_errors = result[1]
    
    ax.set_title(hp_name)
    ax.plot(range(0, len(hp_errors)), hp_errors)
    plt.sca(ax)
    x_labels = hyperparams[hp_name]
    plt.xticks(range(0, len(hp_errors)), x_labels)
    
fig.tight_layout()
plt.show()
# edited function to return model
def evaluate_model(input_shape, output_shape,
                   hyperparams, train_set, validation_set,
                   train_epochs=100, verbose=0):
    model = simple_reccurent_model(trainX.shape[1], trainY.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=hyperparams['optimizer'],
                  metrics=['accuracy'])
    
    model.fit(train_set[0], train_set[1], batch_size=hyperparams['batch_size'], epochs=train_epochs, verbose=verbose)
    _, train_accuracy = model.evaluate(train_set[0], train_set[1])
    _, validation_accuracy = model.evaluate(validation_set[0], validation_set[1])
    print("Train accuaracy:", train_accuracy, "Validation Accuracy:", validation_accuracy)
    return validation_accuracy, model
tuned_hyperparams = {
    'lstm_size': [32],
    'dp': 0.2,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 32
}

acc, model = evaluate_model(trainX.shape[1], trainY.shape[1],
    tuned_hyperparams, (trainX, trainY), (validationX, validationY), 100, 0)
tuned_hyperparams = {
    'lstm_size': [32],
    'dp': 0.5,
    'dense_size': [64],
    'dense_activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 32
}

acc, model2 = evaluate_model(trainX.shape[1], trainY.shape[1],
    tuned_hyperparams, (trainX, trainY), (validationX, validationY), 50, 0)
error_count = 0
for i, y in enumerate(validationY):
    #p = model.predict(np.array([validationX[i]])).round()[0].astype('int32')
    prediction = model2.predict(np.array([validationX[i]])).argmax() > 3
    actual = y.argmax() > 3
    if prediction != actual:
        print("Validation", i)
        print("Predicted review is good:", prediction, ", actual review is good:", actual)
        #print(validationX[i])
        text = []
        for word_i in validationX[i]:
            if word_i in reverse_word_map:
                text.append(reverse_word_map[word_i])
        print(' '.join(text))
        print()
        error_count+=1
print("Accuracy of prediction whether review was good:",(validationY.shape[0] - error_count)/validationY.shape[0] * 100)
predicted = [x > 3 for x in model2.predict(validationX).argmax(axis=1)]
actual = [x > 3 for x in validationY.argmax(axis=1)]
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(actual, predicted)
sns.heatmap(cnf_matrix)
tn, fp, fn, tp = cnf_matrix.ravel()
print("True positive:", tp, ", True negative:", tn,
      ", False positive:", fp, ", False negative:", fn)
_, test_accuracy = model2.evaluate(testX, testY)
print(test_accuracy)
def check_general_accuracy(model, setX, setY):
    predicted = [x > 3 for x in model.predict(setX).argmax(axis=1)]
    actual = [x > 3 for x in setY.argmax(axis=1)]

    cnf_matrix = confusion_matrix(actual, predicted)
    sns.heatmap(cnf_matrix)
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("True positive:", tp, ", True negative:", tn,
          ", False positive:", fp, ", False negative:", fn)

    print("Total accuracy:", (tp+tn)/testX.shape[0]*100)
check_general_accuracy(model2, testX, testY)
from keras.layers import Conv1D, MaxPooling1D
def simple_conv_model(input_shape, output_shape):
    model = Sequential()
    model.add(embedding_layer)
    
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(output_shape, activation='softmax'))
    return model
model = simple_conv_model(trainX.shape[1], trainY.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
model.summary()
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("weights-improvements.hdf5",
     monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(trainX, trainY, validation_data=(validationX, validationY),
          batch_size=64, callbacks=[checkpoint], epochs=100)
model.load_weights("weights-improvements.hdf5")
_, test_accuracy = model.evaluate(testX, testY)
print(test_accuracy)
check_general_accuracy(model, testX, testY)
