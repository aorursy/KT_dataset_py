from keras.datasets import reuters
import numpy as np

# use reuters news dataset from keras datasets
(x_train, y_train), (x_test, y_test) = reuters.load_data()

word_index = reuters.get_word_index(path="reuters_word_index.json")
reverse_word_index = {v:k for k,v in word_index.items()}

# get train document vector
for i in range(x_train.__len__()):
    for j in range(x_train[i].__len__()):
        if(reverse_word_index.get(x_train[i][j])) == None:
            continue
        x_train[i][j] = reverse_word_index[x_train[i][j]]
    x_train[i] = ' '.join(map(str, x_train[i]))

# get test document vector
for i in range(x_test.__len__()):
    for j in range(x_test[i].__len__()):
        if(reverse_word_index.get(x_test[i][j])) == None:
              continue
        x_test[i][j] = reverse_word_index[x_test[i][j]]
    x_test[i] = ' '.join(map(str, x_test[i]))

# append test split into train split, I will use k fold    
x_train = np.append(x_train, [x_test])
y_train = np.append(y_train, [y_test])
import keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

# create feature vectors of word counts
encoded_docs = tokenizer.texts_to_matrix(x_train, mode='count')
x_train = encoded_docs

# make label vector of 46 categories
y_train_cat = keras.utils.to_categorical(y_train, num_classes=46, dtype='float32')

feature_size = encoded_docs[0].__len__();
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import KFold, StratifiedKFold

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib

# Instantiate the cross validator
skf = StratifiedKFold(n_splits=5, shuffle=True)

csv_scores = []
i = 1
for train, test in skf.split(x_train, y_train):
    print("Train on %d. validation split\n" % i)
    i += 1
    
    # Clear model, and create it
    model = None
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=feature_size))
    model.add(Dense(46, activation='softmax'))
    
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # train model
    history = model.fit(x_train[train], y_train_cat[train], epochs=15, batch_size=4096, validation_data=(x_train[test], y_train_cat[test]))

    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    
    accuracy_history = history.history['acc']
    val_accuracy_history = history.history['val_acc']
    
    csv_scores.append(val_accuracy_history[-1])

    # plot losses
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # plot metrics
    plt.plot(accuracy_history)
    plt.plot(val_accuracy_history)
    plt.title('Metrics')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    
print("Average accuracy across kfold splits: %.4f%% (+/- %.4f%%)" % (100*np.mean(csv_scores), 100*np.std(csv_scores)))