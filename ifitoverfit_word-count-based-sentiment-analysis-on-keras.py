with open("../input/amazon_cells_labelled.txt") as f:
    content1 = f.readlines()

with open("../input/imdb_labelled.txt") as f:
    content2 = f.readlines()
    
with open("../input/yelp_labelled.txt") as f:
    content3 = f.readlines()
    
# merge sentences from different sources
content = [] + content1 + content2 + content3

# get train inputs and labels   
docs = [x[:x.__len__()-2] for x in content]
docslabels = [int(x[x.__len__()-2]) for x in content]
from keras.preprocessing.text import Tokenizer

# tokenize each input(sentence)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

# create feature vectors of word counts
encoded_docs = tokenizer.texts_to_matrix(docs, mode='count')

input_size = encoded_docs.__len__()
feature_size = encoded_docs[0].__len__()
import numpy as np

# set train inputs and labels
x_train = encoded_docs
y_train = np.asarray(docslabels)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from keras import regularizers

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
    model.add(Dense(64, input_dim=feature_size, activation='softsign'))
    model.add(Dense(64, activation='softsign', kernel_regularizer=regularizers.l2(0.4)))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    
    # train model
    history = model.fit(x_train[train], y_train[train], epochs=20, batch_size=32, validation_data=(x_train[test], y_train[test]))

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
    
    
print("Average accuracy across kfold splits: %.2f%% (+/- %.2f%%)" % (100*np.mean(csv_scores), 100*np.std(csv_scores)))
keras.backend.clear_session()