import pandas as pd
import pickle
#load pkl files
train_df_5_sample = pd.read_pickle("../input/train_df_5.pkl")
test_df_5_sample = pd.read_pickle("../input/test_df_5_random_sample.pkl")
test_df_5 = pd.read_pickle("../input/test_df_5.pkl")
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('training.log', separator=',', append=False)
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 3), tokenizer=nltk.word_tokenize) 
vectorizer.fit(train_df_5_sample['text'])
import keras
from sklearn.preprocessing import LabelEncoder

Y_train = train_df_5_sample['emotion']
Y_test = test_df_5_sample['emotion']

label_encoder = LabelEncoder()
label_encoder.fit(Y_train)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

Y_train = label_encode(label_encoder, Y_train)
Y_test = label_encode(label_encoder, Y_test)
len(test_df_5_sample)
label_encoder.classes_
len(test_df_5)
X_train = vectorizer.transform(train_df_5_sample['text'])
X_test = vectorizer.transform(test_df_5_sample['text'])
X_predict = vectorizer.transform(test_df_5['text'])
#Get shapes
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
import time

def train(model,epochs,batch_size):
    '''train a model'''
    history=model.fit(X_train, Y_train, validation_split=0.33, epochs = epochs, batch_size=batch_size, verbose=1, callbacks=[csv_logger])
    print(history)
    return history
def validate(model, batch_size):
    '''validate a model'''
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
def predict(model):
    '''predict with a model'''
    positive_count = 0
    negative_count = 0
    positive_correct = 0
    negative_correct = 0

    for x in range(len(X_validate)):

        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                negative_correct += 1
            else:
                positive_correct += 1

        if np.argmax(Y_validate[x]) == 0:
            negative_count += 1
        else:
            positive_count += 1

    print("Positive Accuracy", positive_correct/positive_count*100, "%")
    print("Negative Accuracy", negative_correct/negative_count*100, "%")
    print(positive_correct)
    print(positive_count)
    print(negative_correct)
    print(negative_count)
import matplotlib.pyplot as plt

def plotTrainingHistory(history):
    '''plots training history'''
    plt.figure(0)
    plt.plot(history.history['acc'], 'r')
    plt.plot(history.history['val_acc'],'g')
    epochs = len(history.history['loss'])
    plt.xticks(np.arange(0, epochs+1, epochs/5))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy per epoch")
    plt.legend(['train','validation'])
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Conv1D, GlobalMaxPooling1D, Embedding, CuDNNLSTM, LSTM, Flatten, SpatialDropout1D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


def createModel3(output_dim, embedding_dim, lstm_out_dim, input_length, max_features):
    '''creates a model for input_dim, embedding_dim, lstm_out_dim, batch_size, epochs, 
    output_dim, model_path params'''
    
    #embedding_matrix = load_embedding_matrix(model_path, tokenizer, embedding_dim)
    #embedding_layer = Embedding(max_features, embedding_dim,input_length=X.shape[1])
    model = Sequential()
    #model.add(Embedding(max_features, embedding_dim,input_length=X_train.shape[1]))
    #model.add(SpatialDropout1D(0.2))
    #model.add(CuDNNLSTM(lstm_out_dim))
    #model.add(Flatten())
    model.add(Conv1D(filters=filters, kernel_size=2, padding='valid',activation='relu',strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_dim, activation='relu'))
    model.add(Dense(output_dim=output_dim,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Embedding, CuDNNLSTM, LSTM, Flatten, SpatialDropout1D,MaxPooling2D,Input,ReLU, Softmax
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model

def createModelTA(output_dim, dense_dim):
    '''creates a model for input_dim, embedding_dim, lstm_out_dim, batch_size, epochs, 
    output_dim, model_path params'''
    


    # input layer
    model_input = Input(shape=(X_train.shape[1], ))  # 500
    X = model_input

    # 1st hidden layer
    X_W1 = Dense(units=dense_dim)(X)  # 64
    H1 = ReLU()(X_W1)

    # 2nd hidden layer
    H1_W2 = Dense(units=dense_dim)(H1)  # 64
    H2 = ReLU()(H1_W2)

    # output layer
    H2_W3 = Dense(units=output_dim)(H2)  # 4
    H3 = Softmax()(H2_W3)

    model_output = H3

    # create model
    model = Model(inputs=[model_input], outputs=[model_output])

    # loss function & optimizer
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # show model construction
    model.summary()
    
    return model


#friday evening
output_dim = len(label_encoder.classes_)
dense_dim = 64
embedding_dim = 64
lstm_out_dim = 64
filters = 100
batch_size = 32
epochs = 10
input_length = max(train_df_5_sample.text_tok_keras.apply(lambda x: len(x)))
max_features = 10000

keras_model = None
keras_model = createModel3(output_dim, embedding_dim, lstm_out_dim, input_length,max_features)
#keras_model = createModelTA(output_dim,dense_dim)
history=train(keras_model,epochs=epochs,batch_size=1)
validate(keras_model,batch_size)
keras_model.save("model_3k.h5")
keras_model.save_weights("model_3k_weights.h5")
def predictToCSV(model, filename, X_test):
    '''dump prediction results to csv'''
    pathname = filename +'.csv'
    pred_result = model.predict(X_test,batch_size=1,verbose = 2)
    pred_result = label_decode(label_encoder, pred_result)
    df_results = pd.DataFrame()
    df_results['id'] = test_df_5['tweet_id']
    df_results['emotion'] = pred_result
    df_results.to_csv(pathname, index=False)
    return df_results
df_results=predictToCSV(keras_model,"predictions_kaggle_conv",X_predict)
#from keras.models import load_model
#keras_model_TA = load_model("../input/model_TA2k.h5")
#predictToCSV(keras_model_TA,"predictions_tfidf_fc",X_predict)
import pandas as pd
log_data = pd.read_csv('training.log', sep=',',engine='python')
log_data.to_csv('training_log_conv.csv', index=False)
plotTrainingHistory(history)