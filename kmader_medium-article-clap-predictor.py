!ls -R ../input | head
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
# define network parameters
max_features = 20000
maxlen = 100
text_field = 'text'
train = pd.read_csv("../input/medium-articles/articles.csv")
train['len_text'] = train['text'].str.len()
train['claps'] = train['claps'].apply(lambda s: int(float(s[:-1]) * 1000) if s[-1] == 'K' else int(s))
print(train.shape[0], 'total sentences')
outlier_cutoff = np.percentile(train['claps'], 90)
c_vec = np.clip(train['claps'], 0, outlier_cutoff)
median_score = np.median(c_vec)
std_score = train['claps'].std()
print(outlier_cutoff, median_score, std_score)
train['claps'] = c_vec
train['clap_zscore'] = ((c_vec-median_score)/std_score)

train, test = train_test_split(train, 
                               random_state = 2018, 
                               test_size = 0.25, 
                               stratify = pd.qcut(c_vec.values, 10, duplicates = 'drop'))
print('train sentences', train.shape[0], 'test sentences', test.shape[0])
train.sample(5)
list_sentences_train = train[text_field].fillna("Invalid").values
list_sentences_test = test[text_field].fillna("Invalid").values
train_y = train['clap_zscore']
test_y = test['clap_zscore']
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
ax1.hist(train_y, 10, label = 'Train Scores')
ax1.hist(test_y, 10, label = 'Test Scores', alpha = 0.5)
ax1.legend()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
def build_model(conv_layers = 2, max_dilation_rate = 3):
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    prefilt_x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process 
    for dilation_rate in range(max_dilation_rate):
        x = prefilt_x
        for i in range(3):
            x = Conv1D(32*2**(i), 
                       kernel_size = 3, 
                       dilation_rate = dilation_rate+1)(x)    
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)    
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="tanh")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model

model = build_model()
model.summary()
batch_size = 32
epochs = 10

file_path="weights.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, train_y, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_split=0.2, 
          callbacks=callbacks_list)

model.load_weights(file_path)
model.save('full_model.h5')
model.evaluate(X_te, test_y)
test['pred_claps'] = model.predict(X_te)*std_score+median_score
fig, ax1 = plt.subplots(1, 1, figsize = (5, 5))
ax1.plot(test['claps'], test['pred_claps'], 'ro', label = 'Prediction')
ax1.plot(test['claps'], test['claps'], 'b-', label = 'Actual')
ax1.legend()
test['pred_error'] = test['claps']-test['pred_claps']
test.sort_values('pred_error')[['title', 'claps', 'pred_claps']].head(5)
# surprises
test.sort_values('pred_error')[['title', 'claps', 'pred_claps']].tail(5)
