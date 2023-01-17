import numpy as np

import pandas as pd

import tensorflow as tf

import seaborn as sns

import transformers



import nltk

import re





from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



plt.style.use('seaborn')
print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))
#nltk.download('stopwords')
PATH_CSV_TRAIN = '../input/nlp-getting-started/train.csv'

PATH_CSV_TEST = '../input/nlp-getting-started/test.csv'

PATH_CSV_SUBMISSION = '../input/nlp-getting-started/submission.csv'



dataf = pd.read_csv(PATH_CSV_TRAIN)

dataf_test = pd.read_csv(PATH_CSV_TEST)
def clean_text(text):

    #Remove emojis and special chars

    clean=text

    #reg = re.compile('\\.+?(?=\B|$)')

    #clean = text.apply(lambda r: re.sub(reg, string=r, repl=''))

    #reg = re.compile('\x89Û_')

    #clean = clean.apply(lambda r: re.sub(reg, string=r, repl=' '))

    reg = re.compile('\&amp')

    clean = clean.apply(lambda r: re.sub(reg, string=r, repl='&'))

    reg = re.compile('\\n')

    clean = clean.apply(lambda r: re.sub(reg, string=r, repl=' '))



    #Remove hashtag symbol (#)

    #clean = clean.apply(lambda r: r.replace('#', ''))



    #Remove user names

    reg = re.compile('@[a-zA-Z0-9\_]+')

    clean = clean.apply(lambda r: re.sub(reg, string=r, repl='@'))



    #Remove URLs

    reg = re.compile('https?\S+(?=\s|$)')

    clean = clean.apply(lambda r: re.sub(reg, string=r, repl='www'))



    #Lowercase

    #clean = clean.apply(lambda r: r.lower())

    return clean
dataf['clean'] = clean_text(dataf['text'])

dataf_test['clean'] = clean_text(dataf_test['text'])
dataf.head(3)
from transformers import TFXLNetModel, XLNetTokenizer
# This is the identifier of the model. The library need this ID to download the weights and initialize the architecture

# here is all the supported ones:

# https://huggingface.co/transformers/pretrained_models.html

xlnet_model = 'xlnet-large-cased'

xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
def create_xlnet(mname):

    """ Creates the model. It is composed of the XLNet main block and then

    a classification head its added

    """

    # Define token ids as inputs

    word_inputs = tf.keras.Input(shape=(120,), name='word_inputs', dtype='int32')



    # Call XLNet model

    xlnet = TFXLNetModel.from_pretrained(mname)

    xlnet_encodings = xlnet(word_inputs)[0]



    # CLASSIFICATION HEAD 

    # Collect last step from last hidden state (CLS)

    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)

    # Apply dropout for regularization

    doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)

    # Final output 

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(doc_encoding)



    # Compile model

    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])



    return model
xlnet = create_xlnet(xlnet_model)
xlnet.summary()
tweets = dataf['clean']

labels = dataf['target']



X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.15, random_state=196)
def get_inputs(tweets, tokenizer, max_len=120):

    """ Gets tensors from text using the tokenizer provided"""

    inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in tweets]

    inp_tok = np.array([a['input_ids'] for a in inps])

    ids = np.array([a['attention_mask'] for a in inps])

    segments = np.array([a['token_type_ids'] for a in inps])

    return inp_tok, ids, segments



def warmup(epoch, lr):

    """Used for increasing the learning rate slowly, this tends to achieve better convergence.

    However, as we are finetuning for few epoch it's not crucial.

    """

    return max(lr +1e-6, 2e-5)



def plot_metrics(pred, true_labels):

    """Plots a ROC curve with the accuracy and the AUC"""

    acc = accuracy_score(true_labels, np.array(pred.flatten() >= .5, dtype='int'))

    fpr, tpr, thresholds = roc_curve(true_labels, pred)

    auc = roc_auc_score(true_labels, pred)



    fig, ax = plt.subplots(1, figsize=(8,8))

    ax.plot(fpr, tpr, color='red')

    ax.plot([0,1], [0,1], color='black', linestyle='--')

    ax.set_title(f"AUC: {auc}\nACC: {acc}");

    return fig
inp_tok, ids, segments = get_inputs(X_train, xlnet_tokenizer)
callbacks = [

    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02, restore_best_weights=True),

    tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0),

    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6)

]
hist = xlnet.fit(x=inp_tok, y=y_train, epochs=15, batch_size=16, validation_split=.15, callbacks=callbacks)
inp_tok, ids, segments = get_inputs(X_test, xlnet_tokenizer)
preds = xlnet.predict(inp_tok, verbose=True)
plot_metrics(preds, y_test);
pred_analysis_df = pd.DataFrame({'tweet':X_test.values, 'pred':preds.flatten(), 'real':y_test})

pred_analysis_df['rounded'] = np.array(pred_analysis_df['pred'] > 0.5, dtype='int')

diff = pred_analysis_df[pred_analysis_df['real'] != pred_analysis_df['rounded']]
#change to see other examples

idx = 44



tweet, real, pred = diff.iloc[idx, [0,2,3]]

print(tweet)

print("PRED: " + str(pred))

print("REAL: " + str(real))
tweets = dataf_test['clean']



inp_tok, ids, segments = get_inputs(tweets, xlnet_tokenizer)
preds = xlnet.predict(inp_tok, verbose=True)
dataf_test['target'] = preds

dataf_test['target'] = np.array(dataf_test['target'] >= 0.5, dtype='int')

dataf_test[['id', 'target']].to_csv('submission.csv', index=False)