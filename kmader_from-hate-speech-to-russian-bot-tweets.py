# define network parameters
max_features = 20000
maxlen = 100
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import clear_output
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
list_sentences_train = train["comment_text"].fillna("Invalid").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
def build_model(conv_layers = 2, max_dilation_rate = 4):
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
                       dilation_rate = 2**dilation_rate)(x)    
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)    
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model

model = build_model()
batch_size = 512
epochs = 15

file_path="weights.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_split=0.1, 
          callbacks=callbacks_list)
model.load_weights(file_path)
clear_output()
eval_results = model.evaluate(X_t, y, batch_size=batch_size)
for c_name, c_val in zip(model.metrics_names, eval_results):
    print(c_name, '%2.3f' % (c_val))
from glob import glob
import os
import dask.dataframe as ddf
rustweet_dir = os.path.join('..', 'input', 'russian-troll-tweets')
all_tweets_ddf = ddf.read_csv(os.path.join(rustweet_dir, '*.csv'), assume_missing=True)
english_tweets_ddf = all_tweets_ddf[all_tweets_ddf['language'].isin(['English'])]
content_cat_ddf = english_tweets_ddf[['content', 'account_category']]
all_tweets_ddf
%%time
content_cat_df = content_cat_ddf.sample(frac=0.2).compute().drop_duplicates()
print(content_cat_df.shape[0], 'tweets loaded')
fig, ax1 = plt.subplots(1,1, figsize = (15, 5))
content_cat_df['account_category'].hist(ax=ax1)
content_cat_df.sample(3)
# test data
list_tweets = content_cat_df["content"].fillna("Invalid").values
list_tokenized_tweets = tokenizer.texts_to_sequences(list_tweets)
X_twe = sequence.pad_sequences(list_tokenized_tweets, maxlen=maxlen)
# run the model on all data
y_twe = model.predict(X_twe, batch_size=1024, verbose=True)
toxicity_df = pd.DataFrame(y_twe, columns = list_classes)
toxicity_df['content_category'] = content_cat_df['account_category'].values.copy()
toxicity_df['total_hatefulness'] = np.sum(y_twe, 1)
sns.pairplot(toxicity_df, hue = 'content_category')
from IPython.display import Markdown, display
display_markdown = lambda x: display(Markdown(x))
def show_sentence(sent_idx):
    display_markdown('# Input Sentence:\n `{}`'.format(list_tweets[sent_idx]))
    c_pred = model.predict(X_twe[sent_idx:sent_idx+1])[0]
    display_markdown('## Scores')
    for k, p in zip(list_classes, c_pred):
        display_markdown('- {}, Prediction: {:2.2f}%'.format(k, 100*p))
show_sentence(50)
worst_tweets = np.argsort(-1*toxicity_df['total_hatefulness'].values)
for _, idx in zip(range(5), 
                  worst_tweets):
    show_sentence(idx)
toxicity_df.groupby('content_category').agg(lambda x: round(100*np.mean(x))).reset_index().sort_values('total_hatefulness', ascending=False)
toxicity_df.groupby('content_category').agg(lambda x: round(100*np.max(x))).reset_index().sort_values('total_hatefulness', ascending=False)
cat_sample_df = toxicity_df.groupby('content_category').apply(lambda x: x.sample(250, replace=False if x.shape[0]>1000 else True)).reset_index(drop=True)
sns.factorplot(y='content_category', x='identity_hate', kind='swarm', data=cat_sample_df, size=5)
# rescale the axes a bit
clip_tox_df = toxicity_df.copy()
for c_class in list_classes:
    clip_tox_df[c_class] = np.sqrt(np.clip(clip_tox_df[c_class], 0, .025))
sns.factorplot(y='content_category', x='identity_hate', kind='violin', data=clip_tox_df)
sns.factorplot(y='content_category', x='threat', kind='violin', data=clip_tox_df, size=5)
from sklearn.model_selection import train_test_split
tx_train_df, tx_valid_df = train_test_split(toxicity_df, 
                                            test_size = 0.25,
                                            random_state = 2018,
                                            stratify=toxicity_df['content_category'])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
dmc = DummyClassifier()
def fit_and_show(in_skl_model):
    in_skl_model.fit(tx_train_df[list_classes], tx_train_df['content_category'])
    out_pred = in_skl_model.predict(tx_valid_df[list_classes])
    print('%2.2f%%' % (100*accuracy_score(out_pred, tx_valid_df['content_category'])), 'accuracy')
    print(classification_report(out_pred, tx_valid_df['content_category']))
    sns.heatmap(confusion_matrix(tx_valid_df['content_category'], out_pred))
fit_and_show(dmc)
from sklearn.linear_model import LogisticRegression
lrm = LogisticRegression()
fit_and_show(lrm)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
fit_and_show(rfc)
