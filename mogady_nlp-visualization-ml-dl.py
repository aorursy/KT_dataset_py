## install empath libirary needed for scattertext
!pip install empath
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np 
import seaborn as sns
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head(5)
test.head(5)
train.groupby('Sentiment').Phrase.count().plot(kind='bar')
len(train.SentenceId.unique())
import scattertext as st
import spacy
from pprint import pprint
from IPython.display import IFrame
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

data=train[(train['Sentiment']==0)|(train['Sentiment']==4)]
data['cat']=data['Sentiment'].astype("category").cat.rename_categories({0:'neg',4:'pos'})
corpus = st.CorpusFromPandas(data, 
                             category_col='cat',                               
                             text_col='Phrase',
                             nlp=st.whitespace_nlp_with_sentences).build()
term_freq_df = corpus.get_term_freq_df()
term_freq_df['positive Score'] = corpus.get_scaled_f_scores('pos')
pprint(list(term_freq_df.sort_values(by='positive Score', 
                                      ascending=False).index[:10]))
term_freq_df['negative Score'] = corpus.get_scaled_f_scores('neg')
pprint(list(term_freq_df.sort_values(by='negative Score', 
                                      ascending=False).index[:10]))
html = st.produce_scattertext_explorer(corpus,
         category='pos',category_name='positive',         
        not_category_name='neg',width_in_pixels=1000,
          metadata=data['cat'])
open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))
IFrame(src='Convention-Visualization.html', width = 1300, height=700)
feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(data,
                                              category_col='cat',
                                              feats_from_spacy_doc=feat_builder,
                                              parsed_col='Phrase').build()
html = st.produce_scattertext_explorer(empath_corpus,
                                        category='pos',
                                        category_name='Positive',
                                        not_category_name='Negative',
                                        width_in_pixels=1000,
                                        metadata=data['cat'],
                                        use_non_text_features=True,
                                        use_full_doc=True,
                                        topic_model_term_lists=feat_builder.get_top_model_term_lists())
open("Convention-Visualization-Empath.html", 'wb').write(html.encode('utf-8'))
IFrame(src='Convention-Visualization-Empath.html', width = 1300, height=700)
data=train[(train['Sentiment']!=1)&(train['Sentiment']!=3)]
data['cat']=data['Sentiment'].astype("category").cat.rename_categories({0:'neg',2:'neu',4:'pos'})
corpus = st.CorpusFromPandas(
    data,
    category_col='cat',
    text_col='Phrase',
    nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus()

semiotic_square = st.SemioticSquare(
    corpus,
    category_a='pos',
    category_b='neg',
    neutral_categories=['neu'],
    scorer=st.RankDifference(),
    labels={'not_a_and_not_b': 'Neutral', 'a_and_b': 'Reviews'})

html = st.produce_semiotic_square_explorer(semiotic_square,
                                           category_name='Positive',
                                           not_category_name='Negative',
                                           x_label='pos-neg',
                                           y_label='neu-Review',
                                           neutral_category_name='neutral',
                                           metadata=data['cat'])

open("lexicalized_semiotic_squares.html", 'wb').write(html.encode('utf-8'))
IFrame(src='lexicalized_semiotic_squares.html', width = 1600, height=900)
import re
def clean(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text
train['Phrase'] = train['Phrase'].apply(lambda x: clean(x))
test['Phrase']=test['Phrase'].apply(lambda x: clean(x))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
x_train, x_test, y_train, y_test = train_test_split(train['Phrase'], train['Sentiment'], train_size=0.8)
vectorizer = TfidfVectorizer().fit(x_train)
x_train_v = vectorizer.transform(x_train)
x_test_v  = vectorizer.transform(x_test)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from time import time
entries = []
def training():
    models = {
        "LogisticRegression": LogisticRegression(),
        "SGDClassifier": SGDClassifier(),
        "Multinomial":MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "GBClassifier":GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0)
    }
    for model in models:
        print("training model"+model)
        start = time()
        models[model].fit(x_train_v, y_train)
        end = time()
        print("trained in {} secs".format(end-start))
        y_pred = models[model].predict(x_test_v)
        entries.append((model,accuracy_score(y_test, y_pred)))
training()
cv_df = pd.DataFrame(entries, columns=['model_name','accuracy'])
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


corpus = st.CorpusFromScikit(
    X=CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(x_test[0:1000]),
    y=y_test[0:1000].values,
    feature_vocabulary=vectorizer.vocabulary_,
    category_names=['neg','som_neg','neu','som_pos','pos'],
    raw_texts=x_test[0:1000].values
).build()

clf=LinearSVC()
clf.fit(x_test_v,y_test)
html = st.produce_frequency_explorer(
    corpus,
    'neg',
    scores=clf.coef_[0],
    use_term_significance=False,
    terms_to_include=st.AutoTermSelector.get_selected_terms(corpus, clf.coef_[0])
)
file_name = "test_sklearn.html"
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1300, height=700)
corpus = st.CorpusFromScikit(
    X=CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(x_train[0:1000]),
    y=y_train[0:1000].values,
    feature_vocabulary=vectorizer.vocabulary_,
    category_names=['neg','som_neg','neu','som_pos','pos'],
    raw_texts=x_train[0:1000].values
).build()

clf=LinearSVC()
clf.fit(x_train_v,y_train)
html = st.produce_frequency_explorer(
    corpus,
    'neg',
    scores=clf.coef_[0],
    use_term_significance=False,
    terms_to_include=st.AutoTermSelector.get_selected_terms(corpus, clf.coef_[0])
)
file_name = "train_sklearn.html"
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1300, height=700)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
full_text=list(train['Phrase'].values) + list(test['Phrase'].values)
tokenizer.fit_on_texts(full_text)
train_seq = tokenizer.texts_to_sequences(train['Phrase'])
test_seq=tokenizer.texts_to_sequences(test['Phrase'])
voc_size=len(tokenizer.word_counts)
m=len(max(full_text, key=len))
X_train = pad_sequences(train_seq, maxlen = m)
X_test = pad_sequences(test_seq, maxlen = m)
y_train=train['Sentiment'].values
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (m,))
    x = Embedding(19479, 300)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100,activation='relu') (x))
    x = Dense(5, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    print(model.summary())
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 10, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model,history
model,history = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


pred = model.predict(X_test, batch_size = 1024)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
sub['Sentiment'] = predictions
sub.to_csv("blend.csv", index=False)
