import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

print(os.listdir("../input/sentiment-analysis-prepared-data"))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

import xgboost as xgb

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split



import nltk

from nltk.stem.porter import PorterStemmer
df = pd.read_csv('../input/sentiment-analysis-prepared-data/enriched_train_test_text.csv')

df.shape
labels = pd.read_csv('../input/data-sentiment-analysis/products_sentiment_train.tsv', 

                       sep = '\t', header = None, names = ['text', 'label'])['label']



labels.shape
df.head()
def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    return text



def tokenizer(text):

    return text.split()



porter = PorterStemmer()





def tokenizer_porter(text):

    return ' '.join([porter.stem(word) for word in text.split()])
df['text_porter'] = df['text'].apply(tokenizer_porter)
from nltk.corpus import stopwords

stop = stopwords.words('english')
# feature_nwords = [len(s.split()) for s in df['text']]

# feature_nsymb = [len(s) for s in df['text']]

# feature_avglenwords = np.divide(feature_nsymb, feature_nwords)
%%time

# countvectorizer words



cv_words = CountVectorizer(ngram_range=(1,1), tokenizer=tokenizer, stop_words='english', preprocessor=preprocessor)

X_words = cv_words.fit_transform(df['text'] + df['text_translated'] + df['text_w2v_twitter'] + df['text_w2v_news']).toarray()

X_words.shape
cv_phrases = CountVectorizer(ngram_range=(2,3), tokenizer=tokenizer, stop_words='english', preprocessor=preprocessor)

X_phrases = cv_phrases.fit_transform(df['text'] + df['text_translated']).toarray()

X_phrases.shape
%%time

# tfidf symbols

tfidf_symb = TfidfVectorizer(ngram_range = (3,5), use_idf = True, analyzer = 'char_wb', preprocessor = None)

X_symbols = tfidf_symb.fit_transform(df['text'] + df['text_translated']).toarray()

X_symbols.shape
X = np.column_stack((X_words, X_phrases, X_symbols))

X.shape
# %%time

# X_tsne = TSNE(n_components = 3).fit_transform(X)

# X_tsne.shape
# %%time

# X_pca = PCA(n_components = 2000).fit_transform(X)

# X_pca.shape
# X_pca_nwords = np.column_stack((np.array(X_pca), np.array(feature_nwords), np.array(feature_nsymb)))
# logistic regression

param_grid_lr = [{

                  'C': [0.1, 1.0, 10.0],

                 'class_weight': ['', 'balanced'],

                 'l1_ratio': [0.00, 0.20]}]



# SVC

param_grid_svc = [{'penalty': ['l2'],

                  'C': [0.1, 1.0, 10.0, 100.0],

                 'class_weight': ['', 'balanced']}]



# SGD

param_grid_sgd = [{'penalty': ['l1', 'l2'],

                  'alpha': [0.1, 1.0, 10.0, 100.0],

                 'class_weight': ['', 'balanced']}]



# random forest

param_grid_rf = [{'n_estimators': [400],

                  'class_weight': ['balanced'],

                  'max_depth': [50]}]



# gradient boosting

param_grid_gb = [{'learning_rate': [0.05, 0.1],

                  'max_depth': [3, 6],

                 'min_child_weight': [3, 9],

                 'subsample': [0.2, 0.5],

                 'n_estimators': [100]}]
gs_lr = GridSearchCV(LogisticRegression(random_state = 0), 

                     param_grid_lr, scoring = 'accuracy', cv = 5, n_jobs = -1)



gs_svc = GridSearchCV(LinearSVC(random_state = 0), 

                     param_grid_svc, scoring = 'accuracy', cv = 5, n_jobs = -1)



gs_sgd = GridSearchCV(SGDClassifier(random_state = 0), 

                     param_grid_sgd, scoring = 'accuracy', cv = 5, n_jobs = -1)



gs_rf = GridSearchCV(RandomForestClassifier(random_state = 0), 

                     param_grid_rf, scoring = 'accuracy', cv = 5, n_jobs = -1)



gs_gb = GridSearchCV(xgb.XGBClassifier(random_state = 0), 

                     param_grid_gb, scoring = 'accuracy', cv = 5, n_jobs = -1)
# %%time

# gs_lr.fit(X[:2000,:], labels)

# print(gs_lr.best_params_)

# print(gs_lr.best_score_)
# %%time

# gs_lr.fit(X_base[:2000,:], labels)

# print(gs_lr.best_params_)

# print(gs_lr.best_score_)
# %%time

# gs_svc.fit(X[:2000,:], labels)

# print(gs_svc.best_params_)

# print(gs_svc.best_score_)
# %%time

# gs_sgd.fit(X[:2000,:], labels)

# print(gs_sgd.best_params_)

# print(gs_sgd.best_score_)
# %%time

# gs_rf.fit(X[:2000,:], labels)

# print(gs_rf.best_params_)

# print(gs_rf.best_score_)
# %%time



# gs_gb.fit(X_pca_nwords[:2000,:], labels)

# print(gs_gb.best_params_)

# print(gs_gb.best_score_)
from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
df_train_stacked = pd.DataFrame({'text' : []})

df_test_stacked = pd.DataFrame({'text' : []})

df_stacked = pd.DataFrame({'text' : []})



df_train_stacked['text']=pd.concat([df['text'][:2000].apply(preprocessor), df['text_translated'][:2000].apply(preprocessor),

                        df['text_w2v_twitter'][:2000].apply(preprocessor), df['text_w2v_news'][:2000].apply(preprocessor)],

                                  ignore_index = True)



df_test_stacked['text']=pd.concat([df['text'][2000:].apply(preprocessor), df['text_translated'][2000:].apply(preprocessor),

                        df['text_w2v_twitter'][2000:].apply(preprocessor), df['text_w2v_news'][2000:].apply(preprocessor)],

                                 ignore_index = True)



df_stacked['text'] = pd.concat([df_train_stacked['text'], df_test_stacked['text']], ignore_index = True)



df_stacked['text'] = df_stacked['text'].apply(tokenizer_porter)



df_stacked.shape
%%time



tokenizer = Tokenizer(num_words = None, lower = True, split = ' ')

tokenizer.fit_on_texts(df_stacked['text'])



X_sequences = tokenizer.texts_to_sequences(df_stacked['text'])
print('Length of the original text = ', len(df_stacked.iloc[1100].text.split()))

print('The full example:', df_stacked.iloc[1100].text)



print('Length of the digital text = ', len(X_sequences[1100]))

print('The full example:', X_sequences[1100])
idx_word = tokenizer.index_word



' '.join(idx_word[w] for w in X_sequences[1100])
top_words = 50000            # ограниичимся словарём из top_words самых частых слов

max_review_length = 500      # обрежем все отзывы до max_review_length слов

embedding_vector_length = 32 # размерность эмбендинга



X_sequences = sequence.pad_sequences(X_sequences, maxlen=max_review_length)

y_sequences = pd.concat([labels, labels, labels, labels])



print('X:', X_sequences.shape)

print('y:', y_sequences.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X_sequences[:8000], y_sequences, test_size=0.2, random_state=42,

                                                     stratify = y_sequences)



print('X_train:', X_train.shape)

print('X_valid:', X_valid.shape)
%%time



model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))
%%time



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32)

scores = model.evaluate(X_valid, y_valid, verbose=0)

print('Accuracy: %.4f'%(scores[1]))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(['Train loss', 'Validation loss'])
lr = LogisticRegression(random_state = 0, C = 0.1, l1_ratio = 0, class_weight = 'balanced')

lr_model = lr.fit(X[:2000,:], labels)
# lr_base = LogisticRegression(random_state = 0, C = 1, l1_ratio = 0, class_weight = 'balanced')

# lr_model_base = lr_base.fit(X_base[:2000,:], labels)
%%time

rf = RandomForestClassifier(class_weight = 'balanced', max_depth = 50, n_estimators = 400)

rf_model = rf.fit(X[:2000,:], labels)
# %%time

# gb = xgb.XGBClassifier(learning_rate = 0.05, max_depth = 3, subsample = 0.5, 

#                        n_estimators = 100, min_child_weight = 3)

# gb_model = gb.fit(X_pca_nwords[:2000,:], labels)
lr_res = lr_model.predict(X[2000:,:])

lr_prob = lr_model.predict_proba(X[2000:,:])



# lr_base_res = lr_model_base.predict(X_base[2000:,:])

# lr_base_prob = lr_model_base.predict_proba(X_base[2000:,:])



rf_res = rf_model.predict(X[2000:,:])

rf_prob = rf_model.predict_proba(X[2000:,:])



# gb_res = gb_model.predict(X_pca_nwords[2000:,:])

# gb_prob = gb_model.predict_proba(X_pca_nwords[2000:,:])
lstm_prob_1 = model.predict_proba(X_sequences[8000:10000])



lstm_res = []

lstm_prob = []

threshold = 0.5



for i, s in enumerate(lstm_prob_1):

    if s[0] >= threshold:

        lstm_res.append(1)

        lstm_prob.append(s[0])

    else:

        lstm_res.append(0)

        lstm_prob.append(1-s[0])

    #print(s[0], ' ', lstm_res[i])
df_analyze = pd.DataFrame({'text': df['text'].iloc[2000:].values, 

                           'lr_result': lr_res, 'lr_prob': [max(i) for i in lr_prob],

                           'rf_result': rf_res, 'rf_prob': [max(i) for i in rf_prob],

                            'lstm_result': lstm_res[1500:2000], 'lstm_prob': [i for i in lstm_prob[1500:2000]] })

                        #   'gb_result': gb_res, 'gb_prob': [max(i) for i in gb_prob]})
pd.options.display.max_colwidth = 200

df_analyze[(df_analyze['lr_result'] + df_analyze['rf_result'] + df_analyze['lstm_result'])%3 != 0]
best_answers = []

probability = ['lr_prob', 'rf_prob', 'lstm_prob']

results = ['lr_result', 'rf_result', 'lstm_result']

indices = df_analyze[probability].idxmax(axis = 1)



for n, row in enumerate(df_analyze[results].iterrows()):

    col_ind = df_analyze[probability].columns.get_loc(indices[n])

    best_answers.append(row[1][col_ind])

    

best_answers = np.int_(df_analyze[results].sum(axis = 1)/3)
result = pd.DataFrame({ 'Id':[i for i in range(len(X[2000:,:]))], 'y': best_answers })

result_lstm = pd.DataFrame({ 'Id':[i for i in range(len(X[2000:,:]))], 'y': lstm_res[1500:2000] })
sns.countplot(result_lstm.y);

plt.title('Train: Target distribution');
result.to_csv('kaggle_submission_07_12_2019_all3.csv', index = False)

result_lstm.to_csv('kaggle_submission_07_12_2019_lstm.csv', index = False)