# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import string

import gc



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from scipy.stats import spearmanr

from nltk.corpus import stopwords

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



eng_stopwords = set(stopwords.words("english"))
%%time

train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

sample_sub_df = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
pd.set_option('display.max_columns', None)

train_df.head()
test_df.head()
sample_sub_df.head()
print (f'Sahpe of training set: {train_df.shape}')

print (f'Sahpe of testing set: {test_df.shape}')
train_df.columns
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style="whitegrid")
total = len(train_df)
ax = sns.barplot(train_df['category'].value_counts().keys(), train_df['category'].value_counts())

ax.set(xlabel='Category', ylabel='# of records', title='Category vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 



plt.show()
v = np.vectorize(lambda x: x.split('.')[0])

sns.set(rc={'figure.figsize':(15,8)})

ax = sns.barplot(v(train_df['host'].value_counts().keys().values), train_df['host'].value_counts())

ax.set(xlabel='Host platforms', ylabel='# of records', title='Host platforms vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
wc = WordCloud(background_color='white', max_font_size = 85, width=700, height=350)

wc.generate(','.join(train_df['question_title'].tolist()))

plt.figure(figsize=(15,10))

plt.axis("off")

plt.imshow(wc, interpolation='bilinear')
wc.generate(','.join(train_df['question_body'].tolist()).replace('gt', '').replace('lt', ''))

plt.figure(figsize=(15,10))

plt.axis("off")

plt.imshow(wc, interpolation='bilinear')
wc.generate(','.join(train_df['answer'].tolist()).replace('gt', '').replace('lt', ''))

plt.figure(figsize=(15,10))

plt.axis("off")

plt.imshow(wc, interpolation='bilinear')
target_cols = sample_sub_df.drop(['qa_id'], axis=1).columns.values

target_cols
X_train = train_df.drop(np.concatenate([target_cols, np.array(['qa_id'])]), axis=1)

Y_train = train_df[target_cols]
print (f'Shape of X_train: {X_train.shape}')

print (f'Shape of Y_train: {Y_train.shape}')
X_train.head()
X_test = test_df

del test_df

gc.collect()
%%time

# Size of answers

X_train['answer_size'] = X_train['answer'].apply(lambda x: len(str(x).split()))

X_test['answer_size'] = X_test['answer'].apply(lambda x: len(str(x).split()))



# Size of question body

X_train['question_body_size'] = X_train['question_body'].apply(lambda x: len(str(x).split()))

X_test['question_body_size'] = X_test['question_body'].apply(lambda x: len(str(x).split()))



# Size of question title

X_train['question_title_size'] = X_train['question_title'].apply(lambda x: len(str(x).split()))

X_test['question_title_size'] = X_test['question_title'].apply(lambda x: len(str(x).split()))



# Number of unique words in the answer

X_train['answer_num_unique_words'] = X_train['answer'].apply(lambda x: len(set(str(x).split())))

X_test['answer_num_unique_words'] = X_test['answer'].apply(lambda x: len(set(str(x).split())))



# Number of unique words in the question body

X_train['question_body_num_unique_words'] = X_train['question_body'].apply(lambda x: len(set(str(x).split())))

X_test['question_body_num_unique_words'] = X_test['question_body'].apply(lambda x: len(set(str(x).split())))



# Number of characters in the answer

X_train['answer_num_chars'] = X_train['answer'].apply(lambda x: len(str(x)))

X_test['answer_num_chars'] = X_test['answer'].apply(lambda x: len(str(x)))



# Number of characters in the question body

X_train['question_body_num_chars'] = X_train['question_body'].apply(lambda x: len(str(x)))

X_test['question_body_num_chars'] = X_test['question_body'].apply(lambda x: len(str(x)))



# Number of stopwords in the answer

X_train['answer_num_stopwords'] = X_train['answer'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

X_test['answer_num_stopwords'] = X_test['answer'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



# Number of stopwords in the question body

X_train['question_body_num_stopwords'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

X_test['question_body_num_stopwords'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



# Number of punctuations in the answer

X_train['answer_num_punctuations'] = X_train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

X_test['answer_num_punctuations'] = X_test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



# Number of punctuations in the question body

X_train['question_body_num_punctuations'] = X_train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

X_test['question_body_num_punctuations'] = X_test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



# # Average length of the words in the answer

# X_train['answer_mean_word_len'] = X_train['answer'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# X_test['answer_mean_word_len'] = X_test['answer'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# # Average length of the words in the question body

# X_train['question_body_mean_word_len'] = X_train['question_body'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# X_test['question_body_mean_word_len'] = X_test['question_body'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# Number of upper case words in the answer

X_train['answer_num_words_upper'] = X_train['answer'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

X_test['answer_num_words_upper'] = X_test['answer'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



# Number of upper case words in the question body

X_train['question_body_num_words_upper'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

X_test['question_body_num_words_upper'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



# Number of title case words in the answer

X_train['answer_num_words_title'] = X_train['answer'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

X_test['answer_num_words_title'] = X_test['answer'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



# Number of title case words in the question body

X_train['question_body_num_words_title'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

X_test['question_body_num_words_title'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
X_train.head()
X_train = X_train.drop(['question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url'], axis=1)

X_test = X_test.drop(['question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url', 'qa_id'], axis=1)
tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

tsvd = TruncatedSVD(n_components = 1000)



question_title = tfv.fit_transform(X_train['question_title'].values).toarray()

question_title_test = tfv.transform(X_test['question_title'].values).toarray()

#question_title = tfv.fit_transform(X_train['question_title'].values)

#question_title_test = tfv.transform(X_test['question_title'].values)

#question_title = tsvd.fit_transform(question_title)

#question_title_test = tsvd.transform(question_title_test)



question_body = tfv.fit_transform(X_train['question_body'].values).toarray()

question_body_test = tfv.transform(X_test['question_body'].values).toarray()

#question_body = tfv.fit_transform(X_train['question_body'].values)

#question_body_test = tfv.transform(X_test['question_body'].values)

#question_body = tsvd.fit_transform(question_body)

#question_body_test = tsvd.transform(question_body_test)



answer = tfv.fit_transform(X_train['answer'].values).toarray()

answer_test = tfv.transform(X_test['answer'].values).toarray()

#answer = tfv.fit_transform(X_train['answer'].values)

#answer_test = tfv.transform(X_test['answer'].values)

#answer = tsvd.fit_transform(answer)

#answer_test = tsvd.transform(answer_test)
cat_le = LabelEncoder()

cat_le.fit(X_train['category'])

category = cat_le.transform(X_train['category'])

category_test = cat_le.transform(X_test['category'])
host_le = LabelEncoder()

host_le.fit(pd.concat([X_train['host'], X_test['host']], ignore_index=True))

host = host_le.transform(X_train['host'])

host_test = host_le.transform(X_test['host'])
meta_features_train = X_train.drop(['question_title', 'question_body', 'answer', 'category', 'host'], axis=1).to_numpy()

meta_features_test = X_test.drop(['question_title', 'question_body', 'answer', 'category', 'host'], axis=1).to_numpy()
X_train = np.concatenate([question_title, question_body, answer], axis=1)

X_test = np.concatenate([question_title_test, question_body_test, answer_test], axis=1)
del question_title

del question_title_test

del answer

del answer_test

del question_body

del question_body_test

gc.collect()
X_train = np.column_stack((X_train, category, host, meta_features_train))

X_test = np.column_stack((X_test, category_test, host_test, meta_features_test))
del category

del host

del meta_features_train

del category_test

del host_test

del meta_features_test

gc.collect()
print (X_train.shape)

print (X_test.shape)
np.isnan(X_train).any()
len(X_test)
folds = 5

seed = 666



kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

test_preds = np.zeros((len(X_test), len(target_cols)))

fold_scores = []



for train_index, val_index in kf.split(X_train):

    x_train, y_train = X_train[train_index, :], Y_train.iloc[train_index]

    x_val, y_val = X_train[val_index, :], Y_train.iloc[val_index]

    

    model = Sequential([

        Dense(256, input_shape=(X_train.shape[1],)),

        Dropout(0.25),

        Activation('relu'),

        Dense(128),

        Dropout(0.20),

        Activation ('relu'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    

    model.compile(optimizer='adam', loss='binary_crossentropy')

    

    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

    

    preds = model.predict(x_val)

    overall_score = 0

    

    for col_index, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, col_index], y_val[col].values).correlation/len(target_cols)

        

    fold_scores.append(overall_score)

#     models.append(model)

    test_preds += model.predict(X_test)/folds

    del x_train

    del y_train

    del x_val

    del y_val

    gc.collect()



print(fold_scores)
for col_index, col in enumerate(target_cols):

    sample_sub_df[col] = test_preds[:, col_index]
sample_sub_df.to_csv("submission.csv", index = False)