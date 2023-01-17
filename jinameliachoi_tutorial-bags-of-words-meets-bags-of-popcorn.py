# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv', sep='\t', header=0, quoting=3)

data.head(3)
print(data['review'][0])
# HTML 형식에서 추출한 <br/> 태그가 여전히 존재. 공란으로 변경(삭제)

# 영어가 아닌 숫자/특수문자 역시 sentiment을 위한 피처로는 별 의미가 없기 떄문에 공란으로 변경

import re



# <br> html 태그는 공백으로 변환

data['review'] = data['review'].str.replace('<br />', ' ')



# 파이썬 정규 표현식 모듈 이용하여 영어 문자열이 아닌 문자는 모두 공백으로 변환

data['review'] = data['review'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
# 데이터 세트 분리

from sklearn.model_selection import train_test_split



class_data = data['sentiment']

feature_data = data.drop(['id', 'sentiment'], axis=1, inplace=False)



x_train, x_test, y_train, y_test = train_test_split(feature_data, class_data,

                                                    test_size=.3,

                                                    random_state=156)

x_train.shape, x_test.shape
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score



# count, tf-idf 차례로 적용



pipeline = Pipeline([

    ('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1,2))),

    ('lr', LogisticRegression(C=10))

])



pipeline.fit(x_train['review'], y_train)

pred = pipeline.predict(x_test['review'])

pred_probs = pipeline.predict_proba(x_test['review'])[:, 1]



print('예측 정확도는 {0:.4f}, ROC-AUC는 {1:.4f}'.format(accuracy_score(y_test, pred),

                                                    roc_auc_score(y_test, pred_probs)))
# tf-idf 적용



pipeline = Pipeline([

    ('tfidf-vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),

    ('lr', LogisticRegression(C=10))

])



pipeline.fit(x_train['review'], y_train)

pred = pipeline.predict(x_test['review'])

pred_probs = pipeline.predict_proba(x_test['review'])[:, 1]



print('예측 정확도는 {0:.4f}, ROC-AUC는 {1:.4f}'.format(accuracy_score(y_test, pred),

                                                    roc_auc_score(y_test, pred_probs)))
# SentiWordNet을 이용한 감성 분석



import nltk

nltk.download('all')
from nltk.corpus import wordnet as wn



# wordnet기반의 품사 tag로 변환

def penn_to_wn(tag):

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    elif tag.startswith('V'):

        return wn.VERB
# 문서를 문장 > 단어 토큰 > 품사 태깅 후에 sentisynset 클래스를 생성하고 polarity score를 합산하는 함수 생성

from nltk.stem import WordNetLemmatizer

from nltk.corpus import sentiwordnet as swn

from nltk import sent_tokenize, word_tokenize, pos_tag



def swn_polarity(text):

    # 감성 지수 초기화

    sentiment = 0.0

    tokens_count = 0

    

    lemmatizer = WordNetLemmatizer()

    raw_sentences = sent_tokenize(text)

    # 분해된 문장 별로 단어 토큰 > 품사 태깅 후에 sentisynset 생성 > 감성 지수 합산

    for raw_sentence in raw_sentences:

        # 품사 태깅 문장 추출

        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:

            

            # 품사 태깅과 어근 추출

            wn_tag = penn_to_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):

                continue

                

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)

            if not lemma:

                continue

            

            # 어근을 추출한 단어와 wordnet 기반 품사 태깅을 입력해 synset 객체를 생성

            synsets = wn.synsets(lemma, pos=wn_tag)

            if not synsets:

                continue

            # 감성 synset 추출

            # 모든 단어에 대해 긍정은 + 부정은 -

            synset = synsets[0]

            swn_synset = swn.senti_synset(synset.name())

            sentiment += (swn_synset.pos_score() - swn_synset.neg_score())

            tokens_count += 1

            

    if not tokens_count:

        return 0

    

    # 총 score가 0 이상인 경우 긍정 1, 그렇지 않은 경우 부정 0 반환

    if sentiment >= 0:

        return 1

    

    return 0
data['preds'] = data['review'].apply(lambda x: swn_polarity(x))

y_target = data['sentiment'].values

preds = data['preds'].values
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from sklearn.metrics import recall_score, f1_score, roc_auc_score

import numpy as np



print(confusion_matrix(y_target, preds))

print('정확도: ', np.round(accuracy_score(y_target, preds), 4))

print('정밀도: ', np.round(precision_score(y_target, preds), 4))

print('재현율: ', np.round(recall_score(y_target, preds), 4))
# VADER를 이용한 감정 분석

from nltk.sentiment.vader import SentimentIntensityAnalyzer



senti_analyzer = SentimentIntensityAnalyzer()

senti_scores = senti_analyzer.polarity_scores(data['review'][0])

print(senti_scores)
def vader_polarity(review, threshold=0.1):

    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(review)

    

    # compound값에 기반해 threshold 입력값보다 크면 1, 아니면 0을 바환

    agg_score = scores['compound']

    final_sentiment = 1 if agg_score >= threshold else 0

    return final_sentiment



# apply lambda식을 이용해 레코드별로 vader_polarity()를 수행하고 결과는 'vader_preds'에 저장

data['vader_preds'] = data['review'].apply(lambda x: vader_polarity(x, 0.1))

y_target = data['sentiment'].values

vader_preds = data['vader_preds'].values



print(confusion_matrix(y_target, vader_preds))

print('정확도: ', np.round(accuracy_score(y_target, vader_preds), 4))

print('정밀도: ', np.round(precision_score(y_target, vader_preds), 4))

print('재현율: ', np.round(recall_score(y_target, vader_preds), 4))