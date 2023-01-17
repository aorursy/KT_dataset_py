!pip install nagisa
import itertools

import nagisa

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('../input/aiboinfo/aibo.csv')
add_noun = [

    '水谷豊',

    '寺脇康文',

    '及川光博',

    '成宮寛貴',

    '反町隆史',

    '川原和久',

    '六角精児',

    '鈴木砂羽',

    '岸部一徳',

    '原田龍二',

    '真飛聖',

    '鈴木杏樹',

    '山西惇',

    '神保悟志'

]



tagger = nagisa.Tagger(single_word_list=add_noun)

df['sep_descs'] = [tagger.extract(text, extract_postags=['名詞']).words for text in df['descs']]
df.head()
kameyama_df = df.loc[:120]

kanbe_df = df.loc[130:186]

kai_df = df.loc[187:243]

kaburagi_df = df.loc[244:]
kameyama_words = list(itertools.chain(*list(kameyama_df['sep_descs'])))

kanbe_words = list(itertools.chain(*list(kanbe_df['sep_descs'])))

kai_words = list(itertools.chain(*list(kai_df['sep_descs'])))

kaburagi_words = list(itertools.chain(*list(kaburagi_df['sep_descs'])))



data = [

    ' '.join(kameyama_words),

    ' '.join(kanbe_words),

    ' '.join(kai_words),

    ' '.join(kaburagi_words),

]



vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data).toarray()



tfidf_df = pd.DataFrame(X.T, index=vectorizer.get_feature_names(), columns=['亀山', '神戸', '甲斐', '冠城'])
tfidf_df.sort_values('亀山', ascending=False).head(10)
tfidf_df.sort_values('神戸', ascending=False).head(10)
tfidf_df.sort_values('甲斐', ascending=False).head(10)
tfidf_df.sort_values('冠城', ascending=False).head(10)