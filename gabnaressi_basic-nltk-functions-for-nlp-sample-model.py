import numpy as np 

import pandas as pd 

import nltk

import string

import re



from itertools import repeat

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col='id')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col='id')
df.head(5)
def prep_step1(df):

    df['text_prep'] = df['text'].copy()





    def count_regex(text, regex):

        return len(re.findall(regex,str(text)))



    df['count_links'] = list(map(count_regex, df['text_prep'],repeat('((http|https|www)[^ ]+\040?)|((.*\.com)[^ ]*\040?)')))

    df['text_prep'] = df['text_prep'].str.replace('((http|https|www)[^ ]+\040?)|((.*\.com)[^ ]*\040?)', '')



    df['count_uppercasew'] = list(map(count_regex, df['text_prep'],repeat('([A-Z]{2,}) ')))

    df['text_prep'] =  df['text_prep'].str.lower()



    df['count_punct'] = list(map(count_regex, df['text_prep'],repeat('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')))

    df['text_prep'] = df['text_prep'].str.replace('[{}]'.format(string.punctuation), '')



    df['count_numbers'] = list(map(count_regex, df['text_prep'],repeat('[0-9]')))

    df['text_prep'] = df['text_prep'].str.replace('[0-9]', '')



    df['text_prep'] = df['text_prep'].str.strip()



    df['text_prep'] = df['text_prep'].astype(str)

    

    return df.copy()



df = prep_step1(df)
def prep_step2(df):

    stop_words = set(stopwords.words('english')) 



    def remove_stopwords(text):

        return ' '.join([x for x in word_tokenize(text) if x not in stop_words ])



    df['text_prep_nonstop'] = list(map(remove_stopwords,df['text_prep'].to_list()))

    

    return df.copy()



df = prep_step2(df)
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

# Ugly code, I know



def prep_step3(df):

    from nltk.stem import WordNetLemmatizer 

    lemmatizer = WordNetLemmatizer()



    def lemmatize_text(text):

        word_and_pos = nltk.pos_tag(word_tokenize(text))    

        lemmas = list()

        for x in word_and_pos:

            lemma = ''

            try:

                # lemmatize(Word, POS) can raise errors but lemmatize(Word) always returns a value

                lemma = lemmatizer.lemmatize(x[0],x[1])

            except:

                lemma = lemmatizer.lemmatize(x[0])

            lemmas.append(lemma)



        return ' '.join(lemmas)



    df['text_prep_nonstop_lemmatized'] = list(map(lemmatize_text, df['text_prep_nonstop'].to_list()))

    return df.copy()



df = prep_step3(df)
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english',min_df=0.005)

tfidf = tfidfvectorizer.fit_transform(df['text_prep_nonstop_lemmatized'])
x = pd.DataFrame.sparse.from_spmatrix(tfidf,columns=tfidfvectorizer.get_feature_names(), index=df.index)

y = df['target'].copy()



x = x.merge(df[['count_links','count_uppercasew','count_numbers','count_punct']], left_index=True, right_index=True)



x
from sklearn.linear_model import LogisticRegression



# Create the model with 100 trees

model = LogisticRegression(max_iter=1500)

# Fit on training data

model.fit(x, y)
from sklearn import model_selection

scores = model_selection.cross_val_score(model, x, y, cv=3, scoring="f1")
scores
#df_test = prep_step1(df_test)

#df_test = prep_step2(df_test)

#df_test = prep_step3(df_test)



#tfidf_test = tfidfvectorizer.transform(df_test['text_prep_nonstop_lemmatized'])



#x_test = pd.DataFrame.sparse.from_spmatrix(tfidf_test,columns=tfidfvectorizer.get_feature_names(), index=df_test.index)

#y_test = df_test['target'].copy()



#pred = model.predict(x_test)