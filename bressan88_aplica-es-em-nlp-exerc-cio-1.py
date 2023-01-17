import pandas as pd

import numpy as np

import string

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.corpus import wordnet

from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('../input/questionanswer-dataset/S08_question_answer_pairs.txt', sep='\t')
df.head(10)
df.columns
df.info()
df = df.drop_duplicates( subset='Question' )

df.head(10)
df['ArticleTitle'] = df['ArticleTitle'].str.replace('_',' ')

df.head()
df['Question'] = df['ArticleTitle'] + ' ' + df['Question']

df.head()
df02 = df[['Question', 'Answer']]

df02.head()
df02.shape
df02 = df02.dropna()

df02.shape
print( type( tuple( df['Question'] ) ) )
stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()



def my_tokenizer(doc):

    words = word_tokenize(doc)

    pos_tags = pos_tag(words)

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        

        lemmas.append(lemmatizer.lemmatize(w[0], pos))



    return lemmas
tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)

#tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(df02['Question']))

print(tfidf_matrix.shape)
def pergunta_resposta(pergunta):

    

    query_vect = tfidf_vectorizer.transform([pergunta])

    similaridade = cosine_similarity(query_vect, tfidf_matrix)

    similaridade_max = np.argmax(similaridade)

    

    print('\n> Similarity:\n- {:.2%}\n'.format(similaridade[0, similaridade_max]))

    print('> Nearest Question:\n- {}\n'.format(df02.iloc[similaridade_max]['Question']))

    print('> Answer:\n- {}'.format(df02.iloc[similaridade_max]['Answer']))
question_user = input('Type your question here: ')

pergunta_resposta(question_user)
pergunta_resposta("Did Lincoln's Wife's Family support slavery?")
pergunta_resposta("Abraham_Lincoln Did Lincoln's Wife's Family support slavery?")