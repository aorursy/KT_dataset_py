import pandas as pd
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
import nltk
from nltk import SnowballStemmer
import matplotlib.pyplot as plt
import itertools
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
%matplotlib inline
import os
print(os.listdir("../input/"))
files = {'amazon': '../input/amazon_cells_labelled.txt', 'imdb': '../input/imdb_labelled.txt', 'yelp': '../input/yelp_labelled.txt'}
df = pd.DataFrame()
for f in files.keys():
    df1 = pd.read_csv(files[f], sep='\t', names=['text','label'])
    df1['origin'] = f
    df = pd.concat([df, df1])
df['origin'].value_counts()
pd.crosstab(df['origin'],df['label'])
def pre_processing(text):
    # remove pontuacao e caracteres especiais
    text = text.lower()
    punctuation = re.compile('[%s]' % re.escape('!"#$%&\'()*´,-./:;<=>?@[\\]^_`{|}~|0123456789'))
    text = punctuation.sub('', text)
    # tokeniza palavras
    tokens = nltk.word_tokenize(text)
    # retira stopwords
    stops = set(stopwords.words("english"))
    tokens = [w for w in tokens if (not w in stops)]
    # realiza o stemming
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
df['text_vec'] = df['text'].apply(lambda x: pre_processing(x))
def analyze_words(df):
    print('Vectorize')
    counts = Counter(list(itertools.chain.from_iterable(df['text_vec'].values)))
    print('WordCloud')
    plt.figure(figsize=[15,30])
    wordcloud = WordCloud(max_words=1000, background_color='white', relative_scaling=.1, width=1200,
                          height=600).generate_from_frequencies(counts)
    wordcloud.to_file('wordcloud.jpg')
    plt.imshow(wordcloud)
    plt.show()
analyze_words(df)
analyze_words(df[df['label']==1])
analyze_words(df[df['label']==0])
# WordCloud Amazon
print("Amazon")
print("-----")
analyze_words(df[df['origin']=='amazon'])

print("IMDB")
print("-----")
analyze_words(df[df['origin']=='imdb'])

print("Yelp")
print("-----")
analyze_words(df[df['origin']=='yelp'])

# metodo pra calcular a media do vetor para as palavras da frase
def calcula_vetor(text, vector_model, n_vec):
    vetor = np.zeros(n_vec)
    for t in text:
        try:
            vetor = vetor + vector_model[t]
        except:
            pass
    vetor = vetor/len(text)
    return vetor
        
text = df['text_vec'].values
# tamanho do vetor de palavras padrão é 100, o que pode ser mudado passando o parâmetro size
n_vec = 150
text_vector = Word2Vec(text, size = n_vec)
X = [calcula_vetor(t, text_vector, n_vec) for t in text]
print(len(X))
print(X[0])
X = pd.DataFrame(X)
X = X.replace([-np.inf, np.inf], np.nan).fillna(0)
Y = df['label']
#Separando 20% dos dados para validação
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=0)
model_class = RandomForestClassifier(n_estimators=300, random_state=20)
model_class = model_class.fit(X_train, y_train)
y_pred = model_class.predict_proba(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred[:, 1])
roc = pd.DataFrame(
    {'tf': pd.Series(tpr - (1 - fpr)), 'threshold': pd.Series(threshold), 'tpr': tpr,
     'fpr': fpr})
roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
print(roc_t)
print(metrics.auc(fpr, tpr))
plt.plot(fpr,tpr)
plt.title('AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
