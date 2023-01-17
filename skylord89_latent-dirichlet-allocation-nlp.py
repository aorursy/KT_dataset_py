import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('french')
lemma=WordNetLemmatizer()
from string import punctuation
import unicodedata
import sys
from nltk.corpus import stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
%matplotlib inline
data = pd.read_csv('../input/Trainv2.csv')[:500]
stop_words = stopwords.words('french')
stop_words.append('les')
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
def clean_text(col):
    result=[]
    long = len(col) # nombre de lignes à traiter
    
    for i in range(0,long):
        sys.stdout.write('\r'+str(i+1)+'/'+str(long)) # message de progression
        text=str(col[i])        
        text=re.sub('(\d*:\d*)|(\d*/\d*/\d*)|(\')|(\()|(\))|(,)|( - )|(\.)|(/)|(@)|(\|)',' ',text)        
        text = " ".join([stemmer.stem(i) for i in text.split() if (i.lower() not in stop_words) and (len(i) > 2)])
        text = strip_accents(text)
        result.append(text)
        
    return result
data['avi_text_cleaned'] = clean_text(data['avi_text'])
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})    
    
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
freq_words(data['avi_text_cleaned'])
tokenized = pd.Series(data['avi_text_cleaned']).apply(lambda x: x.split())
dictionary = corpora.Dictionary(tokenized)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized]
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
LDA = gensim.models.ldamodel.LdaModel
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=39, random_state=100, chunksize=1000, passes=50)
lda_model.print_topics()
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis