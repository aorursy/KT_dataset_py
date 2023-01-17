import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import math
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return 1 - (float(numerator) / denominator)
    
def ConstructMatrixSynopsis(data):
    dsyn = np.zeros((len(data),len(data)))
    for i,dti in enumerate(data):
        #create frequency dictionary
        counter_i=Counter(dti)
        for j,dtj in enumerate(data[0:i+1]):
            counter_j=Counter(dtj)
            if i == j:
                dsyn[i][j] = -1
            else:
                dsyn[i][j] = float("{0:.5f}".format(get_cosine(counter_i,counter_j)))
    msyn = np.matrix(dsyn)
    newsyn = msyn + np.transpose(msyn)
    dfnewsyn = pd.DataFrame(newsyn,index=df['Anime_ID'],columns=df['Anime_ID'])
    return dfnewsyn
        
pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
en_stop = get_stop_words('en')
lemmatizer = WordNetLemmatizer()
# Input from csv
df = pd.read_csv('../input/datasynopsis-all-share-new.csv',sep='|')

# sample data
print(df['Synopsis'].head(2))
# list for tokenized documents in loop
texts = []

# loop through document list
for i in df['Synopsis'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in en_stop]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens]
    
    # remove one character
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]

       
    # add tokens to list
    texts.append(new_lemma_tokens)

# sample data
print(texts[0])
synopsis_matrix = ConstructMatrixSynopsis(texts)
print(synopsis_matrix.head(5))
synopsis_matrix.to_csv('sim_synopsis.csv')