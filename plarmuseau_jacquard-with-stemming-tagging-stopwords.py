import pandas as pd

# timing function

import time   

start = time.clock() #_________________ measure efficiency timing



# read data

test = pd.read_csv('../input/test.csv',encoding='utf8')[:100]

train = pd.read_csv('../input/train.csv',encoding='utf8')[:100]

print(train.head(2))

train.fillna(value='leeg',inplace=True)

test.fillna(value='leeg',inplace=True)



end = time.clock()

print('open:',end-start)
# Imports

import nltk.corpus

import nltk.stem.snowball

from nltk.corpus import wordnet

import string



# Get default English stopwords and extend with punctuation

stopwords = nltk.corpus.stopwords.words('english')



stopwords.extend(string.punctuation)

stopwords.append('')



def get_wordnet_pos(pos_tag):

    if pos_tag[1].startswith('J'):

        return (pos_tag[0], wordnet.ADJ)

    elif pos_tag[1].startswith('V'):

        return (pos_tag[0], wordnet.VERB)

    elif pos_tag[1].startswith('N'):

        return (pos_tag[0], wordnet.NOUN)

    elif pos_tag[1].startswith('R'):

        return (pos_tag[0], wordnet.ADV)

    else:

        return (pos_tag[0], wordnet.NOUN)



# Create tokenizer and stemmer



lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()



def is_ci_partial_noun_set_token_stopword_lemma_match(a, b):

    """Check if a and b are matches."""

    

    pos_a = map(get_wordnet_pos, nltk.pos_tag(a.lower().strip(string.punctuation).split()))

    pos_b = map(get_wordnet_pos, nltk.pos_tag(b.lower().strip(string.punctuation).split()))

    lemmae_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in stopwords]

    lemmae_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_b if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in stopwords]

    q1=''.join(lemmae_a).split()

    q2=''.join(lemmae_b).split()

    # Calculate Jaccard similarity

    ratio = len(set(lemmae_a).intersection(lemmae_b)) / float(len(set(lemmae_a).union(lemmae_b))+.001)

    return (round(ratio,2))



result=[]

for xi in range(0,len(train)):

    result.append(is_ci_partial_noun_set_token_stopword_lemma_match(train.iloc[xi]['question1'],train.iloc[xi]['question2']))



train['jacq']=result

print(train)

end = time.clock()

print('open:',end-start)

print((end-start)/len(result))