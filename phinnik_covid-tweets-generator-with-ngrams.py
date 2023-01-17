import pandas as pd
import re
from collections import defaultdict
import random

df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
df.head()
def preprocess_text(text: str):
    """
    
    :param text: - tweet text
    :returns: - list of preprocessed sentences of a tweet.
    """
    
    # lower capital letters
    text = text.lower()
    
    # delete links
    text = re.sub(r'https.+?', '', text)

    
    # delete everything except punctuation marks
    text = re.sub(r'[^a-z !?.\n]', '', text)
    
    # remove whitespace before punctuation mark/whitespace/end of line
    text = re.sub(r' (\?|\!|\.|\n| |$)', r'\1', text)

    
    # remove whitespace at the begining of the line or after punctuation mark
    text = re.sub(r'(\?|\!|\.|\n|^) ', r'\1', text)
    
    # spliting by puncuation mark or new line
    texts = re.split('\?|\.|\!|\n', text)
    
    # deleting empty sentences
    texts = [t for t in texts if t]
    

    return texts
test_tweet = 'BiG, brother is 100% watching  you .Are You scared???'
preprocess_text(test_tweet)
corpus = []
for t in df['text']:
    corpus.extend(preprocess_text(t))
for c in corpus[:10]:
    print(c)
def create_ngrams(N: int, min_count: int = 5, min_tokens: int = 5):
    
    """
    :param N: ngram size
    :param min_count: minimum acceptable count of ngram founds
    :param min_tokens: minimum acceptable count of tokens in sentence
    
    :returns: Dict[Tuple[str], int] 
    """
    ngram = defaultdict(int)
    
    for line in corpus:
        tokens = line.split()
        
        if len(tokens) > min_tokens:
            for i in range(len(tokens) - N + 1):
                ngram[tuple(tokens[i : i + N])] += 1
                
            ngram[tuple(['^'] + tokens[0 : N - 1])] += 1
            
            ngram[tuple(tokens[-N:-1] + ['$'])] += 1
            
    ngram = {key: value for key, value in ngram.items() if value > min_count}
    
    return ngram
    
trigrams = create_ngrams(3)
len(trigrams)
list(trigrams.items())[:10]
def get_next_token(previous_tokens: tuple, ngrams: dict, method: str = 'random'):
    """
    Returns the next token if found one
    
    :param previous_tokens: previous tokens
    :param ngrams: ngrams - ngrams, where to search for the next token
    :param method: method of search. 
        'random' - searches for the random token. 
        'weighted' - searches for random token, but token wich was found more times will have more probability to be searched
        'most_common' - searches for the most common token
        
    :returns: str
    """
    matching_ngrams = {key: value for key, value in ngrams.items() if key[:2] == previous_tokens}
    
    if matching_ngrams:
        
        if method == 'random':
            return random.choice(list(matching_ngrams.keys()))[-1]
        
        elif method == 'weighted':
            return random.choices(list(matching_ngrams.keys()), weights=list(matching_ngrams.values()))[0][-1]
        
        elif method == 'most_common':
            return sorted(matching_ngrams.items(), key=lambda x: x[1])[-1][0][-1]
previous_tokens = ('in', 'the')

print('random:')
for _ in range(3):
    print('\t', get_next_token(previous_tokens, trigrams, method='random'))
    
print('weighted:')
for _ in range(3):
    print('\t', get_next_token(previous_tokens, trigrams, method='weighted'))

print('most_common:')
for _ in range(3):
    print('\t', get_next_token(previous_tokens, trigrams, method='most_common'))
def get_starter(ngrams, method='random'):
    starters = {key: value for key, value in ngrams.items() if key[0] == '^'}

    if method == 'random':
        return random.choice(list(starters.keys()))
    
    elif method == 'weighted':
        return random.choices(list(starters.keys()), weights=list(starters.values()))[0]
    
    elif method == 'most_common':
        return sorted(starters.items(), key=lambda x: x[1])[-1][0]
print('random:')
for _ in range(3):
    print('\t', get_starter(trigrams, method='random'))
    
print('weighted:')
for _ in range(3):
    print('\t', get_starter(trigrams, method='weighted'))

print('most_common:')
for _ in range(3):
    print('\t', get_starter(trigrams, method='most_common'))
def generate_tweet(ngrams, min_length: int = 10, starter_method='random', next_method='most_common'):
    
    N = len(list(ngrams.keys())[0])
    
    starter = get_starter(ngrams, starter_method)
    tokens = list(starter)[1:]
    
    next_token = get_next_token(tuple(list(starter)[1:]), ngrams, next_method)
    while next_token:
        tokens.append(next_token)
        
        last_tokens = tuple(tokens[-N+1:])
        next_token = get_next_token(last_tokens, ngrams, next_method)
    
    if len(tokens) < min_length or tokens[-1] != '$':
        tweet = generate_tweet(ngrams, min_length, starter_method, next_method)
    else:
        tweet = ' '.join(tokens)
    return tweet 
random.seed(43)
tweets = [generate_tweet(trigrams) for _ in range(10)]
for tweet in tweets:
    print('*', tweet)
for tweet in tweets:
    print(tweet)
    for text in df['text']:
        if tweet[:-1] in text:
            print('\t', text.replace('\n', ' ').replace('\t', ''))
            
    else:
        print('\t--No matches')
    print()