word_list = nltk.corpus.words.words()
type_labels = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ',
               'ISTP', 'ISFP', 'INFP', 'INTP',
               'ESTP', 'ESFP', 'ENFP', 'ENTP',
               'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
merge_list = word_list + type_labels
def cleaning(text):
    """
    remove punctuation
    remove numbers
    remove stop words
    remove gibberish
    """
    
    punc_num = string.punctuation + '0123456789'
    punct = ''.join([c for c in text if c not in punc_num]).lower()
    stop = ([w for w in punct.split() if w not in nltk.corpus.stopwords.words('english')])
    
    return ([x for x in stop if x in merge_list] )
weblinks['clean'] = weblinks['text'].apply(cleaning)
cleaning(weblinks['text'][0])