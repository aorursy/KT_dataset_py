article = "Articles deliver information effectively, like other persuasive writing compositions. Which explains why article writing is an important skill which needs to be developed. The process of article writing, as compared to writing other compositions can be tricky. For example, a news article needs to be written without carrying any biased opinion from the writer. Article writing requires the writer to gather accurate information from reliable sources of information. You may also see essay writing examples Basically, article writing helps the writer develop both the writing and data gathering writing skills—which in turn develops his/her communication skills. At the end of the day, article writing, or writing in general, helps in improving an individual’s communication skills in general."
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from string import punctuation
sents = sent_tokenize(article)

sents
word_sent = word_tokenize(article.lower())

word_sent
_stopwords = set(stopwords.words('english') + list(punctuation))

_stopwords
word_sent = [word for word in word_sent if word not in _stopwords]
word_sent
from nltk.probability import FreqDist

freq = FreqDist(word_sent)

freq
from heapq import nlargest
nlargest(10, freq, key=freq.get)
from collections import defaultdict
ranking = defaultdict(int)

for i, sent in enumerate(sents):

    for w in word_tokenize(sent.lower()):

        if w in freq: 

            ranking[i] += freq[w]
ranking
sent_idx = nlargest(2, ranking, key=ranking.get)

sent_idx
[sents[i] for i in sent_idx]
def summarize(text, n):

    sents = sent_tokenize(text)

    

    assert n <= len(sents)

    words = word_tokenize(text.lower())

    _stopwords = set(stopwords.words('english') + list(punctuation))

    

    

    # filtering words

    words = [word for word in words if word not in _stopwords]

    

    # calculate freq distr

    freq = FreqDist(words)

    

    # calcualte significance score

    ranking = defaultdict(int)

    for i,sent in enumerate(sents):

        for word in word_tokenize(sent.lower):

            if word in freq:

                ranking[i] += freq[w]

                

    sent_idx = nlargest(2, ranking, key=ranking.get)

    return [sent[i] for i in sent_idx]

            

    

    