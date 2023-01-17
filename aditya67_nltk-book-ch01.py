import nltk
from nltk.book import *
text1
text1.concordance('monstrous')
text2.concordance("affection")
text1.similar("monstrous")
text1.common_contexts(['curious','wise'])
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
len(text3)
sorted(set(text3))
len(set(text3))
def lexical_diversity(text):
    return len(set(text))/len(text)
def percentage(count,total):
    return 100*count/total
lexical_diversity(text3)
percentage(text3.count("the"),len(text3))
sent2
saying = ['After', 'all', 'is', 'said', 'and', 'done','more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]
fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(10)
fdist1.plot(50, cumulative=True)
fdist1.hapaxes()
text4.collocations()
b  = [w for w in set(text5) if w.startswith('b')]
print(sorted(b))
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')
