import nltk
from nltk.corpus import floresta

words = floresta.words()
fd = nltk.FreqDist(words)
print("Количество слов:", len(words))
print("Количество уникальных:", len(fd))
print("Слово с максимальной частотой:", fd.max())
def concordance(word, context = 30):
    print("Окружение слова '" + word + "'\n")
    for sent in floresta.sents():
        if word in sent:
            pos = sent.index(word)
            left = ' '.join(sent[ :pos])
            right = ' '.join(sent[pos + 1: ])
            print('%*s %s %-*s' % (context, left[-context: ], word, context, right[ :context]))
            
concordance("um")
from nltk.corpus import genesis

words = genesis.words()
fd = nltk.FreqDist(words)
print("Количество слов:", len(words))
print("Количество уникальных:", len(fd))
print("Слово с максимальной частотой:", fd.max())
def concordance2(word, context = 30):
    print("Окружение слова '" + word + "'\n")
    for sent in genesis.sents():
        if word in sent:
            pos = sent.index(word)
            left = ' '.join(sent[ :pos])
            right = ' '.join(sent[pos + 1: ])
            print('%*s %s %-*s' % (context, left[-context: ], word, context, right[ :context]))
            
concordance2("God")
import sys

def sentsLen():
    sents = genesis.sents("english-kjv.txt")
    av = 0; count = 0; maxL = 0; minL = sys.maxsize
    for sent in sents:
        currLen = len(sent)
        
        if currLen > maxL:
            maxL = currLen
        
        if currLen < minL:
            minL = currLen
            
        av += currLen
        count += 1
    
    print("Максимальная длина предложения:", maxL)
    print("Минимальная длина предложения:", minL)
    print("Средняя длина предложения:", round(av / count))
    
sentsLen()