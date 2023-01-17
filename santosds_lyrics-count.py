lyrics =[word.replace(',','').strip().upper().split() for word in open('../input/dubist.txt','r')]
import collections
import itertools
import matplotlib.pyplot as plt
words_lyrics=(list(itertools.chain.from_iterable(lyrics)))
def indexes(wordlist,word):
    return wordlist.index(word)
lyrics_dict=collections.defaultdict(list)
for word in words_lyrics:
        lyrics_dict[indexes(words_lyrics,word)].append(word)
x=[]
y=[]
for ind,words in lyrics_dict.items():
    if len(words)>1:
        x.append(','.join(list(set(words))))
        y.append(len(words))
plt.figure(num=None, figsize=(80, 45), dpi=60, facecolor=(0.9, 0.9, 0.9), edgecolor='black')
plt.title('Du Bist So Schmutzig lyrics by Scorpions',fontsize=100,verticalalignment='bottom')
plt.xticks(rotation=-80)
plt.tick_params(labelsize=55)
plt.xlabel('Repeated words',fontsize=70, verticalalignment='center')
plt.ylabel('Word count',fontsize=70, horizontalalignment='right')
plt.bar(x,y,0.7,color=(0.8, 0.15, 0.15))
plt.show()
#plt.savefig('plot.png')
