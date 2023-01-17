import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import stopwords
from wordcloud import WordCloud
train = pd.read_csv('../input/train.csv')
train.head()
print('total', train.shape[0])
print('sincere questions', train[train['target'] == 0].shape[0])
print('insincere questions', train[train['target'] == 1].shape[0])
class Vocabulary(object):
    
    def __init__(self):
        self.vocab = {}
        self.STOPWORDS = set()
        self.STOPWORDS = set(stopwords.words('english'))
        
    def build_vocab(self, lines):
        for line in lines:
            for word in line.split(' '):
                word = word.lower()
                if (word in self.STOPWORDS):
                    continue
                if (word not in self.vocab):
                    self.vocab[word] = 0
                self.vocab[word] +=1 
sincere_vocab = Vocabulary()
sincere_vocab.build_vocab(train[train['target'] == 0]['question_text'])
sincere_vocabulary = sorted(sincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])
for word, count in sincere_vocabulary[:10]:
    print(word, count)
insincere_vocab = Vocabulary()
insincere_vocab.build_vocab(train[train['target'] == 1]['question_text'])
insincere_vocabulary = sorted(insincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])
for word, count in insincere_vocabulary[:10]:
    print(word, count)
sincere_score = {}
for word, count in sincere_vocabulary:
    sincere_score[word] = count / insincere_vocab.vocab.get(word, 1)

wordcloud_sincere = WordCloud(width = 800, height = 800,background_color ='white', min_font_size = 10)
wordcloud_sincere.generate_from_frequencies(sincere_score) 
  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_sincere) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
insincere_score = {}
for word, count in insincere_vocabulary:
    insincere_score[word] = count / sincere_vocab.vocab.get(word, 1)

wordcloud_insincere = WordCloud(width = 800, height = 800,background_color ='white', min_font_size = 10)
wordcloud_insincere.generate_from_frequencies(insincere_score) 
  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_insincere) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 