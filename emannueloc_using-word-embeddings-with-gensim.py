import pandas as pd

df = pd.read_csv('../input/train.csv')

corpus_text = '\n'.join(df[:5000]['comment_text'])

sentences = corpus_text.split('\n')

sentences = [line.lower().split(' ') for line in sentences]
def clean(s):

    return [w.strip(',."!?:;()\'') for w in s]

sentences = [clean(s) for s in sentences if len(s) > 0]
from gensim.models import Word2Vec



model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)
vectors = model.wv

del model
vectors['good']
print(vectors.similarity('you', 'your'))

print(vectors.similarity('you', 'internet'))
vectors.most_similar('i')