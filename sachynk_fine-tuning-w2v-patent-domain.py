from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import gensim

import pandas as pd

from nltk.tokenize import RegexpTokenizer
law_patents = pd.read_excel("../input/law-patent/law patents without summary.xlsx")
sample_patents = law_patents.Abstract.astype(str)

sample_patents = sample_patents.apply(lambda x: x.replace("Abstract:\n\n",""))

tokenizer = RegexpTokenizer(r'\w+')

sample_patents_tokenize = [w.lower() for w in sample_patents]

sample_patents_tokenize = [tokenizer.tokenize(i) for i in sample_patents_tokenize]
from gensim.models import FastText

model_ted = FastText(sample_patents_tokenize, size=300, window=5, min_count=5, workers=4,sg=1)
# update existing embedding

model_2 = Word2Vec(size=100, min_count=1)

model_2.build_vocab(sample_patents_tokenize)

total_examples = model_2.corpus_count

model_2.intersect_word2vec_format("../input/w2vec-patent-domain/W2Vec_Patent_Domain.txt", binary=False, lockf=1.0)

model_2.train(sample_patents_tokenize, total_examples=total_examples, epochs=5)
model_2.save("word_embedding.model")

model_2.wv.save_word2vec_format('word_embedding_w2v.model')
model_2.most_similar('display')
model_ted.most_similar('display')