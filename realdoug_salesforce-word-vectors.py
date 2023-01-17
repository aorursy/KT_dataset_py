import gensim

import os
DATA_LOC = '../input/sfdc-corpus'



class MySentences(object):

    def __init__(self, dirname):

        self.dirname = dirname

 

    def __iter__(self):

        for fname in os.listdir(self.dirname):

            for line in open(os.path.join(self.dirname, fname)):

                yield line.split()

 

sentences = MySentences(DATA_LOC)



model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4)

model.save('sfw2v')
model.wv.most_similar(positive=['account'])
# glove: 0.255370411166

model.similarity('record', 'object')