import numpy as np

from gensim.models import Word2Vec

from annoy import AnnoyIndex

from functools import reduce
import os
os.listdir('/kaggle/input')
with open('/kaggle/input/sessions', 'r') as f:

    sessions = [line[:-1].split(' ') for line in f]
model = Word2Vec(sessions, min_count=5, window=5, workers=8, size=100, iter=20)
# create a quick mapping from product to how many sessions it appeared in

# proxy for popularity 

i_to_num_sessions = {}

for session in sessions:

    for i in session:

        i_to_num_sessions[i] = i_to_num_sessions.get(i, 0) + 1
with open('/kaggle/input/id_to_title', 'r') as f:

    id_to_title = [line[:-1] for line in f]
# split a title by space and make lowercase

def tokenize(title):

    return title.lower().split(" ")
available_product_ids = set(model.wv.index2word)

inverted_index = {}



for i, title in enumerate(id_to_title):

    if i % 100000 == 0: print(i)

    if str(i) in available_product_ids:

        tokens = tokenize(title)

        for token in tokens:

            inverted_index[token] = inverted_index.get(token, set())

            inverted_index[token].add(i)
findTitlesLike = lambda words: sorted(list(map(lambda i:(i, id_to_title[i]), list(reduce(lambda acc, word: acc.intersection(inverted_index.get(word, set())), words, inverted_index[words[0]])))), key=lambda x: -1 * i_to_num_sessions[str(x[0])])[:10] 



getSimilarProducts = lambda i: list(map(lambda p: (p, id_to_title[int(p[0])]), model.wv.similar_by_word(str(i))))

findTitlesLike(['macbook', 'pro', 'retina', '256gb'])[:5]
getSimilarProducts(497370)[:5]
findTitlesLike(['nintendo', 'switch', 'console'])[:5]
getSimilarProducts(813190)[:5]
findTitlesLike(['instant', 'pot'])
getSimilarProducts(925724)[:5]
findTitlesLike(['harry', 'potter', 'deathly', 'hallows'])[:5]
getSimilarProducts(102115)[:5]
findTitlesLike(['airpods'])[:5]
getSimilarProducts(322055)[:5] # airpods headphones
getSimilarProducts(152574)[:5] # airpods case
import time
start = time.time()



for i in range(100):

    pid = model.wv.index2word[i]

    model.wv.similar_by_word(pid)

end = time.time()



print(end - start)
ai = AnnoyIndex(100)
for i, v in enumerate(model.wv.vectors):

    try:

        ai.add_item(int(model.wv.index2word[i]), v)

    except:

        print(model.wv.index2word[i])
ai.build(4)
findTitlesLike(['nintendo', 'switch', 'console'])[:5]
[(i, id_to_title[i]) for i in ai.get_nns_by_item(813190, 5)]
start = time.time()



for i in range(10, 45000):

    pid = model.wv.index2word[i]

    ai.get_nns_by_item(int(pid), 5)

end = time.time()



print(end - start)