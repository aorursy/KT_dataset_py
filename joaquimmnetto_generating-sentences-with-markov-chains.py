
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
#from plotnine import *

discursos = pd.read_csv("../input/discursos-impeachment.csv",
            header=0,
            names=['deputado','partido', 'estado', 'voto', 'genero', 'fala'])


discursos.deputado = discursos.deputado.apply(lambda v: v.lower()).astype('category')
discursos.partido = discursos.partido.astype(CategoricalDtype(discursos.partido.value_counts().index.tolist(), ordered=True))
discursos.estado = discursos.estado.astype(CategoricalDtype(discursos.estado.value_counts().index.tolist(), ordered=True))
discursos.voto = discursos.voto.apply(lambda v: v.lower())
discursos.voto = discursos.voto.astype(CategoricalDtype(list(set(discursos.voto)), ordered=True))
discursos.genero = discursos.genero.apply(lambda v: v.upper()).astype('category')
discursos.fala = discursos.fala.apply(lambda v: str(v).lower())


discursos.head()
discursos.tail()
discursos.describe()
corpus = ' '.join(discursos.fala.tolist())
len(corpus.split(' ')), len(corpus)
unig_corpus = [w for w in corpus.split(' ') if w.strip() != ''] 
def build_neighbor_dict(corpus):
    result = dict([(w,dict()) for w in list(set(corpus))])
    for i in range(0,len(corpus)-1):
        word = corpus[i]
        next_word = corpus[i+1]
        if next_word not in result[word].keys():
            result[word][next_word] = 0
        result[word][next_word] += 1
    
    for word, counts in result.items():
        total = sum(counts.values())
        result[word] = dict(list(map(lambda t: (t[0], t[1]/total), counts.items())))
    
    return result 
neigh_dict = build_neighbor_dict(unig_corpus)
def sentence_gen(neigh_dict, first_word_list, stop_condition):
    current_word = np.random.choice(first_word_list)
    sentence = [current_word.capitalize()]
    word_count = 0
    while 1:
        current_probs = neigh_dict[current_word]
        chosen_word = np.random.choice(np.array(list(current_probs.keys())), p = np.array(list(current_probs.values())))
        sentence.append(chosen_word)        
        if stop_condition(sentence, word_count): #Stops when there is at least min_word_count and the current word ends in a '.'.
            return ' '.join(sentence)
        else:
            current_word = chosen_word
            word_count+=1
        

#Feijão-com-arroz
for i in range(0,5):
    print(i,":",
          sentence_gen(neigh_dict, 
             first_word_list = list(set(unig_corpus)),
             stop_condition = lambda _, word_count: word_count == 20), '\n') 
for i in range(0,5):
    print(i,":",
          sentence_gen(neigh_dict, 
             first_word_list = list(set(discursos.fala.apply(lambda s: s.split(" ")[0]))),
             stop_condition = lambda _, word_count: word_count == 20), '\n') 

#Mais polimento - Primeira palavra e condição de parada simples (ponto final)
for i in range(0,5):
    print(i,":",
          sentence_gen(neigh_dict, 
             first_word_list = discursos.fala.apply(lambda s: s.split(" ")[0]),
             stop_condition = lambda sentence, word_count: 
                       sentence[-1][-1] in ['.','!','?'] and word_count > 20),'\n')
#Polishing the most of that pile of turd
for i in range(0,5):
    print(i,":",
          sentence_gen(neigh_dict, 
             first_word_list = discursos.fala.apply(lambda s: s.split(" ")[0]),
             stop_condition = lambda sentence, word_count: 
                       ('sim' in ' '.join(sentence) or 'não' in ' '.join(sentence)) and 
                       sentence[-1][-1] in ['.','!','?'] and 
                       word_count > np.random.normal(53,32))
          ,'\n')
def ngramify(corpus, n):
    return list(['_'.join(corpus[i : i+n]) for i in range(len(corpus)-n+1)])
n = 2
sentences = [["$"]*(n-1) + sentence.split(' ') for sentence in discursos.fala.tolist()]
sentences = [ [w for w in sentence if w.strip() != ''] for sentence in sentences]
ngrams = []
for sentence in sentences:
    ngrams +=ngramify(sentence, n)
del(sentences)
ngrams[:5]
ngram_neigh_dict = build_neighbor_dict(ngrams)
def ngram_sentence_gen(neigh_dict, n, first_word_list, stop_condition):    
    first_word = ''
    while len(first_word.strip()) == 0:
        first_word = np.random.choice(first_word_list)
    sentence = [first_word.capitalize()]
    if n > 1:
        current_ngram = '_'.join(["$"]*(n-1)) + "_" + first_word
    else:
        current_ngram = first_word
    word_count = 0
    while 1:         
        next_ngrams = np.array(list(neigh_dict[current_ngram].keys()))
        next_probs = np.array(list(neigh_dict[current_ngram].values()))
        if next_ngrams.shape[0] == 0: return ' '.join(sentence)        
        chosen_ngram = np.random.choice(next_ngrams, p = next_probs)
        sentence.append(chosen_ngram.split('_')[-1])                
        if stop_condition(sentence, word_count):
            return ' '.join(sentence)
        else:
            current_ngram = chosen_ngram
            word_count+=1
        

#Polishing the most of that pile of turd
for i in range(0,5):
    print(i,":",
          ngram_sentence_gen(ngram_neigh_dict, n, 
             first_word_list = list(set(discursos.fala.apply(lambda s: s.split(" ")[0]))),
             stop_condition = lambda sentence, word_count: 
                       ('sim' in ' '.join(sentence) or 'não' in ' '.join(sentence)) and 
                       sentence[-1][-1] in ['.','!','?'] and 
                       word_count > np.random.normal(53,32))
          ,'\n')
generated_sentences = [
    ngram_sentence_gen(ngram_neigh_dict, n, 
             first_word_list = list(set(discursos.fala.apply(lambda s: s.split(" ")[0]))),
             stop_condition = lambda sentence, word_count: 
                       ('sim' in ' '.join(sentence) or 'não' in ' '.join(sentence)) and 
                       sentence[-1][-1] in ['.','!','?'] and 
                       word_count > np.random.normal(53,32)) 
    for i in range(0, 100)]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
discursos.fala.tolist()


def build_count_model(sentences, generated_sentences):
        model = CountVectorizer()
        tfidf = TfidfTransformer()
        return tfidf.fit_transform(model.fit_transform(sentences + generated_sentences))
        
matrix = build_count_model(discursos.fala.tolist(), generated_sentences)
corpus_vecs = matrix[:-100,]
generated_vecs = matrix[-100:,]

sim_matrix = cosine_similarity(corpus_vecs, generated_vecs)
max_sims = np.amax(sim_matrix, 0)



import matplotlib.pyplot as plt
plt.boxplot(max_sims)
plt.show()

def generate_for_n(n, generated_count):
    sentences = [["$"]*(n-1) + sentence.split(' ') for sentence in discursos.fala.tolist()]
    sentences = [ [w for w in sentence if w.strip() != ''] for sentence in sentences]
    ngrams = []
    for sentence in sentences:
        ngrams +=ngramify(sentence, n)
    ngram_neigh_dict = build_neighbor_dict(ngrams)
    fwords = discursos.fala.apply(lambda s: s.split(" ")[0])
    generated_sentences = [
        ngram_sentence_gen(ngram_neigh_dict, n, 
                 first_word_list = list(set(fwords)),
                 stop_condition = lambda sentence, word_count: 
                           ('sim' in ' '.join(sentence) or 'não' in ' '.join(sentence)) and 
                           sentence[-1][-1] in ['.','!','?'] and 
                           word_count > np.random.normal(53,32)) 
    for i in range(0, generated_count)]
    matrix = build_count_model(discursos.fala.tolist(), generated_sentences)
    corpus_vecs = matrix[:-generated_count,]
    generated_vecs = matrix[-generated_count:,]

    sim_matrix = cosine_similarity(corpus_vecs, generated_vecs)
    max_sims = np.amax(sim_matrix, 0)
    
    return max_sims, generated_sentences
sims_vecs = pd.DataFrame()
sentences = []
for n in range(1,21):    
    max_sims, sents = generate_for_n(n, 100)
    sentences.append(sents)
    sims_vecs[str(n)] = (max_sims)

plt.boxplot(sims_vecs.values)
plt.show()
print("2-grams")
for i,sent in enumerate(sentences[1][:5]):
    print(i, ":", sent)
print("3-grams")
for i,sent in enumerate(sentences[2][:5]):
    print(i, ":", sent)
print("10-grams")
for i,sent in enumerate(sentences[9][:5]):
    print(i, ":", sent)
print("20-grams")
for i,sent in enumerate(sentences[19][:5]):
    print(i, ":", sent)
