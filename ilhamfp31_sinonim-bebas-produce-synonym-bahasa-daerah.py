import gensim

import json

DIR = '../input/sinonim-bebas-build-word2vec'

TOP_N = 6
def produce_synonym(w2v):

    top_n_similar = {}

    for key in w2v.wv.vocab.keys():

        top_n_similar[key] = w2v.wv.similar_by_word(key, topn=TOP_N)



    top_n_similar_true = {}

    for key in top_n_similar.keys():

        for tuple_similar in top_n_similar[key]:

            key_similar = tuple_similar[0]

            proba_similar = tuple_similar[1]

            list_of_similar_from_key_similar = [x[0] for x in top_n_similar[key_similar]]



            if key not in list_of_similar_from_key_similar:

                continue

            

            synonim = {

                    "word":key_similar,

                    "similarity":proba_similar

                }

            if key in top_n_similar_true:

                top_n_similar_true[key].append(synonim)

            else:

                top_n_similar_true[key] = [synonim]



    print(len(top_n_similar))

    print(len(top_n_similar_true))

    print(list(top_n_similar_true.keys())[:10])

    return top_n_similar_true



def produce_json(synonym_dict, language_code):

    result = []

    for key in synonym_dict.keys():

        result.append({

            "word":key,

            "synonyms":synonym_dict[key]

        })

        

    with open('data_{}.json'.format(language_code), 'w+') as outfile:

        json.dump(result, outfile)
!ls '../input/sinonim-bebas-build-word2vec'
language_code = 'su'

path = '{}/{}wiki_word2vec_100.model'.format(DIR, language_code)

w2v = gensim.models.word2vec.Word2Vec.load(path)
synonym_dict = produce_synonym(w2v)

produce_json(synonym_dict, language_code)
synonym_dict['siga']
language_code = 'min'

path = '{}/{}wiki_word2vec_100.model'.format(DIR, language_code)

w2v = gensim.models.word2vec.Word2Vec.load(path)

synonym_dict = produce_synonym(w2v)

produce_json(synonym_dict, language_code)
synonym_dict['saurang']
synonym_dict['tanamo']
synonym_dict['babeda']
language_code = 'ban'

path = '{}/{}wiki_word2vec_100.model'.format(DIR, language_code)

w2v = gensim.models.word2vec.Word2Vec.load(path)

synonym_dict = produce_synonym(w2v)

produce_json(synonym_dict, language_code)
language_code = 'jv'

path = '{}/{}wiki_word2vec_100.model'.format(DIR, language_code)

w2v = gensim.models.word2vec.Word2Vec.load(path)

synonym_dict = produce_synonym(w2v)

produce_json(synonym_dict, language_code)
language_code = 'ace'

path = '{}/{}wiki_word2vec_100.model'.format(DIR, language_code)

w2v = gensim.models.word2vec.Word2Vec.load(path)

synonym_dict = produce_synonym(w2v)

produce_json(synonym_dict, language_code)