import numpy as np

import re

from nltk.corpus import brown

import random

import operator
def bigrams(corpus):

    if type(corpus) is str:

        corpus = re.findall('[\w\'\"’]+', corpus)

    tokens = [word.lower() for word in corpus]

    return [[tokens[i], tokens[i+1]] for i, word in enumerate(tokens) if not i == len(tokens)-1]





def prednext(bigram, word):

    pwords = [pair[1] for pair in bigram if pair[0] == word]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return (pwords[probability.index(max(probability))])



def generate(bigram, start, no_of_word):

    sentence = []

    sentence.append(start)

    for i in range(no_of_word):

        try:

            sentence.append(prednext(bigram, sentence[-1]))

        except:

            print('word not in corpus')

            break

    return (' ').join(sentence)
bigram = bigrams(brown.words())
generate(bigram[:10000], 'a', 15)
def trigrams(corpus):

    if type(corpus) is str:

        corpus = re.findall('[\w\'\"’]+', corpus)

    tokens = [word.lower() for word in corpus]

    return [[tokens[i], tokens[i+1], tokens[i+2]] for i, word in enumerate(tokens) if not i > len(tokens)-3]
def prednext2(trigram, word):

    pwords = [('8@').join(pair[1:]) for pair in trigram if pair[0] == word]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return pwords[probability.index(max(probability))].split("8@")



def generate2(trigram, start, no_of_word):

    sentence = []

    sentence.append(start)

    for i in range(no_of_word):

        try:

            sentence += prednext2(trigram, sentence[-1])

            if len(sentence) > no_of_word:

                break

        except:

            print('word not in corpus')

            break

    return (' ').join(sentence)
trigram = trigrams(brown.words())
generate2(trigram[:100000], 'a', 20)
def prednext3(trigram, word1, word2):

    pwords = [pair[2] for pair in trigram if pair[0] == word1 and pair[1] == word2]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return pwords[probability.index(max(probability))]



def generate3(trigram, word1, word2, no_of_word):

    sentence = []

    sentence.append(word1)

    sentence.append(word2)

    for i in range(no_of_word):

        try:

            sentence.append(prednext3(trigram, sentence[-2], sentence[-1]))

            if len(sentence) > no_of_word:

                break

        except:

            print('word not in corpus')

            break

    return (' ').join(sentence)
generate3(trigram[:100000], 'it', 'is', 100)
def prednext4b(trigram, word1, word3):

    pwords = [pair[1] for pair in trigram if pair[0] == word1 and pair[2] == word3]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return pwords[probability.index(max(probability))]



def prednext4a(trigram, word1, word2):

    pwords = [pair[2] for pair in trigram if pair[0] == word1 and pair[1] == word2]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    word3 =  pwords[probability.index(max(probability))]

    w2 = prednext4b(trigram, word1, word3)

    return  w2, word3



def generate4(trigram, word1, word2, no_of_word):

    sentence = []

    sentence.append(word1)

    sentence.append(word2)

    for i in range(no_of_word):

        try:

            w2, word3 = prednext4a(trigram, sentence[-2], sentence[-1])

            sentence[-1] = w2

            sentence.append(word3)

            if len(sentence) > no_of_word:

                break

        except:

            print('word not in corpus')

            break

    return (' ').join(sentence)
generate4(trigram[:10000], 'it', 'is', 100)
def prednext5c(trigram, word2, word3):

    pwords = [pair[0] for pair in trigram if pair[1] == word2 and pair[2] == word3]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return pwords[probability.index(max(probability))]



def prednext5b(trigram, word1, word3):

    pwords = [pair[1] for pair in trigram if pair[0] == word1 and pair[2] == word3]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    return pwords[probability.index(max(probability))]



def prednext5a(trigram, word1, word2):

    pwords = [pair[2] for pair in trigram if pair[0] == word1 and pair[1] == word2]

    random.shuffle(pwords)

    probability = []

    for pword in pwords:

        prob = pwords.count(pword)

        probability.append((prob)/len(pwords))

    word3 =  pwords[probability.index(max(probability))]

    w2 = prednext5b(trigram, word1, word3)

    w1 = prednext5c(trigram, w2, word3)

    return  w1, w2, word3



def generate5(trigram, word1, word2, no_of_word):

    sentence = []

    sentence.append(word1)

    sentence.append(word2)

    for i in range(no_of_word):

        try:

            w1, w2, word3 = prednext5a(trigram, sentence[-2], sentence[-1])

            sentence[-2] = w1

            sentence[-1] = w2

            sentence.append(word3)

            if len(sentence) > no_of_word:

                break

        except:

            print('word not in corpus')

            break

    return (' ').join(sentence)
generate5(trigram[:10000], 'it', 'a', 20)
# bigram

print(generate(bigram[:10000], 'it', 20))
#trigram type 1

(generate2(trigram[:10000], 'it', 20))
#trigram type 2

generate3(trigram[:100000], 'it', 'is',20)
# trigram type 4

generate4(trigram[:100000], 'it', 'is',20)
def get_word2idx(n_vocab=2000):



    word2idx = {'START' : 0, 'END' : 1}

    idx2word = ['START', 'END']

    word_freq_count = {

        0 : float('inf'), 

        1 : float('inf')

    }

    i = 2

    sentences = brown.sents()

    idxsentences = []

    for sentence in sentences:

        idxsentence = []

        idxsentence.append(0)

        for word in sentence:

            word = word.lower()

            if word not in word2idx:

                word2idx[word] = i

                idx2word.append(word)

                i += 1

            word_freq_count[word2idx[word]] = word_freq_count.get(word2idx[word], 0)+1

            idxsentence.append(word2idx[word])

        idxsentence.append(1)

        idxsentences.append(idxsentence)



    sortedwordfreqcount = sorted(word_freq_count.items(), key = operator.itemgetter(1), reverse = True)



    word2idx_small = {}

    new_idx = 0

    idx_new_idx_map = {}

    for idx, count in sortedwordfreqcount[:n_vocab]:

        word = idx2word[idx]

        word2idx_small[word] = new_idx

        idx_new_idx_map[idx] = new_idx

        new_idx += 1



    word2idx_small['UNKNOWN'] = new_idx 

    unknown = new_idx



    sentences_small = []

    for sentence in idxsentences:

        if len(sentence) > 1:

            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]

            sentences_small.append(new_sentence)



    return sentences_small, word2idx_small, unknown





sentences, word2idx, unknown = get_word2idx(10000)
V = len(word2idx) # vocabulary size



start_idx = word2idx['START']

end_idx = word2idx['END']



V
def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):

    bigram_probs = np.ones((V, V)) * smoothing

    for sentence in sentences:

        for i in range(len(sentence)):

            if i == 0:

                # beginning word

                bigram_probs[start_idx, sentence[i]] += 1

            else:

                # middle word

                bigram_probs[sentence[i-1], sentence[i]] += 1

        

            if i == len(sentence) - 1:

                # last word

                bigram_probs[sentence[i], end_idx] += 1

    

    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)

    return bigram_probs
bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)
def get_score(sentence):

    score = 0

    for i in range(len(sentence)):

        if i == 0:

            # beginning word

            score += np.log(bigram_probs[start_idx, sentence[i]])

        else:

            # middle word

            score += np.log(bigram_probs[sentence[i-1], sentence[i]])

            

    # final word

    score += np.log(bigram_probs[sentence[-1], end_idx])



    # normalize the score

    return score / (len(sentence) + 1)
def get_index(sentence):

    return [word2idx[word] if word in word2idx.keys() else unknown for word in sentence.split(" ")]
sentence = generate5(trigram[:100000], 'it', 'is',20)

print(sentence)

get_score(get_index(sentence))
sentence = generate4(trigram[:100000], 'it', 'is',20)

print(sentence)

get_score(get_index(sentence))
sentence = generate3(trigram[:100000], 'it', 'is',20)

print(sentence)

get_score(get_index(sentence))
sentence = generate2(trigram[:100000], 'it',20)

print(sentence)

get_score(get_index(sentence))
sentence = generate(bigram[:100000], 'it',20)

print(sentence)

get_score(get_index(sentence))