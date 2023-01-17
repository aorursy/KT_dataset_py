from nltk import word_tokenize, pos_tag

from nltk.corpus import wordnet as wn



 

def penn_to_wn(tag):

    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """

    if tag.startswith('N'):

        return 'n'

 

    if tag.startswith('V'):

        return 'v'

 

    if tag.startswith('J'):

        return 'a'

 

    if tag.startswith('R'):

        return 'r'

 

    return None

 

def tagged_to_synset(word, tag):

    wn_tag = penn_to_wn(tag)

    if wn_tag is None:

        return None

    try:

        return wn.synsets(word, wn_tag)[0]

    except:

        return None

 

def sentence_similarity(sentence1, sentence2):

    """ compute the sentence similarity using Wordnet """

    # Tokenize and tag

    sentence1 = pos_tag(word_tokenize(sentence1))

    sentence2 = pos_tag(word_tokenize(sentence2))

 

    # Get the synsets for the tagged words

    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]

    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

 

    # Filter out the Nones

    synsets1 = [ss for ss in synsets1 if ss]

    synsets2 = [ss for ss in synsets2 if ss]

 

    score, count = 0.0, 0

 

    # For each word in the first sentence

    for synset in synsets1:

        # Get the similarity value of the most similar word in the other sentence

        _temp = [synset.path_similarity(ss) for ss in synsets2]

        try:

          best_score = max([x for x in _temp if x])

        except:

          best_score = 0

 

        # Check that the similarity could have been computed

        if best_score is not None:

            score += best_score

            count += 1

    # Average the values

    try:

        score /= count

    except:

        score = 0.0

    return score

# Sentences taken from Wikipedia.

sentences = [

    "The dog and the extant gray wolf are sister taxa as modern wolves.",

    "The dog was the first species to be domesticated.",

    "The Welsh Corgi is a small type of herding dog that originated in Wales.",

    "The Persian cat is a long-haired breed of cat.",

    "The wolf is also known as the gray wolf or grey wolf.",

    "The Siberian Husky dog breed has a beautiful, thick coat.",

    "The Siberian Husky is one of dog breeds."

]

 

focus_sentence = "The Siberian Husky is one of dog breeds."



result = []

for sentence in sentences:

    dat = []

    dat.append(sentence_similarity(focus_sentence, sentence))

    dat.append(focus_sentence)

    dat.append(sentence)

    result.append(dat)

result.sort(reverse=True)

for res in result:

    print ("Similarity(\"" + res[1] + "\", \"" + res[2] + "\")")

    print ("= " + str(res[0]))