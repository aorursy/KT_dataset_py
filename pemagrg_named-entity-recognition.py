

import spacy

nlp = spacy.load("en")

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

text = nlp(text)

labels = set([w.label_ for w in text.ents])

for label in labels:

    entities = [e.string for e in text.ents if label==e.label_]

    entities = list(set(entities))

    print( label,entities)
import nltk

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """



word = nltk.word_tokenize(text)

pos_tag = nltk.pos_tag(word)

chunk = nltk.ne_chunk(pos_tag)

for ele in chunk:

    if isinstance(ele, nltk.Tree):

        print (ele)

# NE = [ " ".join(w for w, t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]

# print (NE)