import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk import Tree, pos_tag, ne_chunk

from nltk.sem.relextract import NE_CLASSES

sentence = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
def preprocess(sentence):

    sentence = nltk.word_tokenize(sentence)

    sentence = nltk.pos_tag(sentence)

    return sentence
def chunking(sentence):

    pattern = 'NP: {<DT>?<JJ>*<NN>}'

    cp = nltk.RegexpParser(pattern)

    cs = cp.parse(sentence)

    return sentence

    
def parsing_nltk(sentence):

    words = nltk.word_tokenize(sentence)

    tagged = nltk.pos_tag(words)

    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

    chunkParser = nltk.RegexpParser(chunkGram)

    chunked = chunkParser.parse(tagged)

    print(chunked)
def ner(sentence):

    ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

    print(ne_tree)
def namedentity(sentence):

    tagged_sent = ne_chunk(pos_tag(sent.split()))

    print(tagged_sent)

    ace_tags = NE_CLASSES['ace']

    for node in tagged_sent:

         if type(node) == Tree and node.label() in ace_tags:

                words, tags = zip(*node.leaves())

                print (node.label() + '\t' +  ' '.join(words))
print(sentence)

print("POS TAG OF USING NLTK ")

print(preprocess(sentence))

print("\n")

print("SHALLOW PARSING OF SENTENCE USING NLTK")

print(parsing_nltk(sentence))

print("\n")

print("NAMED ENTITY RECOGNITION OF SENTENCE USING NLTK")

ner(sentence)

print("-----------------------------------------------")
import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')


def spacyparser(i):

    print('Original Sentence: %s' % (i))

    doc = nlp(i)

    print('POS TAGGING USING SPACY')

    for token in doc:

        print(token.text, token.tag_)

    print("\n")

    print('DEPENDECY PARSING USING SPACY')

    for chunk in doc.noun_chunks:

        print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)

    displacy.render(doc, style='dep', jupyter=True, options={'distance': 50})

    print("\n")

    print('\nNAMED ENTITY RECOGNITION USING SPACY')

    for ent in doc.ents:

        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    #displacy.render(doc, style='ent', jupyter=True)
spacyparser(sentence)

    