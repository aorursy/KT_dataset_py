import nltk

sentence = "My name is Amardeep Kumar. I live in Dhanabad"
token = nltk.word_tokenize(sentence)
nltk.pos_tag(token)
grammar = ('''

    NP: {<DT>?<JJ>*<NN>} # NP

    ''')

sentence = "the little yellow dog barked at the cat"

chunkParser = nltk.RegexpParser(grammar)

tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

tagged
tree = chunkParser.parse(tagged)
for subtree in tree.subtrees():

    print(subtree)

import spacy

from nltk import Tree





en_nlp = spacy.load('en')



def tok_format(tok):

    return "(->)".join([tok.orth_, tok.tag_, tok.dep_])





def to_nltk_tree(node):

    if node.n_lefts + node.n_rights > 0:

        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])

    else:

        return tok_format(node)

doc = en_nlp("The quick brown fox jumps over the lazy dog.")
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 

doc2 = "My father spends a lot of time driving my sister around to dance practice."

doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc_complete = [doc1]

doc_clean = [doc.split() for doc in doc_complete]
doc1 = "Sugar is bad to consume. My brother likes to have sugar, but not my father." 

doc2 = "My father spends a lot of time driving my brother around to dance practice."

doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc_complete = [doc1, doc2, doc3]

doc_clean = [doc.split() for doc in doc_complete]
import gensim 

from gensim import corpora



# Creating the term dictionary of our corpus, where every unique term is assigned an index.  

dictionary = corpora.Dictionary(doc_clean)



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]



# Creating the object for LDA model using gensim library

Lda = gensim.models.ldamodel.LdaModel



# Running and Training LDA model on the document term matrix

ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)



# Results 

print(ldamodel.print_topics())