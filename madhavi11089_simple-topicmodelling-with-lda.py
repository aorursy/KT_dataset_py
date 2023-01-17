# considering below some sample of documents 

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."

doc2 = "Doctors, dietitians, and other health experts believe that this is because sugary foods are worse at providing satiety, or a sense of fullness."

doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."

doc5 = "A small study in the Journal of Hypertension found that consuming high levels of salt could have an immediate impact on the proper functioning of a person’s blood vessels. Excess sodium intake also has links to fluid retention."

doc6= "We found that dancers and musicians differed in many white matter regions, including sensory and motor pathways, both at the primary and higher cognitive levels of processing,'lead author Chiara Giacosa."



# compile documents

docs= [doc1, doc2, doc3, doc4, doc5, doc6]
# perform some cleaning in the documents

import nltk

nltk.download('stopwords')

nltk.download('punkt')

from nltk.stem.wordnet import WordNetLemmatizer

import string

from string import punctuation

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



stopwords=stopwords.words('english') + list(punctuation)



lemma=WordNetLemmatizer()



def cleaning_text(documents):   # documents is list of list of tokens

    stops_free=[[i for i in text if i not in stopwords] for text in documents ]

    normalized=[[lemma.lemmatize(word) for word in text] for text in stops_free]

    

    return normalized





# lowering the input doc

docs_lower=[doc.lower() for doc in docs]

docs_tokens=[word_tokenize(doc) for doc in docs_lower]  # create a list of list of tokens of text

print(docs_tokens)

final_docs=cleaning_text(docs_tokens)

print('final normalized documents:',final_docs)
import gensim

from gensim import corpora



#creating the term dictionary of the corpus,where each unique word is assigned with unique integer value ,refer as index.

dictionary = corpora.Dictionary(final_docs)



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.



doc2term_matrix=[dictionary.doc2bow(doc) for doc in final_docs]
dictionary.token2id
doc2term_matrix
# training LDA Model

Lda=gensim.models.ldamodel .LdaModel



ldamodel=Lda(doc2term_matrix,num_topics=3,id2word=dictionary,passes=50)

#results

print(ldamodel.print_topics(num_topics=3,num_words=3))
import pyLDAvis

import pyLDAvis.gensim 



pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(ldamodel, doc2term_matrix, dictionary)

vis