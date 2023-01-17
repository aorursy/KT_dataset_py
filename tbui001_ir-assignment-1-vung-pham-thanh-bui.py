import bz2

from nltk.corpus import stopwords,words

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

from string import punctuation

stemmer = SnowballStemmer('english')



def process_sentence(sentence, tokens_to_remove, English_words, max_tokens=None):

    words = word_tokenize(sentence) # Tokenize

    if max_tokens is not None and len(words) < max_tokens:

        return None

    else:

        words = [w.lower() for w in words if not w.isdigit()] # Convert to lowercase and also remove digits

        filter_words = [stemmer.stem(w) for w in words if w not in tokens_to_remove and w in English_words] # remove tokens + check english words + stem

        return filter_words

%%time



max_docs = None # test with this number of docs first. If would like to do for all docs, set this value to None

filename = bz2.open('../input/amazonreviews/test.ft.txt.bz2','rt',encoding='utf-8')

corpusfile = 'corpus_text.txt'



stop_words = set(stopwords.words('english'))

labels = ['__label__1','__label__2']

tokens_to_remove = stop_words.union(set(punctuation)).union(set(labels))

English_words = set(words.words())





doc_count = 0

with bz2.open('../input/amazonreviews/test.ft.txt.bz2','rt',encoding='utf-8') as inputfile:

    with open(corpusfile, 'w') as outputfile:

        for line in inputfile:

            filter_words = process_sentence(line, tokens_to_remove, English_words, 100)

            if filter_words is not None:

                outputfile.write(f'{" ".join(filter_words)}\n') # write the results

                doc_count += 1

                if  max_docs and doc_count >= max_docs: # if we do define the max_docs

                    break
# View the file if needed

from IPython.display import FileLink

FileLink('corpus_text.txt')
%%time

# Give some summaries of the text.

vocab = set()

with open('corpus_text.txt') as corpus_text:

    documents = corpus_text.readlines()

    doc_count = len(documents)

    for doc in documents:

        words = word_tokenize(doc)

        vocab = vocab.union(words)

print(f'Number of documents {doc_count}, vocabulary size {len(vocab)}')
import wordcloud

import matplotlib.pyplot as plt

def show_wordcloud(text, title=None):

    # Create and generate a word cloud image:

    wc = wordcloud.WordCloud(background_color='white').generate(text)

    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    if title is not None:

        plt.title(title)

    plt.show()

    

def show_wordcloud_for_doc(docIdx, title=None):

    show_wordcloud(documents[docIdx], title)
# Create and generate a word cloud image:

show_wordcloud(" ".join(documents), 'WordCloud for the whole corpus')
import numpy as np

# Generate word cloud for a random document

docIdx = np.random.randint(doc_count)

show_wordcloud_for_doc(docIdx, f'Word Cloud for document index {docIdx}')
from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords, words

stemmer = SnowballStemmer('english')



stop_words = set(stopwords.words('english'))

labels = ['__label__1','__label__2']

tokens_to_remove = stop_words.union(set(punctuation)).union(set(labels))

English_words = set(words.words())





query = 'game love music book fun good bad product money waste'



tf = {} # a dictionary of terms in the query (key is the term in the query and value is an array of term frequencies in each of the document)

df = {} # a dictionary of terms in the query (key is the term in the query and value is the document appearance)

wtf = {} # weighted term frequency using log

idf = {} # inverse document frequency

tfidf = {} # the tf-idf score for each term for each document



query_terms = process_sentence(query, tokens_to_remove, English_words)

# Initializing

for term in query_terms:

    tf[term] = [0 for _ in range(doc_count)]

    df[term] = 0

    wtf[term] = [0 for _ in range(doc_count)]

    idf[term] = 0

    tfidf[term] = [0 for _ in range(doc_count)]
%%time

for docIdx, doc in enumerate(documents):

    words = word_tokenize(doc) # they were cleaned so we don't have to clean any more

    # update term frequency counts

    for word in words:

        if word in query_terms:

            tf[word][docIdx] += 1 # increase term frequency count for the document

    # update doc frequency counts

    for term in query_terms:

        if term in words: # if the term is inside the doc

            df[term] += 1                
# Now calculate the wtf and the idf

import math

for term in query_terms:

    wtf[term] = [1+math.log10(tf_val) if tf_val != 0 else 0 for tf_val in tf[term]] # weighted term frequency as 1 + log10 of tf

    idf[term] = math.log10(doc_count/df[term]) # invert document frequency as N/df

    tfidf[term] = [wtf_val * idf[term] for wtf_val in wtf[term]] # tf idf is the weighted term frequency * inverse document frequency
%%time

tfidfdoc = []

for docIdx, doc in enumerate(documents):

    words = word_tokenize(doc) # they were cleaned so we don't have to clean any more

    # initialize as zero

    tfidfdoc.append(0)

    for term in query_terms:

        if term in words:

            tfidfdoc[docIdx] += tfidf[term][docIdx]
import numpy as np

import matplotlib.pyplot as plt
for term in query_terms:

    max_count = np.max(tf[term])

    bins =range(0, max_count+1)

    plt.figure()

    plt.hist(tf[term], label=term, bins = bins)

    plt.title(f'Term frequency for {term}')
for term in query_terms:

    plt.figure()

    plt.hist(wtf[term], label=term)

    plt.title(f'Weighted term frequency for {term}')
plt.figure()

_ = plt.bar(query_terms, [df[term] for term in query_terms])

_ = plt.title("Document frequencies for the terms")
plt.figure()

_ = plt.bar(query_terms, [idf[term] for term in query_terms])

_ = plt.title("Inverse document frequencies for the terms")
_ = plt.hist(tfidfdoc)

_ = plt.title("TF-IDF for the query")
for docIdx in np.argsort(-np.array(tfidfdoc))[0:10]:

    show_wordcloud_for_doc(docIdx, f'WordCloud for document index {docIdx}')
# Initialization (docs, query_terms)

tfidfvecspace = np.zeros((doc_count, len(query_terms)))



# Assign the tfidf for each term for each document

for docIdx in range(doc_count): # for each document

    for termIdx, term in enumerate(query_terms): # for each term

        tfidfvecspace[docIdx][termIdx] = tfidf[term][docIdx]
%%time

# normalize the vectors

tfidfvecspace = np.array([vec/np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec for vec in tfidfvecspace])



# due to very large number of documents in our corpus, we will do pairwise comparison of the first few number documents 

doc_limit = 10000

# for every pair

pairwise_comparison = {}

for docIdx1 in range(doc_limit-1):

    vec1 = tfidfvecspace[docIdx1]

    for docIdx2 in range(docIdx1 + 1, doc_limit):

        vec2 = tfidfvecspace[docIdx2]

        key = f'{docIdx1}-{docIdx2}'

        pairwise_comparison[key] = vec1.dot(vec2)
plt.figure()

plt.hist(list(pairwise_comparison.values()))

_ = plt.plot()
# sort the comparison by their similarities

pairwise_comparison = {k: v for k, v in sorted(pairwise_comparison.items(), key=lambda item: item[1], reverse=True)}
iterator = iter(pairwise_comparison.items())

for i in range(10):

    nextItem = next(iterator)

    docIdxs = [int(docIdx) for docIdx in nextItem[0].split('-')]

    doc0 = docIdxs[0]

    doc1 = docIdxs[1]

    print(f'Two similar documents {doc0}, {doc1} with similarity score of {nextItem[1]}, with two vectors {tfidfvecspace[doc0]} vs. {tfidfvecspace[doc1]}')

    show_wordcloud_for_doc(docIdxs[0], f'WordCloud for document index {docIdxs[0]}')

    show_wordcloud_for_doc(docIdxs[1], f'WordCloud for document index {docIdxs[1]}')