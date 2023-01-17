!pip install sparse-dot-topn
import math

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sparse_dot_topn import awesome_cossim_topn
vendor_names = [

    'AUTOMATED DOOR WAYS',

    'AUTOMATED DOOR WAYS, INC',

    'AUTOMATED DOORS & ACCESS, INC.',

    'BEACON ELECTRICAL LLC',

    'BEACON FEDERAL',

    'BEACON HEALTH',

    'BEACON HEALTH LLC',

    'CEDAR CONSULTANTS LLC',

    'CEDAR CREEK MEAT MARKET',

    'CEDAR GATE TECHNOLOGIES INC',

    'CEDAR GATE TECHNOLOGIES INC ONE'

]
spend_vendor_name = 'CEDAR GATE TECHNOLOGIES'
# Convert a collection of text documents to a matrix of token counts

# a document in this case is a vendor name

stop_words = frozenset(['ltd', 'llc', 'inc', 'llp'])

count_vectorizer = CountVectorizer(stop_words=stop_words)



# Learn a vocabulary dictionary of all tokens in the raw documents.

vocabulary = count_vectorizer.fit(vendor_names + [spend_vendor_name]).vocabulary_



print(vocabulary)
# Convert a collection of raw documents to a matrix of TF-IDF features.

tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)



# Learn vocabulary and idf, return term-document matrix.

tfidf_spend_vendor = tfidf_vectorizer.fit_transform([spend_vendor_name])



print(tfidf_spend_vendor)
# this needs to be transposed before multiplying to achieve Cosine Similarity

tfidf_vendor = tfidf_vectorizer.fit_transform(vendor_names).transpose()



print(tfidf_vendor)
results = awesome_cossim_topn(tfidf_spend_vendor, tfidf_vendor, 5, 0)



print(results)
print(spend_vendor_name)

print('-------------------')



for index, i in enumerate(results.indices):

    print('{}: {}'.format(vendor_names[i], results.data[index]))