import glob

import nltk

import numpy as np

import os

import pandas as pd

import random

import seaborn as sns

import string
english_words = nltk.corpus.words.words()
len(english_words)
english_words[:10]
lower_english_words = {word.lower() for word in english_words}
len(lower_english_words)
len(english_words) - len(lower_english_words)
punctuation_tokens = set(string.punctuation)
OOV = '<oov>'
non_word_tokens = punctuation_tokens.union({OOV})
indexed_base_vocabulary = sorted(list(lower_english_words.union(non_word_tokens)))
base_vocabulary = {indexed_base_vocabulary[i]:i for i in range(len(indexed_base_vocabulary))}
assert len(indexed_base_vocabulary) == len(base_vocabulary)



for i in range(len(indexed_base_vocabulary)):

    assert base_vocabulary[indexed_base_vocabulary[i]] == i



print('All good!')
def extract_content(path):

    with open(path, 'r') as book:

        content = book.read()

    return content
def remove_gutenberg_text(content):

    paragraphs = (p for p in content.split('\n') if p != '')

    include = False

    START_PREFIX = '***START OF'

    END_PREFIX = '***END OF'

    

    non_gutenberg_paragraphs = []

    

    for paragraph in paragraphs:

        if paragraph[:len(END_PREFIX)] == END_PREFIX:

            include = False

        

        if include:

            non_gutenberg_paragraphs.append(paragraph)

        

        if paragraph[:len(START_PREFIX)] == START_PREFIX:

            include = True

    

    return '\n'.join(non_gutenberg_paragraphs)
def extract_sentences(book):

    return nltk.tokenize.sent_tokenize(book)
def tokenize(sentence):

    return nltk.tokenize.word_tokenize(sentence)
def lookup_token(token, vocabulary):

    result = vocabulary.get(token.lower())

    if result is None:

        result = vocabulary[OOV]

    return result
def lookup_index(index, indexed_vocabulary):

    return indexed_vocabulary[index]
def encode_document(document, vocabulary):

    return [lookup_token(token, vocabulary) for token in document]
def decode_document(encoded_document, indexed_vocabulary):

    return [lookup_index(index, indexed_vocabulary) for index in encoded_document]
def calculate_frequencies(encoded_document, vocabulary):

    return np.bincount(encoded_document, minlength=len(vocabulary))
DARWIN_DIR = '../input/darwin/darwin'

DICKENS_DIR = '../input/dickens/dickens'
pd.read_csv(os.path.join(DARWIN_DIR, 'metadata.tsv'), delimiter='\t')
pd.read_csv(os.path.join(DICKENS_DIR, 'metadata.tsv'), delimiter='\t')
def build_corpora(vocabulary):

    directories = [('darwin', DARWIN_DIR), ('dickens', DICKENS_DIR)]

    

    book_paths = list(map(

        lambda p: (p[0], glob.glob(os.path.join(p[1], '*.txt'))),

        directories

    ))

    

    books = list(map(

        lambda p: (

            p[0],

            (remove_gutenberg_text(extract_content(path)) for path in p[1])

        ),

        book_paths

    ))

    

    sentences = map(

        lambda p: (

            p[0],

            (sentence for book in p[1] for sentence in extract_sentences(book))

        ),

        books

    )

    

    documents = map(

        lambda p: (

            p[0],

            (tokenize(sentence) for sentence in p[1])

        ),

        sentences

    )

    

    encoded_sentences = list(map(

        lambda p: (

            p[0],

            [encode_document(document, vocabulary) for document in p[1]]

        ),

        documents

    ))

    

    return dict(encoded_sentences)
corpora = build_corpora(base_vocabulary)
encoded_darwin_sentence = corpora['darwin'][42]
print(encoded_darwin_sentence)
decoded_darwin_sentence = decode_document(encoded_darwin_sentence, indexed_base_vocabulary)
print(decoded_darwin_sentence)
print({author:len(corpora[author]) for author in corpora})
eval_pool_sizes = {'darwin': 5000, 'dickens': 15000}
eval_pool_indices = {

    k:random.sample(range(len(corpora[k])), eval_pool_sizes[k])

    for k in corpora}
eval_pool_index_sets = {k:set(eval_pool_indices[k]) for k in eval_pool_indices}
train_pool_indices = {

    k:[i for i in range(len(corpora[k])) if i not in eval_pool_index_sets[k]]

    for k in corpora

}
for k in corpora:

    assert len(train_pool_indices[k]) + len(eval_pool_indices[k]) == len(corpora[k])
EVAL_SET_SIZE = 100
eval_set_indices = {

    k:[eval_pool_indices[k][i*EVAL_SET_SIZE:(i*EVAL_SET_SIZE)+EVAL_SET_SIZE]

       for i in range(int(eval_pool_sizes[k]/EVAL_SET_SIZE))]

    for k in eval_pool_indices

}
def generate_language_model(corpus, vocabulary, mu=0.01):

    current = np.zeros((len(vocabulary),))

    for doc in corpus:

        current += calculate_frequencies(doc, vocabulary)

    trained = current/current.sum()

    

    baseline = np.ones((len(vocabulary),))/len(vocabulary)

    

    return mu*baseline + (1 - mu)*trained
language_models = {

    k:generate_language_model(

        (corpora[k][i] for i in train_pool_indices[k]),

        base_vocabulary

    ) for k in corpora

}
def model_likelihood(model, doc):

    return np.asarray([model[token_index] for token_index in doc]).prod()
def generate_classifier(language_models):

    def classify(doc):

        model_likelihoods = {

            k:model_likelihood(language_models[k], doc) for k in language_models

        }

        

        denominator = sum(model_likelihoods[k] for k in model_likelihoods)

        

        if denominator == 0:

            classification = {k:0.5 for k in model_likelihoods}

        else:

            classification = {k:model_likelihoods[k]/denominator for k in model_likelihoods}

        

        return classification

    

    return classify
document_classifier = generate_classifier(language_models)
def evaluate_classifier_quality(classifier, eval_data):

    results = {}

    for k in eval_data:

        i = 0

        results[k] = []

        for index_set in eval_data[k]:

            i += 1

            individual_accuracies = [

                classifier(corpora[k][i])[k] for i in index_set

            ]

            average_accuracy = np.asarray(individual_accuracies).mean()

            results[k].append(average_accuracy)

    

    return results
evaluations = evaluate_classifier_quality(document_classifier, eval_set_indices)
evaluation_tuples = [

    (p, 1-p, 'darwin') for p in evaluations['darwin']

] + [

    (1-p, p, 'dickens') for p in evaluations['dickens']

]
evaluation_df = pd.DataFrame.from_records(

    evaluation_tuples,

    columns=('p_darwin', 'p_dickens', 'label')

)
sns.violinplot(data=evaluation_df, x='label', y='p_darwin')