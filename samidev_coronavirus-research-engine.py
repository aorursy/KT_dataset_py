!pip install rank_bm25 nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.cluster.util import cosine_distance
from nltk.tokenize import word_tokenize, sent_tokenize

import json
import re
import os
# cleans the text and section property of the article
def getText(x):
    d = []
    for t in x:
        section = t['section'].lower().strip()
        section = re.sub(' +', ' ', section)
        
        if section.isnumeric():
            section = ''
        
        if len(section) > 2:
            section = section[0].upper() + section[1:]
            
        text = t['text']
        text = cleanText(text)

        d.append({ 'text': text, 'section': section })
    return d


# regex to remove characters and extra spaces
def cleanText(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(' +', ' ', text)
    t = ''

    sentences = text.split('. ')
    for i in range(len(sentences)):
        sent = sentences[i]
        sent = sent.strip()
        if len(sent) >= 10:
            if sent[-1] == ' ':
                sent = sent[:-1]
            sent += '. '
            t += sent
            
    return t


def lastElement(i, arr):
    return i == len(arr) - 1


def getAuthors(metadata):
    authors = []
    for author in metadata['authors']:
        name = author['first'] + ' ' + author['last']
        name = name.lstrip('[').rstrip(']').strip()
        if len(name) >= 3:
            authors.append(name)

    return authors
# filters out junk data and returns a list of unique words
def preProcess(paragraph, article_id=False, tokenized_words=False):
    
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    
    # Remove special characters
    paragraph = re.sub('\(|\)|:|,|;|’|”|“|\?|%|>|<', '', paragraph)
    paragraph = re.sub('/', ' ', paragraph)
    paragraph = paragraph.replace("'",'')

    # Tokenize paragraph
    paragraph = word_tokenize(paragraph.lower())
    
    words = []
    
    # filter out stop words
    for word in paragraph:
        if (word not in stop_words and not tokenized_words) or (word not in stop_words and tokenized_words and word not in tokenized_words) and word not in words:
            if len(word.strip()) > 2:
                words.append(word)
                
    if article_id and not tokenized_words:
        words.insert(0, article_id)
        
    return words

    # Reducing words to their root form
    #stemmer = SnowballStemmer("english")
    # words = [stemmer.stem(word) for word in words]
    
    # tokenizing the sentences
    #sentences = sent_tokenize(text)

def get_frequency(article) -> dict:

    # Removing stop words
    stop_words = set(stopwords.words("english"))
    
    tokenized_words = []
    tokenized_sentences = []
    
    for para in article['body'][:3]:
        text = para['text']
        sent = sent_tokenize(text)
        
        # tokenizing the sentences
        if sent not in tokenized_sentences:
            tokenized_sentences.extend(sent)
            
        tokenized_words.extend(preProcess(text, False, tokenized_words))
        
    # Reducing words to their root form
    stem = PorterStemmer()

    # Creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in tokenized_words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table, tokenized_sentences


def calculate_sentence_scores(sentences, frequency_table) -> dict:

    # Algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]
    try:
        sentence_weight[sentence[:7]] = round(sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words, 5)
    except:
        return sentence_weight
    
    return sentence_weight


def calculate_average_score(sentence_weight) -> int:

    # Calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    # Getting sentence average value from source text
    try:
        average_score = (sum_values / len(sentence_weight))
    except:
        return 0

    return average_score


def produce_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary
def getArticleSummary(article):

    # creating a dictionary for the word frequency table and tokenized sentences
    frequency_table, sentences = get_frequency(article)
    
    # algorithm for scoring a sentence by its words
    sentence_scores = calculate_sentence_scores(sentences, frequency_table)

    # getting the threshold
    threshold = calculate_average_score(sentence_scores)
    
    # producing the summary
    article_summary = produce_article_summary(sentences, sentence_scores, 1.5 * threshold)

    if len(article_summary.split(' ')) > 5:
        if article_summary[0] == ' ':
            article_summary = article_summary[1:]
        return article_summary        
    else:
        return False

dirname = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
articles = []
    
# Iterate over all JSON files and capture the title, id, authors, abstract, body and produce a sumamry
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.split('.')[-1] == 'json':
            
            # Convert to JSON
            path = os.path.join(dirname, filename)
            d = json.load(open(path, 'rb'))
            metadata = d['metadata']

            # Get relevant data
            title = metadata['title'].strip()

            if len(title) < 3:
                continue

            authors = getAuthors(metadata)
            abstract = []
            
            if 'abstract' in d:
                abstract = d['abstract']
                
            body = d['body_text']

            # Clean data
            abstract = getText(abstract)
            body = getText(body)

            data = {
                'paper_id': d['paper_id'],
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'body': body,
                'summary': ''
            }
            
            # Get article summary 
            article_summary = getArticleSummary(data)
            
            if article_summary:
                data['summary'] = article_summary
                
            articles.append(data)
            
            # NOTE: It takes way too long to run through the entire dataset, we will only capture and summarise 1000 papers
            if len(articles) > 1000:
                break
len(articles)
def tokenizeArticles(articles):
    tokenized_corpus = []

    for article in articles:
        article_id = article['paper_id']
        abstract = article['abstract']
        body = article['body']
        title = article['title']

        paragraph = title + ' '

        if len(abstract) > 0:
            paragraph += abstract[0]['text']
        elif len(body) > 0:
            paragraph += body[0]['text']

        tokenized_words = preProcess(paragraph, article_id)
        tokenized_corpus.append(tokenized_words)
        
    return tokenized_corpus
def buildIndex(tokenized_corpus):
    return BM25Okapi(tokenized_corpus)
def searchArticles(bm25, tokenized_corpus, articles, query):
    query = query.split(' ')
    
    doc_scores = bm25.get_scores(query)
    results = bm25.get_top_n(query, tokenized_corpus, n=15)

    return getArticles(articles, results)

def getArticles(articles, results):
    result_articles = []

    for words in results:
        article_id = words[0]
        sentences = ' '.join(words[1:])

        for article in articles:
            if article_id == article['paper_id']:
                result_articles.append(article)
                break

    return result_articles
# Tokenize articles by words
tokenized_corpus = tokenizeArticles(articles)
# Build the index using BM25 Search algorithm
bm25_index = buildIndex(tokenized_corpus)
query = 'COVID 19 risk factors'
found_articles = searchArticles(bm25_index, tokenized_corpus, articles, query)
for article in found_articles:
    print(f'Title: {article["title"]}\n')
    
    if len(article['summary']):
        print(f'Summary: {article["summary"]}\n\n')
    elif len(article['abstract']):
        print(f'Abstract: {article["abstract"][0]["text"]}\n\n')

tasks = [
    {
        'name': 'What is known about transmission, incubation and environmental stability?',
        'questions': [
            'incubation periods for the disease in humans',
            'Prevalence of asymptomatic shedding and transmission'
            'Natural history of the virus',
            'Implementation of diagnostics and products to improve clinical processes',
            'Immune response and immunity',
            'Role of the environment in transmission'
        ]
    },
    {
        'name': 'What do we know about COVID-19 risk factors?',
        'questions': [
            'COVID 19 risk factors',
            'Smoking, pre-existing pulmonary disease',
            'Co-infections and other co-morbidities',
            'Neonates and pregnant women',
            'Socio-economic and behavioral factors',
            'Transmission dynamics of the virus serial interval, modes of transmission and environmental factors',
            'risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',
            'Susceptibility of populations',
            'Public health mitigation measures'
        ]
    },
    {
        'name': 'What do we know about virus genetics, origin, and evolution?',
        'questions': [
            'Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination',
            'geographic and temporal diverse sample sets',
            'field surveillance, genetic sequencing, receptor binding',
            'Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia',
            'Experimental infections to test host range for this pathogen',
            'Animal host(s) and any evidence of continued spill-over to humans',
            'Socioeconomic and behavioral risk factors for this spill-over',
            'Sustainable risk reduction strategies'
        ]
    },
    {
        'name': 'What do we know about vaccines and therapeutics?',
        'questions': [
            'Effectiveness of drugs being developed and tried to treat COVID-19 patients.',
            'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients',
            'Exploration of use of best animal models and their predictive value for a human vaccine',
            'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents',
            'Alternative models to aid decision makers',
            'Efforts targeted at a universal coronavirus vaccine',
            'Approaches to evaluate risk for enhanced disease after vaccination'
        ]
    }
]
for task in tasks:
    print(f'Task: {task["name"]}\n')
    for question in task['questions']:
        found_articles = searchArticles(bm25_index, tokenized_corpus, articles, question)
        print(f'\nQuestion: {question}\n')
        for article in found_articles[:3]:
            print(f'Title: {article["title"]}\n')
            if len(article['summary']):
                print(f'Summary: {article["summary"]}')
            elif len(article['abstract']):
                print(f'Abstract: {article["abstract"][0]["text"]}')

            print()
    print()
print('Search for a query, seperate words by a space')

while True:
    try:
        query = input('Search: ')

        # Query the index and to get a list of articles a long with a summary
        found_articles = searchArticles(bm25_index, tokenized_corpus, articles, query)

        print(f'\n{len(found_articles)} Articles found:')

        for article in found_articles[:3]:
            print(f'\t- Title: {article["title"]}')
            if article['summary']:
                print(f'\t- Summary: {article["summary"]}')
            elif len(article['abstract']):
                print(f'\t- Abstract: {article["abstract"][0]["text"]}')
                print()

        print()
        
    except:
        break