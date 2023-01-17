["2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19", "covid 19", "ncov-2019", "sars-cov-2", "wuhan coronavirus", "wuhan pneumonia", "wuhan virus"]
# Install scispacy package
!pip install scispacy
import spacy
import scispacy

nlp = spacy.load("../input/scispacymodels/en_core_sci_sm/en_core_sci_sm-0.2.4")
nlp.max_length = 2000000
!pip install contractions
import re

CURRENCIES = {'$': 'USD', 'zł': 'PLN', '£': 'GBP', '¥': 'JPY', '฿': 'THB',
              '₡': 'CRC', '₦': 'NGN', '₩': 'KRW', '₪': 'ILS', '₫': 'VND',
              '€': 'EUR', '₱': 'PHP', '₲': 'PYG', '₴': 'UAH', '₹': 'INR'}

RE_NUMBER = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))")

RE_URL = re.compile(
    r'((http://www\.|https://www\.|http://|https://)?' +
    r'[a-z0-9]+([\-.][a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(/.*)?)')

# English Stop Word List (Standard stop words used by Apache Lucene)
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
              "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
              "they", "this", "to", "was", "will", "with"}
import string
from typing import List
import ftfy
import contractions

def clean_tokenized_sentence(tokens: List[str],
                             unicode_normalization="NFC",
                             unpack_contractions=False,
                             replace_currency_symbols=False,
                             remove_punct=True,
                             remove_numbers=False,
                             lowercase=True,
                             remove_urls=True,
                             remove_stop_words=True) -> str:
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]

    sentence = ' '.join(tokens)

    if unicode_normalization:
        sentence = ftfy.fix_text(sentence, normalization=unicode_normalization)

    if unpack_contractions:
        sentence = contractions.fix(sentence, slang=False)

    if replace_currency_symbols:
        for currency_sign, currency_tok in CURRENCIES.items():
            sentence = sentence.replace(currency_sign, f'{currency_tok} ')

    if remove_urls:
        sentence = RE_URL.sub('_URL_', sentence)

    if remove_punct:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # strip double spaces
    sentence = re.sub(r' +', ' ', sentence)

    if remove_numbers:
        sentence = RE_NUMBER.sub('_NUMBER_', sentence)

    if lowercase:
        sentence = sentence.lower()

    return sentence

def clean_sentence(sentence) -> str:
    doc = nlp(sentence)
    tokens = [str(token) for token in doc]
    return clean_tokenized_sentence(tokens)
print(clean_sentence("Let's clean this sentence!"))
import pandas as pd
sentences_df = pd.read_csv('../input/covid19sentencesmetadata/sentences_with_metadata.csv')
sentences_df.head()
print(f"Sentence count: {len(sentences_df)}")
from gensim.models.phrases import Phraser
bigram_model = Phraser.load("../input/covid19phrasesmodels/covid_bigram_model_v0.pkl")
bigram_model["despite social media often vehicle fake news boast news hype also worth noting tremendous effort scientific community provide free uptodate information ongoing studies well critical evaluations".split()]
trigram_model = Phraser.load("../input/covid19phrasesmodels/covid_trigram_model_v0.pkl")
def clean_sentence(sentence, bigram_model=None, trigram_model=None) -> str:
    doc = nlp(sentence)
    tokens = [str(token) for token in doc]
    cleaned_sentence = clean_tokenized_sentence(tokens)
    
    if bigram_model and trigram_model:
        sentence_with_bigrams = bigram_model[cleaned_sentence.split(' ')]
        sentence_with_trigrams = trigram_model[sentence_with_bigrams]
        return ' '.join(sentence_with_trigrams)
    
    return cleaned_sentence
print(clean_sentence("On 23 January 2020, the Coalition for Epidemic Preparedness Innovations (CEPI) announced that they will fund vaccine development programmes with Inovio", bigram_model, trigram_model))
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from matplotlib import pylab
%matplotlib inline
fasttext_model_dir = '../input/fasttext-no-subwords-trigrams'
num_points = 400

first_line = True
index_to_word = []
with open(os.path.join(fasttext_model_dir, "word-vectors-100d.txt"),"r") as f:
    for line_num, line in enumerate(f):
        if first_line:
            dim = int(line.strip().split()[1])
            word_vecs = np.zeros((num_points, dim), dtype=float)
            first_line = False
            continue
        line = line.strip()
        word = line.split()[0]
        vec = word_vecs[line_num-1]
        for index, vec_val in enumerate(line.split()[1:]):
            vec[index] = float(vec_val)
        index_to_word.append(word)
        if line_num >= num_points:
            break
word_vecs = normalize(word_vecs, copy=False, return_norm=False)
tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
labels = index_to_word[:num_points]
def plot(embeddings, labels):
    pylab.figure(figsize=(20,20))
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()

plot(two_d_embeddings, labels)
from pprint import pprint
import gensim.models.keyedvectors as word2vec

fasttext_model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(fasttext_model_dir, "word-vectors-100d.txt"))
def print_most_similar(search_term):
    print(f"Synonyms of '{search_term}':")
    synonyms = fasttext_model.most_similar(search_term)
    pprint(synonyms)
print_most_similar("new_coronavirus")
print_most_similar("fake_news")
print_most_similar("pathogen")
[(0, '0.079"•" + 0.019"blood" + 0.015"associated" + 0.013"cells" + ' '0.012"ace2" + 0.012"protein" + 0.011"important" + 0.011"levels" + ' '0.010"diseases" + 0.010"cell"'), (1, '0.110"who" + 0.088"it" + 0.056"response" + 0.043"could" + 0.036"under" ' '+ 0.035"available" + 0.032"major" + 0.032"as" + 0.030"without" + ' '0.024"muscle"'), (2, '0.173"■" + 0.020"some" + 0.013"drugs" + 0.010"transmission" + ' '0.009"surgery" + 0.009"must" + 0.009"drug" + 0.009"there" + ' '0.008"increased" + 0.008"high"'), (3, '0.071"de" + 0.036"were" + 0.025"patient" + 0.023"1" + 0.022"after" + ' '0.018"a" + 0.018"more" + 0.015"all" + 0.015"when" + 0.014"cause"'), (4, '0.044"the" + 0.035"from" + 0.028"should" + 0.019"other" + 0.018"risk" ' '+ 0.017"oral" + 0.017"which" + 0.017"in" + 0.013"use" + 0.013"cases"'), (5, '0.069"may" + 0.033"can" + 0.031"have" + 0.029"disease" + 0.028"dental" ' '+ 0.022"also" + 0.020"has" + 0.020"been" + 0.018"health" + ' '0.016"virus"'), (6, '0.051"la" + 0.031"en" + 0.025"2" + 0.023"3" + 0.016"que" + 0.016"el" ' '+ 0.016"y" + 0.014"los" + 0.014"4" + 0.013"les"'), (7, '0.045"s" + 0.041"et" + 0.031"during" + 0.023"al" + 0.022"had" + ' '0.021"people" + 0.020"à" + 0.018"local" + 0.017"days" + 0.016"2020"'), (8, '0.062"patients" + 0.030"treatment" + 0.028"care" + 0.020"used" + ' '0.014"clinical" + 0.014"infection" + 0.013"common" + 0.013"severe" + ' '0.013"respiratory" + 0.012"dentistry"'), (9, '0.030"using" + 0.020"areas" + 0.018"ct" + 0.014"described" + ' '0.014"performed" + 0.013"lesions" + 0.013"above" + 0.012"day" + ' '0.011"learning" + 0.011"reactions"')]
def create_articles_metadata_mapping(sentences_df: pd.DataFrame) -> dict:
    sentence_id_to_metadata = {}
    for row_count, row in sentences_df.iterrows():
        sentence_id_to_metadata[row_count] = dict(
            paper_id=row['paper_id'],
            cord_uid=row['cord_uid'],
            source=row['source'],
            url=row['url'],
            publish_time=row['publish_time'],
            authors=row['authors'],
            section=row['section'],
            sentence=row['sentence'],
        )
    return sentence_id_to_metadata
sentence_id_to_metadata = create_articles_metadata_mapping(sentences_df)
import operator
from datetime import datetime

class SearchEngine:
    def __init__(self,
                 sentence_id_to_metadata: dict,
                 sentences_df: pd.DataFrame,
                 bigram_model,
                 trigram_model,
                 fasttext_model):
        self.sentence_id_to_metadata = sentence_id_to_metadata
        self.cleaned_sentences = sentences_df['cleaned_sentence'].tolist()
        print(f'Loaded {len(self.cleaned_sentences)} sentences')

        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.fasttext_model = fasttext_model

    def _get_search_terms(self, keywords, synonyms_threshold):
        # clean tokens
        cleaned_terms = [clean_tokenized_sentence(keyword.split(' ')) for keyword in keywords]
        # remove empty terms
        cleaned_terms = [term for term in cleaned_terms if term]
        # create bi-grams
        terms_with_bigrams = self.bigram_model[' '.join(cleaned_terms).split(' ')]
        # create tri-grams
        terms_with_trigrams = self.trigram_model[terms_with_bigrams]
        # expand query with synonyms
        search_terms = [self.fasttext_model.most_similar(token) for token in terms_with_trigrams]
        # filter synonyms above threshold (and flatten the list of lists)
        search_terms = [synonym[0] for synonyms in search_terms for synonym in synonyms
                        if synonym[1] >= synonyms_threshold]
        # expand keywords with synonyms
        search_terms = list(terms_with_trigrams) + search_terms
        return search_terms

    def search(self,
               keywords: List[str],
               optional_keywords=None,
               top_n: int = 10,
               synonyms_threshold=0.7,
               keyword_weight: float = 3.0,
               optional_keyword_weight: float = 0.5) -> List[dict]:
        if optional_keywords is None:
            optional_keywords = []

        search_terms = self._get_search_terms(keywords, synonyms_threshold)

        optional_search_terms = self._get_search_terms(optional_keywords, synonyms_threshold) \
            if optional_keywords else []

        print(f'Search terms after cleaning, bigrams, trigrams and synonym expansion: {search_terms}')
        print(f'Optional search terms after cleaning, bigrams, trigrams and synonym expansion: {optional_search_terms}')

        date_today = datetime.today()

        # calculate score for each sentence. Take only sentence with at least one match from the must-have keywords
        indexes = []
        match_counts = []
        days_diffs = []
        for sentence_index, sentence in enumerate(self.cleaned_sentences):
            sentence_tokens = sentence.split(' ')
            sentence_tokens_set = set(sentence_tokens)
            match_count = sum([keyword_weight if keyword in sentence_tokens_set else 0
                               for keyword in search_terms])
            if match_count > 0:
                indexes.append(sentence_index)
                if optional_search_terms:
                    match_count += sum([optional_keyword_weight if keyword in sentence_tokens_set else 0
                                       for keyword in optional_search_terms])
                match_counts.append(match_count)
                article_date = self.sentence_id_to_metadata[sentence_index]["publish_time"]

                if article_date == "2020":
                    article_date = "2020-01-01"

                article_date = datetime.strptime(article_date, "%Y-%m-%d")
                days_diff = (date_today - article_date).days
                days_diffs.append(days_diff)

        # the bigger the better
        match_counts = [float(match_count)/sum(match_counts) for match_count in match_counts]

        # the lesser the better
        days_diffs = [(max(days_diffs) - days_diff) for days_diff in days_diffs]
        days_diffs = [float(days_diff)/sum(days_diffs) for days_diff in days_diffs]

        index_to_score = {}
        for index, match_count, days_diff in zip(indexes, match_counts, days_diffs):
            index_to_score[index] = 0.7 * match_count + 0.3 * days_diff

        # sort by score descending
        sorted_indexes = sorted(index_to_score.items(), key=operator.itemgetter(1), reverse=True)

        # take only the sentence IDs
        sorted_indexes = [item[0] for item in sorted_indexes]

        # limit results
        sorted_indexes = sorted_indexes[0: min(top_n, len(sorted_indexes))]

        # get metadata for each sentence
        results = []
        for index in sorted_indexes:
            results.append(self.sentence_id_to_metadata[index])
        return results
search_engine = SearchEngine(sentence_id_to_metadata, sentences_df, bigram_model, trigram_model, fasttext_model)
def search(keywords, optional_keywords=None, top_n=10, synonyms_threshold=0.8, only_sentences=False):
    print(f"\nSearch for terms {keywords}\n\n")
    results = search_engine.search(
        keywords, optional_keywords=optional_keywords, top_n=top_n, synonyms_threshold=synonyms_threshold
    )
    print("\nResults:\n")
    
    if only_sentences:
        for result in results:
            print(result['sentence'] + "\n")
    else:
        pprint(results)
search(keywords=["spillover", "bats", "snakes", "exotic animals", "seafood"],
       optional_keywords=["new coronavirus", "coronavirus", "covid19"],
      top_n=3)
task_id = 10
import json

with open(f"../input/covid19seedsentences/{task_id}.json") as in_fp:
    seed_sentences_json = json.load(in_fp)

print(seed_sentences_json['taskName'])
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# This will take time on the first time since it downloads the model
tokenizer_summarize = BartTokenizer.from_pretrained('bart-large-cnn')
model_summarize = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
class BartSummarizer:
    def __init__(self, tokenizer_summarize, model_summarize):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer_summarize = tokenizer_summarize
        self.model_summarize = model_summarize
        self.model_summarize.to(self.torch_device)
        self.model_summarize.eval()

    def create_summary(self, text: str,
                       repetition_penalty=1.0) -> str:
        text_input_ids = self.tokenizer_summarize.batch_encode_plus(
            [text], return_tensors='pt', max_length=1024)['input_ids']
        
        min_length = min(text_input_ids.size()[1], 128)
        max_length = min(text_input_ids.size()[1], 1024)

        print(f"Summary Min length: {min_length}")
        print(f"Summary Max length: {max_length}")

        text_input_ids = text_input_ids.to(self.torch_device)

        summary_ids = self.model_summarize.generate(text_input_ids,
                                                    num_beams=4,
                                                    length_penalty=1.4,
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    no_repeat_ngram_size=4,
                                                    repetition_penalty=repetition_penalty)
        summary = self.tokenizer_summarize.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summary
bart_summarizer = BartSummarizer(tokenizer_summarize, model_summarize)
with open(f"../input/covid19seedsentences/{task_id}_relevant_sentences.json") as in_fp:
    relevant_sentences_json = json.load(in_fp)
answers_results = []
for idx, sub_task_json in enumerate(relevant_sentences_json["subTasks"]):
    sub_task_description = sub_task_json["sub_task_description"]
    print(f"Working on task: {sub_task_description}")
    best_sentences = seed_sentences_json["subTasks"][idx]["bestSentences"]
    relevant_sentences = sub_task_json["relevant_sentences"]
    relevant_sentences_texts = [result["sentence"] for result in relevant_sentences]
    sub_task_summary = bart_summarizer.create_summary(" ".join(best_sentences + relevant_sentences_texts))
    answers_results.append(dict(sub_task_description=sub_task_description, relevant_sentences=relevant_sentences, sub_task_summary=sub_task_summary))
from IPython.display import display, HTML
pd.set_option('display.max_colwidth', 0)
def display_summary(summary: str):
    return display(HTML(f"<div>{summary}</div>"))

def display_sub_task_description(sub_task_description):
    return display(HTML(f"<h2>{sub_task_description}</h2>"))

def display_task_name(task_name):
    return display(HTML(f"<h1>{task_name}</h1>"))
def visualize_output(sub_task_json):
    """
    Prints output for each sub-task
    """
    # print description
    display_sub_task_description(sub_task_json.get("sub_task_description"))
    display_summary(sub_task_json.get("sub_task_summary"))

    # print output sentences
    results = sub_task_json.get('relevant_sentences')
    sentence_output = pd.DataFrame(sub_task_json.get('relevant_sentences'))
    sentence_output.rename(columns={"sentence": "Relevant Sentence","cord_id": "CORD UID",
                                    "publish_time": "Publish Time", "url": "URL",
                                    "source": "Source"}, inplace=True)
    
    display(HTML(sentence_output[['cord_uid', 'Source', 'Publish Time', 'Relevant Sentence', 'URL']].to_html(render_links=True, escape=False)))
display_task_name(seed_sentences_json["taskName"])
for sub_task_json in answers_results:
    visualize_output(sub_task_json)
def save_output(seed_sentences, sub_task_json):
    """
    Saves output for each sub-task
    """
    sentence_output = pd.DataFrame(sub_task_json.get('relevant_sentences'))
    sentence_output.rename(columns={"sentence": "Relevant Sentence","cord_id": "CORD UID",
                                    "publish_time": "Publish Time", "url": "URL",
                                    "source": "Source"}, inplace=True)
    
    return sentence_output[['cord_uid', 'Source', 'Publish Time', 'Relevant Sentence', 'URL']]
relevant_sentences = []
for idx, sub_task_json in enumerate(answers_results):
    task_sentences = save_output(seed_sentences_json["subTasks"][idx]["bestSentences"], sub_task_json)
    relevant_sentences.append(task_sentences)
all_relevant_sentences = pd.concat(relevant_sentences).reset_index()
all_relevant_sentences.head(1)
cleaned_sentences = []
for i in range(len(all_relevant_sentences['Relevant Sentence'])):
    cleaned_sentences.append(clean_sentence(all_relevant_sentences['Relevant Sentence'][i], bigram_model, trigram_model).split())
cleaned_sentences[0]
from gensim import corpora

# Create Dictionary
id2word = corpora.Dictionary(cleaned_sentences)

# Create Corpus
texts = cleaned_sentences

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
import gensim

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# Print the Keyword in the topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]