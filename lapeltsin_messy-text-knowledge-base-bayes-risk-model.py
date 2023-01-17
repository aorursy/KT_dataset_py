import pandas as pd



df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', low_memory=False)

record_count = df.shape[0]

print(f"Our metadata includes {record_count} total records.")

df.head(3)
df = df.dropna(subset=['abstract'])

percent_abstracts = 100 * df.shape[0] / (record_count)

print(f"{percent_abstracts:.2f}% of our records contain abstracts.")
df[['title', 'abstract']].describe(include=['O'])
df.drop_duplicates(subset='abstract', keep="first", inplace=True)

percent_remaining = 100 * df.shape[0] / (record_count)

print(f"{percent_remaining:.2f}% of our records remain after duplicate deletion.")
import re

from functools import partial





def compile_pattern_list(pattern_list, add_bounderies=True, escape=True):

    """This function compiles an aggregated regular expression from a list of string

    patterns. It is frequently used elsewhere within this notebook.

    

    Parameters

    ----------

    pattern_list: [str]:

        A list of strings to be OR-ed together into a single regular expression.

    add_bounderies (optional): Bool

        If True, then word boundaries are added to the regular expression.

    escape (optional): Bool

        If True, then we run `re.escape` on each pattern string, to ensure proper compilation.

        

    Returns

    -------

    regex: re.Pattern

        A compiled regular expression object.

    """

    if escape:

        pattern_list = [re.escape(p) for p in pattern_list]

    

    regex_string = r'|'.join(pattern_list)

    if add_bounderies:

        regex_string = r'\b%s\b' % regex_string

    

    return compile_(regex_string)



# Compiles a case-insenstiive regular expression. Re-used frequently elsewhere in the Notebook.

compile_ = partial(re.compile, flags=re.I)



pattern_list = ['SARS', 'Severe Acute Respiratory Syndrome', 'COVID-19', 'SARS-COV', 

                'coronavirus', 'corona virus', '2019-nCoV', 'SARS-CoV-2']



regex = compile_pattern_list(pattern_list)

is_match = [any([regex.search(e) for e in [title, abstract]])

            for title, abstract in df[['title', 'abstract']].values]

df = df[is_match]



print(f"{df.shape[0]} of our records reference either SARS or some coronavirus.")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

# This matrix has been normalized under default settings

tfidf_matrix = vectorizer.fit_transform(df.abstract)
import numpy as np

from sklearn.decomposition import TruncatedSVD

# Truncated SVD is a stochastic algorithm. We set the random seed to ensure a consistant output.

np.random.seed(0)

lsa_matrix = TruncatedSVD(n_components=100).fit_transform(tfidf_matrix)
np.random.seed(0)

from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize



clusters = KMeans(n_clusters=20).fit_predict(normalize(lsa_matrix))

df['Index'] = range(clusters.size)

df['Cluster'] = clusters

# Clusters are stored as DataFrames for easier analysis.

cluster_groups = [df_cluster for  _, df_cluster in df.groupby('Cluster')]
np.random.seed(0)

import matplotlib.pyplot as plt

from wordcloud import WordCloud





def cluster_to_image(df_cluster, max_words=10, tfidf_matrix=tfidf_matrix,

                     vectorizer=vectorizer):

    """This function converts a text-cluster into a word-cloud image.



    Parameters

    ----------

    df_cluster: DataFrame

        This DataFrame contains the document ids associated with a single cluster.

    max_words (optional): int

        The number of top-ranked words to include in the word-cloud image.

    tfidf_matrix (optional): csr_matrix

        A matrix of TFIDF values. Each row i corresponds to the ith abstract.

    vectorizer (optional): TfidfVectorizer

        A vectorizer object that tracks our vocabulary of words.

    

    Returns

    -------

    word_cloud_image: WordCloud

        A word-cloud image containing the top words within the cluster.



    """

    indices = df_cluster.Index.values

    summed_tfidf = np.asarray(tfidf_matrix[indices].sum(axis=0))[0]

    data = {'Word': vectorizer.get_feature_names(),'Summed TFIDF': summed_tfidf}  

    # Words are ranked by their summed TFIDF values.

    df_ranked_words = pd.DataFrame(data).sort_values('Summed TFIDF', ascending=False)

    words_to_score = {word: score

                     for word, score in df_ranked_words[:max_words].values

                     if score != 0}

    

    # The word cloud's color parameters are modefied to maximize readability.

    cloud_generator = WordCloud(background_color='white',

                                color_func=_color_func,

                                random_state=1)

    wordcloud_image = cloud_generator.fit_words(words_to_score)

    return wordcloud_image



def _color_func(*args, **kwargs):

    # This helper function will randomly assign one of 5 easy-to-read colors to each word.

    return np.random.choice(['black', 'blue', 'teal', 'purple', 'brown'])



wordcloud_image = cluster_to_image(cluster_groups[0])

plt.imshow(wordcloud_image, interpolation="bilinear")

plt.show()
def plot_wordcloud_grid(cluster_groups, num_rows=5, num_columns=4):

    # This function plots all clusters as word-clouds in 5x4 subplot grid.

    figure, axes = plt.subplots(num_rows, num_columns, figsize=(20, 15))

    cluster_groups_copy = cluster_groups[:]

    for r in range(num_rows):

        for c in range(num_columns):

            if not cluster_groups_copy:

                break

                

            df_cluster = cluster_groups_copy.pop(0)

            wordcloud_image = cluster_to_image(df_cluster)

            ax = axes[r][c]

            ax.imshow(wordcloud_image, interpolation="bilinear")   

            # The title of each subplot contains the cluster id, as well as the cluster size.

            ax.set_title(f"Cluster {df_cluster.Cluster.iloc[0]}: {df_cluster.shape[0]}")

            ax.set_xticks([])

            ax.set_yticks([])



plot_wordcloud_grid(cluster_groups)

plt.show()
task_string = """Task Details

What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?



Specifically, we want to know what the literature reports about:



Data on potential risks factors

Smoking, pre-existing pulmonary disease

Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

Neonates and pregnant women

Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

Susceptibility of populations

Public health mitigation measures that could be effective for control"""
vectorizer = TfidfVectorizer(stop_words='english')

matrix = vectorizer.fit_transform(list(df.abstract) + [task_string])

similarities_to_task = matrix[:-1] @ matrix[-1].T
def compute_mean_similarity(df_cluster):

    # Computes the mean cosine similarity of a cluster to the task string.

    indices = df_cluster.Index.values

    return similarities_to_task[indices].mean()



mean_similarities = [compute_mean_similarity(df_cluster) 

                     for df_cluster in cluster_groups]

sorted_indices = sorted(range(len(cluster_groups)),

                        key=lambda i: mean_similarities[i], reverse=True)

sorted_cluster_groups = [cluster_groups[i] for i in sorted_indices]

plot_wordcloud_grid(sorted_cluster_groups)

plt.show()
df = pd.concat(sorted_cluster_groups[:4])

print(f'Our 4 most relevant clusters cover {df.shape[0]} articles.')
df_relevant = pd.concat(sorted_cluster_groups[:4])

print(f'Our 4 most relevant clusters cover {df.shape[0]} relevant articles.')
# These phrases are explicity associated with the SARS-Cov-2 Pandemic.

pattern_list = ['COVID-19', 'novel coronavirus', 'novel corona virus', '2019-nCoV', 'SARS-CoV-2']

regex = compile_pattern_list(pattern_list)



df = df[[regex.search(t) is not None for t in df.title.values]]

print(f"{df.shape[0]} of our clinically relevant records reference the Pandemic in their title.")
import spacy  

import string

nlp = spacy.load("en_core_web_sm")  



abstract_sentences = set() # Contains all unique sentences.

seen_sentences = set() # Used to tract duplicates.

duplicate_count = 0 # Used to count duplicates.



trans_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def simplify_text(text):

    # Removes all spacing, casing and punctuation, for better duplicate detection.

    return text.translate(trans_table).lower().replace(' ', '')



for abstract in df.abstract.values:

    for sentence in nlp(abstract).sents:

        simplified_text = simplify_text(sentence.text)

        if simplified_text not in seen_sentences:

            # Sentence is not a duplicate.

            seen_sentences.add(simplified_text)

            abstract_sentences.add(sentence.text)

            

        else:

            duplicate_count += 1

            

num_unique = len(abstract_sentences)

percent_unique = 100 * num_unique / (num_unique + duplicate_count)

print(f"Our 1334 abstracts contain {num_unique} unique sentences.")

print(f"{int(percent_unique)}% of all sentences are unique.")
import glob

df_full = df[df.has_pdf_parse == True]

relevant_fnames  = {sha + '.json' for sha in df_full.sha.values}

relevant_fpaths = [p for p in glob.glob('../input/CORD-19-research-challenge/**/*.json', recursive=True)

                   if p.split('/')[-1] in relevant_fnames]
import json



body_sentences = set()

for file_path in relevant_fpaths:

    with open(file_path) as f:

        data = json.load(f)

        full_text = ' '.join(t['text'] 

                             for t in data['body_text'])

        

        for sentence in nlp(full_text).sents:

            simplified_text = simplify_text(sentence.text)

            if simplified_text not in seen_sentences:

                # Sentence is not a duplicate.

                seen_sentences.add(simplified_text)

                body_sentences.add(sentence.text)

                

all_sentences = abstract_sentences | body_sentences

print(f"Article body inclusion gives us {len(all_sentences)} unique sentences.")
def search(regex_string):

    # Returns sentences that match the inputted regex.

    regex = compile_(regex_string)

    return sorted([s for s in all_sentences if regex.search(s)])



diabetes_sentences = search('Diabetes')

ebola_sentences = search('Ebola')



print('DIABETES:')

for sentence in diabetes_sentences[-10:]:

    print(sentence)

    

print('\nEBOLA')

for sentence in ebola_sentences[-10:]:

    print(sentence)
num_percent_diabetes = len([s for s in diabetes_sentences if '%' in s])

num_percent_ebola = len([s for s in ebola_sentences if '%' in s])



print(f"{num_percent_diabetes} of the {len(diabetes_sentences)} Diabetes sentences contain a percentage.")

print(f"{num_percent_ebola} of the {len(ebola_sentences)} Ebola sentences contain a percentage")
percent_sentences = {s for s in all_sentences if '%' in s}

print(f"{len(percent_sentences)} of our {len(all_sentences)} sentences contain a percentage.")
from collections import defaultdict

def extract_noun_chunks(text):

    # Returns a set of unique noun chunks present in a text.

    text = remove_parans(text) # Strips our parenthesized text for better noun chunk parsing.

    return {noun_chunk.text.lower()

            for noun_chunk in nlp(text).noun_chunks}



def remove_parans(text):

    # Removes short stretches of parenthesized text for more accurate parsing.

    return re.sub(r'\((.{,20})\)', '', text).replace('  ', ' ')



text = ("Most of the infected patients were men (30 [73%] of 41); less than half "

        "had underlying diseases (13 [32%]), including diabetes (eight [20%]), "

        "hypertension (six [15%]), and cardiovascular disease (six [15%]).'")

extract_noun_chunks(text)
from nltk.corpus import wordnet as wn 

# Common medical hypernyms

medical_hypernyms = {'symptom', 'ill_health', 'negative_stimulus',

                    'bodily_process', 'disease', 'illness', 'pain',

                    'body_substance', 'medicine', 'drug', 'pathology',

                    'breathing', 'therapy', 'medical_care', 'treatment',

                    'disorder'}



def is_medical(text):

    # Returns True any an hypernym of an word in `text` overlaps with `medical_hypernyms`.

    for word in text.lower().split():

        hypernyms = set(get_hypernyms(word))

        if hypernyms & medical_hypernyms:

            return True

        

    return False 



def get_hypernyms(word):

    # Returns a set of hypernyms associated with each word.

    hypernym_list = []

    # Accesses all synsets (synonyms and and usage-categories) of word

    synsets = wn.synsets(word)

    count = 0

    while synsets and count < 4:

        # Extracts all hypernyms of most common sysnset. We limit ourselves to just the 

        # first 4 layers of the hypernym hierarchy.

        hypernyms = synsets[0].hypernyms()

        hypernym_list.extend([h.name().split('.')[0]

                              for h in hypernyms])

        synsets = hypernyms

        count += 1

    return hypernym_list





assert is_medical('Diabetes')

assert is_medical('Breathing problems')

assert not is_medical('problems')

assert not is_medical('men')
print(f"First synset of 'less':\n{wn.synsets('less')[0]}")

print(f"\nHypernyms of 'less':\n{get_hypernyms('less')}")

assert is_medical('less than half')
import requests

from bs4 import BeautifulSoup as bs



def search_wikipedia(noun_chunk):

    """Searches Wikipedia for a noun chunk. If a noun chunk is associated with a medical 

    specialty, then  the function returns both the specialty and the page title. Otherwise,

    it returns a None"""

    

    # Heuristic cleaning is carried out on the noun chuck

    name = _clean_string(noun_chunk)

    # The Wikipedia page that we'll try to load.

    url = f'https://en.wikipedia.org/wiki/{name.replace(" ", "_")}'

    response = _scrape_url(url)

    if response is None:

        # No page loaded.

        return None

    

    # We parse the page using Beautiful Soup.

    soup = bs(response.text)

    table = soup.find('table', class_='infobox')

    if not table:

        # No Info Box has been found.

        return None

    

    specialty = None

    for table_row in table.find_all('tr'):

        text = table_row.text

        if text.startswith('Specialty'):

            # We've uncovered a medical specialy.

            specialty = text[len('Specialty'):].strip()

            break

    

    if not specialty:

        # No specialty has been found.

        return None

    

    # We clean the title, prior to returning the medical specialy and the title.

    title = _clean_title(soup.title.text, name)

            

    return specialty, title



def _clean_string(text):

    # Filters out common patterns at the start of medical strings that interfere with Wiki crawling.

    deletion_patterns = [r'^(a|an|the)\b',  r'^(mild|low|moderate|severe|high|old)\b']

    for regex_string in deletion_patterns:

        text = re.sub(regex_string, '', text, flags=re.I).strip()

        

    return text





def _scrape_url(url):

    # Scrapes the url.

    for _ in range(3):

        # Repeat 3 times in case of timeout.

        try:

            response = requests.get(url, timeout=10)

            # Return scraped response if page has loaded.

            return response if response.status_code == 200 else None

        

        except timeout:

            continue

    

    return None



def _clean_title(title, name):

    # Removes noise from the end of the string.

    title = title.replace('- Wikipedia', '').strip()

    # Sometimes symptom-terms redirect to diease pages. This could lead to confusion later in our

    # analysis, when we execute symptom extraction. Hence, we use a regex to deal with this edge-case.

    if title.endswith('disease') and re.search(r'symptoms?$', name):

        return name.capitalize()

        

    return title



for term in ['a moderate fever', 'dyspnea', 'diabetes', 'hypertension']:

    print(f"Searching for '{term}'.")

    specialty, alias = search_wikipedia(term)

    print(f"The term is also know as '{alias}.'")

    print(f"Its associated clinical specialties are: {specialty}.\n")

    

assert search_wikipedia('less than half') is None
from collections import defaultdict

# A mapping between medical terms and sentence mentions.

medical_terms = defaultdict(list)

# A mapping between medical terms and the associated page-titles / specialties 

# that were scraped from Wikipedia

term_to_wiki_data = {}

# A cache of all encountered noun chunks that are not medical terms.

bad_noun_chunks = set()



# We filter out certain noun-chunks to speed-up search results. This includes numeric nouns

# associated with percentages as well as various variations of the term SARS.

bad_patterns = [r'respiratory(\s+distress)? syndrome', r'[0-9]']

bad_regex = compile_pattern_list(bad_patterns, escape=False, add_bounderies=False)



for sent in percent_sentences & abstract_sentences:

    # We start by iterating over the relevant sentences from the abstracts.

    for noun_chunk in extract_noun_chunks(sent):

        if noun_chunk in bad_noun_chunks or bad_regex.search(noun_chunk):

            continue

                

        if noun_chunk in medical_terms:

            # This noun chunk is a known medical term. We append this sentence to that term's 

            # associated sentence list.

            medical_terms[noun_chunk].append(sent)

            continue



        # We do a wiki-search for previously unseen noun-chunk

        wiki_result = search_wikipedia(noun_chunk)

        if wiki_result is not None:

            # New medical term discovered using Wikipedia.

            medical_terms[noun_chunk].append(sent)

            term_to_wiki_data[noun_chunk] = wiki_result      

        else:

            # We cache the non-medical term.

            bad_noun_chunks.add(noun_chunk)

            

for sent in percent_sentences & body_sentences:

    for noun_chunk in extract_noun_chunks(sent):

        # We iterate over the relevant sentences in the article bodies and subsequently

        # update our sentence mentions.

        if noun_chunk in medical_terms:

            medical_terms[noun_chunk].append(sent)

        
print(f"We've obtained {len(medical_terms)} medical terms.")

print("The top 10 terms / mention-counts are:\n")

for term, sentences in sorted(medical_terms.items(), 

                              key=lambda x: len(x[1]), reverse=True)[:10]:

    print(term, len(sentences))
for sentence in medical_terms['fever'][:5]:

    print(sentence)
!pip install --upgrade pytorch_transformers
import torch

from transformers import DistilBertTokenizer                    

from transformers import DistilBertForQuestionAnswering

                                           

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad') 

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

def answer_question(question, text):

    # Uses DistilBERT SQuaD trained model for question-answering.

    input_text = f"{question} [SEP] {text}" 

    input_ids = tokenizer.encode(input_text) 

    start_scores, end_scores = model(torch.tensor([input_ids])) 

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

    # Strips out pound-signs and extra spaces returned by the model.

    answer = answer.replace(' ##', '').strip()

    return answer



text = ("The most common symptoms at the onset of illness were fever (82.1%), cough (45.8%), "

       "fatigue (26.3%), dyspnea (6.9%) and headache (6.5%).")

answer_question("Percent fever?", text)
def ask_percentage(term, text):

    # Asks for the percentage that's associated with an input medical term.

    question = f'Percent {term}?'

    answer = answer_question(question, text)

    # Strips out extra-spaced returned by DistilBERT model.

    answer = answer.replace(' %', '%').replace(' . ', '.')

    # Extracts the percentage from the asnwer, if any.

    percentage = _extract_percentage(answer)

    # Converts the percentage into a float.

    return float(percentage[:-1]) if percentage else None

    

regex_percent = compile_( r'\b[0-9]{1,2}(\.[0-9]{1,2})?%')

def _extract_percentage(text):

    # Uses a regex to extract a percentage from the text.

    match = regex_percent.search(text)

    return text[match.start(): match.end()] if match else None 



ask_percentage('fever', text)
def match_percentages(term, text_list):

    """The function returns a representative percentage of a term across of a list of mentions, along

    with its representative mention.

    

    Parameters

    ----------

    term: str

        A clinical term whose percentage we wish to obtain (Example: Fever)

    text_list: [str]

        A list of of text mention. Each metion potentially contains percentage match to `term`.

    

    Results

    -------

    representative_percentage: float

        An extracted percentage that is closest to the median.

    representative_mention: str

        The element of `text_list` in which `representative_percentage` has been found.

    """

    matches = []

    for text in text_list:

        percentage = ask_percentage(term, text)

        if percentage is not None and percentage != 0.95:

            # We filter all references to 95%. These mostly correspond with statistical significance,

            # and not with observed fequencies.

            matches.append((percentage, text))

    

    if not matches:

        # No percentages have been found.

        return None, None

    

    percentages = np.array([m[0] for m in matches])

    # We use the median instead of the mean, due to risk of overweighed extreme values.

    dist_to_median = np.abs(percentages - np.median(percentages))

    return matches[dist_to_median.argmin()]

    

rep_percentage, rep_mention = match_percentages('fever', medical_terms['fever'])

print(f"\nThe representative percentage of fever is {rep_percentage}%.")

print(f"It occurs in the following sentence:\n{rep_mention}")
all_fever_sentences = []

fever_aliases = []



_, fever_title =  term_to_wiki_data['fever']

# The term 'fever' redirects to a Wiki=page titled 'Fever'/

assert fever_title == 'Fever'



for term, sentences in medical_terms.items():

    _, wiki_title = term_to_wiki_data[term]

    if wiki_title == fever_title:

        fever_aliases.append(term)

        all_fever_sentences.extend(sentences)





print(f"The following {len(fever_aliases)} terms are aliases of Fever:\n {fever_aliases}")



rep_percentage, rep_mention = match_percentages('fever', all_fever_sentences)

print(f"\nThe representative percentage of Fever is {rep_percentage}%.")

print(f"It occurs in the following sentence:\n{rep_mention}")
# Maps a medical word to its aliases, based on wikipedia redirects.

aliases = defaultdict(list)

# An aggregation of all mentions that's associated with a term

aggregated_sentences = defaultdict(list)

for term, sentences in medical_terms.items():

    _, name = term_to_wiki_data[term]

    aliases[name].append(term)

    aggregated_sentences[name].extend(sentences)

    

table = {'Name': [], 'Count': [], 'Specialties': [], 'Percentage': [],

         'Percentage Text': [], 'Aliases': []}



# Sorting ensures constitancy of indices in DataFrame.

for name, alias_list in sorted(aliases.items()):

    # Iterate over each aggregated medical concept.

    sentences = aggregated_sentences[name]

    count = len(sentences)

    percentage, percent_text = match_percentages(name, sentences)

    specialties = term_to_wiki_data[alias_list[0]][0]

    table['Name'].append(name)

    # The count of total aggregated sentences.

    table['Count'].append(count)

    # The wiki-determined medical specialties associated with the concept.

    table['Specialties'].append(specialties)

    # The computed representative percentage.

    table['Percentage'].append(percentage)

    table['Percentage Text'].append(percent_text)

    # The aggregated aliases of the medical concept

    table['Aliases'].append(alias_list)

    

df_medical = pd.DataFrame(table)

df_medical.head(3)
df_medical.sort_values(['Count', 'Specialties'], ascending=False, inplace=True)

df_medical.head(10)
text = "Fever ( 89.8% ) and Cough ( 67.2% ) were common clinical symptoms, while diabetes and hypertension were a common comorbidity." 

# Please note that question-answering better if we strip out the parantheses fom the text.

answer_question('What were the symptoms?', remove_parans(text))
# Constructs a symptom regular expression. Symptoms are sometimes referenced as "manifestations."

symptom_synonyms = ['symptom', 'manifestation']

# A word-boundary is not added to the right-side of the regex, to allow for plurality.

symptom_regex = compile_pattern_list([r'\b' + s for s in symptom_synonyms],

                                     escape=False, add_bounderies=False,)



def is_symptom(term, min_fraction=0.1): 

    """Returns True a medical term is referred to as a symptom among its mentions.

    

    Parameters

    ----------

    term: str

        A medical term within our `df_medical` DataFrame.

    min_fraction (optional): float

        The minimum fraction of `term` sentences that must refer to the term 

        as symptom for the function to return True. Preset to 10%.

    """

    aliases = df_medical.loc[df_medical.Name == term].Aliases.values[0]

    # Compiles a regular expression containing all the aliases of `term`.

    alias_regex = aliases_to_regex(aliases)

    count = 0

    # The minimum number of symptom matches required to return True.

    min_count = int(min_fraction  * len(aggregated_sentences[term]))

    for sentence in aggregated_sentences[term]:

        # We ask for the symptoms of the sentence.

        answer = ask_for_symptoms(sentence)

        if answer and alias_regex.search(answer):

            # The term is referred to as a symptom in the sentence.

            count += 1

            if count >= min_count:

                return True

            

    return False        

    



def ask_for_symptoms(text):

    """Using question-answering to ask for the symptoms in the text"""

    text = remove_parans(text)

    for question in [f'What are the {s}s?' for s in symptom_synonyms]:

        # We ask for symptoms, as well as clincial manifestations.

        answer = answer_question(question, text)



        if not answer:

            continue

        

        # In order to up precision, we check for the mention of symptoms 

        # in the answer, or right before the answer.

        if re.search(r'\b(symptoms|manifestations)\s+(were|are)$',

                     text.split(answer)[0].strip()):

            return f"symptoms are {answer}"

        

        if symptom_regex.search(answer):

            return answer

    

    return None

                

def aliases_to_regex(aliases):

    # Transforms the alieases into reguler expersions. The longer aliases 

    # are matched first.

    aliases = sorted(aliases, key=len, reverse=True)

    return compile_pattern_list(aliases)



for term in ['Fever', 'Cough', 'Diabetes', 'Hypertension']:

    if is_symptom(term):

        print(f"{term} is a symptom.")

    else:

        print(f"{term} is not a symptom.")
are_symptoms = np.array([is_symptom(name) for name in df_medical.Name.values])

df_symptom = df_medical[are_symptoms]

df_not_symptom = df_medical[~are_symptoms]

print(f"{df_symptom.shape[0]} of our medical terms are symptoms.")

print(f"{df_not_symptom.shape[0]} of our medical terms are not symptoms.")
df_symptom
df_not_symptom.head(10)
from collections import Counter

# We count specilties by the number of terms that fall within each specialty. Each

# unique term is counted only once. Alternatively, we could weighing each specialty

# by the total number of term mentions.

specialties = Counter(df_not_symptom.Specialties.values)

print(f"We have {len(specialties)} medical specialties in total.")

print("The top-ranking specialties are:\n")

for specialty, count in specialties.most_common():

    if count <= 2:

        break

        

    print(f"{specialty}:  {count}")
from IPython.core.display import display, HTML



def display_specialty(specialty):

    # Visualizes key information about a specialty using HTML.

    html = specialty_to_html(specialty)

    display(HTML(html))

    

def specialty_to_html(specialty):

    # Extracts key information about a specialty, highlighting features using HTML.

    df = df_not_symptom[df_not_symptom.Specialties.isin([specialty])]

    #html = f'<h3>{specialty}</h3></br>'

    html = ''

    for index, row in df.iterrows():

        # We iterate of all non-symptometic medical terms within the specialty.

        tup = row[['Name', 'Count', 'Aliases', 'Percentage', 'Percentage Text']]

        name, count, aliases, percentage, text = tup

        # For each medical term, we output its index as well as count.

        html += f'<h4>{index} {name.upper()}: Count {count}</h4>' 

        

        if text:

            # A representative percentage is associated with the medical term.

            # We'll highlight that percentage within the representative text.

            percentage = str(int(percentage)) if int(percentage) == percentage else str(percentage)

            regex = re.compile(r'\b%s\b' % percentage)

            # The percentage is boldened and colored blue.

            text = add_markup(regex, text, multi_matches=False,

                              color='blue', bold=True)     

        else:

            # If no percentage is found, then we choose a random sentence that's

            # associated with the medical term.

            text = aggregated_sentences[name][0]

        

        regex = aliases_to_regex(aliases)

        # We color the medical term red within the text, for a better display.

        text = add_markup(regex, text, color='red')

        html += text + '</br></br>'

    

    return f'<html>{html}</html>'

    

        

def add_markup(regex, text, multi_matches=True, **kwargs):

    """ Adds markup to all matches of a regular expression within the text.

    

    Parameters

    ----------

    regex: re.Pattern

        A compiled regular expression that we match against the text.

    text: str

        Our inputted text string.

    multi_matches (optional): Bool

        If True, then multiple regex matches will be marked up within the

        text

        

    Returns

    -------

    marked_text: str

        Text with HTML markup added to all regex matches.

    """

    offset = 0

    for match in regex.finditer(text):

        old_length = len(text)

        span = (match.start() + offset, match.end() + offset)

        text = _add_span_markup(span, text, **kwargs)

        # Offset tracks length-shift in the text due to markup addition.

        offset += len(text) - old_length

        if not multi_matches:

            break

        

    return text

    

def _add_span_markup(span, text, color='black', bold=False):

    """Adds markup across a single span of text. Colors and optionally

    bolds that span.

    

    Parameters

    ----------

    span: (int, int)

        The start and end span of text that we'll markup.

    text: str

        The complete text

    color (optional): str

        The color to assign the marked-up span.

    bold (optional): bool

        If True, than boldens the marked-up span.

        

    Returns

    -------

    marked_text: str

        Text with HTML markup added across the specified span.

    """

    start, end = span

    snippet = text[start: end]

    html_snippet = f'<font color="{color}">{snippet}</font>'

    if bold:

        html_snippet = f'<b>{html_snippet}</b>'

    

    return text[:start] + html_snippet + text[end:]

display_specialty('Infectious disease')
display_specialty('Pulmonology')
comorbidity_indices = {22, 9}

symptom_indices = {96, 104, 33}
display_specialty('Hematology')
display_specialty('Ophthalmology')
comorbidity_indices.add(37)

symptom_indices.update([52, 42])
display_specialty('Cardiology')
comorbidity_indices.update([56, 15])
display_specialty('Endocrinology')
comorbidity_indices.update([36, 80, 60])
display_specialty('Nephrology')
comorbidity_indices.add(21)
from itertools import chain

comorbid_names = df_not_symptom[df_not_symptom.index.isin(comorbidity_indices)].Name.values

# A set of sentences that mention a comorbidity.

comorbid_sentences = set((chain.from_iterable([aggregated_sentences[name] 

                                               for name in comorbid_names])))

for specialty, count in specialties.most_common()[7:]:

    # We iterate over the remaining 40 specialtities.

    df_tmp = df_not_symptom[df_not_symptom.Specialties.isin([specialty])]

    for name in df_tmp.Name.values:

        if comorbid_sentences & set(aggregated_sentences[name]):

            # The term in the specialty is mentioned in a sentence that also

            # mentions a comorbidity.

            display_specialty(specialty)

            break

comorbidity_indices.update([14, 72, 18, 17, 23, 2, 30, 99, 61, 65])

print(f"We uncovered {len(comorbidity_indices)} total comorbidities")
is_comorbid = df_not_symptom.index.isin(comorbidity_indices)

df_comorbidity = df_not_symptom[is_comorbid]

df_comorbidity.sort_values(['Count', 'Specialties'], ascending=False)

df_comorbidity
df_symptom = df_symptom.append(df_not_symptom.loc[symptom_indices])

df_not_symtpom = df_not_symptom.drop(index=symptom_indices)

print(f"We uncovered {df_symptom.shape[0]} total symptoms")


not_comorbid = ~df_not_symptom.index.isin(comorbidity_indices)

df_other = df_not_symptom[not_comorbid]



total = df_medical.shape[0]

percent_other = int(100 * df_other.shape[0] / total)

print(f"{100 - percent_other}% of our medical terms are symptoms or comorbidities.")

print(f"The remaining {percent_other}% of terms fall into the 'Other' category.")
def aliases_to_regex(df): 

    # Converts all aliases with a medical DataFrame into a regular expression

    patterns = set(np.hstack(df.Aliases.values))

    patterns.update([name.lower() for name in df.Name.values])

    patterns = sorted(patterns, key=len, reverse=True)

    return compile_pattern_list(patterns)



# Our regular expressions match comorbidities, symptoms, other medical terms, and also percentages.

regex_list = [aliases_to_regex(df)

              for df in [df_comorbidity, df_symptom, df_other]] + [regex_percent]



# Markup colors are assigned to each match type.

colors = ['blue', 'green', 'brown', 'black']

format_kwargs = [{'color': c} for c in colors]

# Percentages will be boldened in the marked up text.

format_kwargs[-1]['bold'] = True



def mark_matches(text):

    """A input text is converted into a marked up HTML string, based on 

    terminology matches. All match counts are also returned.

    

    Parameters

    ----------

    text: str

        The text we match against.

        

    Returns

    -------

    marked_text: str

        A marked-up version of the text based on regex matches.

        

    match_counts: [int, int, int, int]

        A list of four match counts, one for each regex in `regex_list`.

    """

    match_counts = []

    for regex, kwargs in zip(regex_list, format_kwargs):

       

        match_counts.append(len(regex.findall(text)))

        text = add_markup(regex, text, **kwargs)

    

    return text, match_counts

text = aggregated_sentences['Diabetes'][0]

marked_text, match_counts = mark_matches(text)

for count, name in zip(match_counts, ['comorbidities', 'symptoms', 'other terms', 'percentages']):

    print(f"The text matches {count} {name}.")



print('\nDisplaying the marked-up text:')

display(HTML(mark_matches(text)[0]))
def ranked_search(query_string, results_per_page=7, page=1):

    """A ranked search tool for extracting relevant sentences.

    The top-ranked matches are displayed as HTML.

    

    Parameters

    ----------

    query_string: str:

        Our query string that's used to match the sentences. This string

        can be a regular expression.

    results_per_page (optional): int

        The number of results to dispaly within a page of output.

    page (optional): int

        The page-number. It allows us to flip through multiple 

        pages of results.

    """

    regex = compile_(query_string)

    matches = [s for s in all_sentences if regex.search(s)]

    # This dictionary maps matched sentences to a tuple that is used for ranking purpuses.

    matches_to_ranking = {}

    for match in matches:

        # All matches to the query are marked in red.

        marked_match = add_markup(regex, match, color='red', bold=True)

        marked_match, match_counts = mark_matches(marked_match)

        # The number of diverse categorical matches.

        num_matches = len([count for count in match_counts if count])

        # The total number of matches.

        num_total_matches = sum(match_counts)

        # Assigns each match a tuple, for ranking purposes.

        matches_to_ranking[marked_match] = (match in abstract_sentences, num_matches,

                                            num_total_matches)

        

    # Ranks matches by importance

    sorted_matches = [m[0] for m in sorted(matches_to_ranking.items(),

                                           key=lambda x: x[1], reverse=True)]

    start = (page - 1) * results_per_page

    end = start + results_per_page

    # Displays the top results using HTML.

    html = '<br>'.join(sorted_matches[start: end])

    display(HTML(html))
ranked_search('risk factors')
ranked_search(r'(pregnan(t|cy)|pre-?natal)')
ranked_search(r'(pregnan(t|cy)|pre-?natal)', page=2)
ranked_search(r'co-?infection')
ranked_search(r'co-?morbidities')
conditionals = {'comorbidities': (72.2, 37.3), 'dyspnea': (63.9, 19.6), 'anorexia': (66.7, 30.4)}
ranked_search(r'progression group')
conditionals['smoking'] = (27.3, 3.0)
ranked_search(r'non-?survivor')
conditionals.update({'disseminated intravascular coagulation': (71.4, 0.6),

                     'BMI>25': (88.24, 18.95)})
ranked_search(r'\bstabilized\b')
prior_probs = [0.141, 0.859]
for condition, tup in conditionals.items():

    prob_a, prob_b = np.array(tup) / 100

    print(f'P({condition} | Deterioration) = {prob_a:.3f}')

    print(f'P({condition} | Stabilization) = {prob_b:.3f}')

    print()
# A matrix of conditional probabilities. Each column corresponds to a feature in `conditionals`.

# The first row corresponds to the prob(feature | deterioration). The second row corresponds

# to prob(feature | stabilization)

conditional_matrix = np.vstack([np.array(tup) / 100 

                                for tup in conditionals.values()]).T

# We'll use Maximum a posteriori estimation, so taking the Log of the

# conditional probabilities will be sufficient.

log_conditional_matrix = np.log(conditional_matrix)



def naive_bayes(feature_vector):

    results = log_conditional_matrix @ feature_vector + np.log(prior_probs)

    # Returns 1 if the conditional probability of deterioration (index 0) is maximized, and 0 otherwise.

    return 1 - results.argmax()



assert naive_bayes([0] * 6) == 0

assert naive_bayes([1] * 6) == 1
features = np.array([1] + 5 * [0])

phrase = 'deteriorate' if naive_bayes(features) else 'stabilize'

print(f"The patient is more likely to {phrase}.")
features[1] = 1

phrase = 'deteriorate' if naive_bayes(features) else 'stabilize'

print(f"The patient is more likely to {phrase}.")
text ="The patient suffers from diabetes and is a chronic smoker. He has a BMI of 28."

comorbidity_regex = aliases_to_regex(df_comorbidity)

assert comorbidity_regex.search(text) is not None
smoking_regex = compile_(r'\bsmok(er|es|ed|ing)\b')

# We include all aliases of the various terms, such as "Shortness of Breath", an alias of dyspnea.

tup = [aliases_to_regex(df_medical[[term in aliases 

                                    for aliases in df_medical.Aliases.values]])

       for term in ['dyspnea', 'anorexia', 'disseminated intravascular coagulation']]



dyspnea_regex, anorexia_regex, coagulation_regex = tup

feature_regex_list = [comorbidity_regex, smoking_regex, dyspnea_regex, anorexia_regex, coagulation_regex]

def features_from_regex(text):

    return [int(regex.search(text) is not None) for regex in feature_regex_list]



print(features_from_regex(text))
answer_question('What is the patient\'s BMI?', text)
def is_large_bmi(text):

    # Returns 1 if a BMI of greater > 25 is mentioned in the text, and zero otherwise.

    answer = answer_question('What is the patient\'s BMI?', text)

    match = re.search(r'[0-9]+', text)

    if match:

        bmi_value = int(text[match.start(): match.end()])

        return int(bmi_value > 25)

    

    return 0



assert is_large_bmi(text) == 1
def predict_risk(text):

    # Predicts a patient's risk of deterioration from a textual description.

    features = features_from_regex(text) 

    # Please note, we are assuming that a BMI description is always included in the text.

    features.append(is_large_bmi(text))

    return(naive_bayes(features))



phrase = 'deteriorate' if predict_risk(text) else 'stabilize'

print(f"The patient is more likely to {phrase}.")