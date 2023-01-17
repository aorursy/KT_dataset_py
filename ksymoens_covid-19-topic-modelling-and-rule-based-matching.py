from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd
import numpy as np
import os
import glob
import spacy
from spacy.matcher import Matcher
import fasttext
import json
import warnings
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.colors as mcolors
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from wordcloud import WordCloud

warnings.simplefilter('ignore')
pd.options.mode.chained_assignment = None  # default='warn'
data_path = '/kaggle/input/cord19researchchallenge-old/CORD-19-research-challenge/' # path for Kaggle
#data_path = os.getcwd() + '/data/CORD-19-research-challenge/'
meta_df = pd.read_csv(data_path + 'metadata.csv',
                      dtype={'pubmed_id': str, 'Microsoft Academic Paper ID': str, 'doi': str})
all_json = glob.glob(f'{data_path}/**/*.json', recursive=True)


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.body_text = []
            self.paragraphs = []
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            # Paragraps
            for entry in content['body_text']:
                self.paragraphs.append((entry['section'].capitalize().replace('\\', ''), entry['text']))
            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):
        return f'{self.paper_id}: \n\n{self.abstract[:200]}... \n\n{self.body_text[:200]}...'

    
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'meta_abstract': [], 'publication_date': [], 'paragraphs': []}

for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 20) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['body_text'].append(content.body_text)
    dict_['paragraphs'].append(content.paragraphs)

    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    dict_['authors'].append(np.nan if meta_data['authors'].empty else meta_data['authors'][0:1].item())
    dict_['title'].append(np.nan if meta_data['title'].empty else meta_data['title'][0:1].item())
    dict_['journal'].append(np.nan if meta_data['journal'].empty else meta_data['journal'][0:1].item())
    dict_['publication_date'].append(np.nan if meta_data['publish_time'].empty else meta_data['publish_time'][0:1].item())
    abstract_text = np.nan if meta_data['abstract'].empty else meta_data['abstract'][0:1].item()
    dict_['abstract'].append(abstract_text if pd.isnull(abstract_text) or abstract_text.partition(' ')[0].lower() != 'abstract' else abstract_text.partition(' ')[2])

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'publication_date', 'paragraphs'])
df_covid.abstract.fillna('', inplace=True)
df_covid.title.fillna('', inplace=True)
df_covid.head()
def merge_section_texts(paragraphs_list):
    """
    Concatenate paragraph texts of the same section
    :param paragraphs_list:
    :return:
    """
    merged_index = 0
    merged_paragraphs = [paragraphs_list[0]]
    for index in range(1, len(paragraphs_list)):
        if merged_paragraphs[merged_index][0] == paragraphs_list[index][0]:
            merged_paragraphs[merged_index] = (merged_paragraphs[merged_index][0], ' '.join(
                [merged_paragraphs[merged_index][1], paragraphs_list[index][1]]))
        else:
            merged_index += 1
            merged_paragraphs.append(paragraphs_list[index])
    return merged_paragraphs


df_covid['sections'] = df_covid.paragraphs.apply(merge_section_texts)
df_covid.drop('paragraphs', axis=1, inplace=True)
lang_detect_model = fasttext.load_model('/kaggle/input/ft-model/lid.176.bin') #https://fasttext.cc/docs/en/crawl-vectors.html


def get_language(content):
    """
    Function to detect the language of a string
    :param content: text
    :return: language
    """
    return lang_detect_model.predict(content.replace('\n', ' '))[0][0][-2:]


df_covid['paper_language'] = df_covid['body_text'].apply(get_language)
df_covid.head()
nlp = spacy.load('en_core_web_lg') #https://spacy.io/models/en
abstracts = list(nlp.pipe(df_covid['abstract'], batch_size=50, n_process=3))
df_abstract = df_covid.copy()
df_abstract['token'] = None
df_abstract['lemma'] = None
df_abstract['lemma_lower'] = None
df_abstract['token_pos'] = None
df_abstract['token_tag'] = None
df_abstract['token_dep'] = None
df_abstract['token_entity'] = None

df_abstract['token'] = df_abstract['token'].astype('object')
df_abstract['lemma'] = df_abstract['lemma'].astype('object')
df_abstract['lemma_lower'] = df_abstract['lemma_lower'].astype('object')
df_abstract['token_pos'] = df_abstract['token_pos'].astype('object')
df_abstract['token_tag'] = df_abstract['token_tag'].astype('object')
df_abstract['token_dep'] = df_abstract['token_dep'].astype('object')
df_abstract['token_entity'] = df_abstract['token_entity'].astype('object')


for i in range(len(abstracts)):

    l_token = list()
    l_lemma = list()
    l_lemma_lower = list()
    l_token_pos = list()
    l_token_tag = list()
    l_token_dep = list()
    l_token_entity = list()

    for token in abstracts[i]:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            l_token.append(token.text)
            l_lemma.append(token.lemma_)
            l_lemma_lower.append(token.lemma_.lower())
            l_token_pos.append(token.pos_)
            l_token_tag.append(token.tag_)
            l_token_dep.append(token.dep_)
            l_token_entity.append(token.ent_type_)

    df_abstract.loc[i,'token'] = l_token
    df_abstract.loc[i,'lemma'] = l_lemma
    df_abstract.loc[i,'lemma_lower'] = l_lemma_lower
    df_abstract.loc[i,'token_pos'] = l_token_pos
    df_abstract.loc[i,'token_tag'] = l_token_tag
    df_abstract.loc[i,'token_dep'] = l_token_dep
    df_abstract.loc[i,'token_entity'] = l_token_entity
df_abstract = pd.read_pickle('/kaggle/input/pickle/df_abstract.pickle')
df_abstract.abstract = df_abstract['abstract'].replace('', np.nan)
df = df_abstract.dropna()
for i in range(len(df)):
    for j in range(len(df.lemma.iloc[i])):
        if df.token_pos.iloc[i][j] not in ['NOUN','PROPN']:
            df.lemma.iloc[i][j] = ""    
for i in range(len(df)):
    df.lemma.iloc[i] = [string for string in df.lemma.iloc[i] if not len(string) < 3]
word_list = list()
for i in range(len(df)):
    for j in range(len(df.lemma.iloc[i])):
        word_list.append(df.lemma.iloc[i][j])
string_word_list = str(word_list)

string_word_list = string_word_list.replace("'","")
wordcloud = WordCloud(width = 800, height = 800,background_color ='white').generate(string_word_list)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 
def get_tf_idf(document_corpus):
# TFIDF -----------------------------------------------------------------------
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=0.001, max_df=0.85, strip_accents='ascii',
                                       tokenizer=lambda x: x,
                                       preprocessor=lambda x: x)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(document_corpus.lemma).toarray()
    # place tf-idf values in a pandas data frame
    df_tfidf = pd.DataFrame(tfidf_vectorizer_vectors, columns=tfidf_vectorizer.get_feature_names())
    return df_tfidf
df_tfidf = get_tf_idf(df)
list(df_tfidf.columns)
len(df_tfidf)
#use a dictionary comprehension to generate the largest_n values in each row of the dataframe. 
#transpose the dataframe and then applied nlargest to each of the columns. 
#use .index.tolist() to extract the desired top_n columns. 
#transposed this result to get the dataframe back into the desired shape.

top_n = 10
df_tfidf_top10 = pd.DataFrame({n: df_tfidf.T[col].nlargest(top_n).index.tolist() 
                  for n, col in enumerate(df_tfidf.T)}).T
df_tfidf_top10.head(30)
len(df_tfidf_top10)
# LDA model -------------------------------------------------------------------

# Creates dictionary, which is a mapping of word IDs to words.
words = corpora.Dictionary(df.lemma)
# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in df.lemma]
n_topics = 10
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=words,
                                            num_topics=n_topics,
                                            random_state=2,
                                            update_every=1,
                                            passes=10,
                                            alpha=0.001,
                                            eta=0.001,
                                            per_word_topics=True,
                                            minimum_probability=0)
print(lda_model.print_topics())
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=df.lemma, dictionary=words, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)
# Read the data
#df_covid = pd.read_pickle(os.getcwd() + '/data/processed/df_covid.pickle')
df_covid = pd.read_pickle("df_covid.pickle")
# Initialize the matcher
matcher = Matcher(nlp.vocab)
# Keep only the English papers
df_covid_en = df_covid[df_covid['paper_language']=='en']
print('Number of English papers in the dataset: ' + str(df_covid_en.shape[0]))
covid19_synonyms = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b']

# Helper functions
def abstract_title_filter(df, search_string):
    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))


def tagger(df, synonym_list, tag_suffix):
    df[f'tag_{tag_suffix}'] = False
    for synonym in synonym_list:
        synonym_filter = abstract_title_filter(df, synonym)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    return df
# Filter the relevant papers
df_covid_en = tagger(df_covid_en, covid19_synonyms, 'covid19')
df_covid19 = df_covid_en[df_covid_en['tag_covid19']]
print('Number of papers about Covid-19: ' + str(df_covid19.shape[0]))
# https://github.com/explosion/spaCy/blob/master/examples/pipeline/custom_sentence_segmentation.py
def prevent_sentence_boundaries(doc):
    for token in doc:
        if not can_be_sentence_start(token):
            token.is_sent_start = False
    return doc


def can_be_sentence_start(token):
    if token.i > 1 and token.nbor(-2).text == 'al':
        return False
    return True

nlp.add_pipe(prevent_sentence_boundaries, before='parser')
# Helper functions for pattern matching
def get_section_title(span, section_list):
    """
    Function to search in which section of the paper
    the detected pattern match (span) is found.
    """
    section_title = ''
    for section in section_list:
        if span.sent.text.strip() in section[1]:
            section_title = section[0]
    return section_title


def get_paragraph(span, doc):
    l_sents = []
    sentences = list(doc.sents)
    for i in range(len(sentences)):
        if sentences[i].text == span.sent.text:
            if i+1 <= len(sentences)-1 and i-1>=0: 
                l_sents = [sentences[i - 1].text, sentences[i].text, sentences[i + 1].text]
            else:
                l_sents = [sentences[i].text]
    return ' '.join(l_sents)


def get_matches(doc, matcher, matcher_string_id):
    """
    Function to search for all relevant mathes in a document (doc).
    We only match on the requested match patterns 
    (matcher_string_id: list of IDs referring to the matching pattern you want to use)
    """
    matches = []
    raw_matches = matcher(doc)
    matcher_id = []
    for id_ in matcher_string_id:
        matcher_id.append(nlp.vocab.strings[id_])
    for match_id, start, end in raw_matches:
        if end - start < 10 and match_id in matcher_id:
            span = doc[start:end]
            matches.append({'id': match_id, 'span': span})
    return matches


def matched_paragraphs(doc, matcher, matcher_id, section_list):
    matches, l_matched_paragraphs, section_titles = [], [], []
    matches = get_matches(doc, matcher, matcher_id)
    for match in matches:
        section_titles.append(get_section_title(match['span'], section_list))
        l_matched_paragraphs.append(get_paragraph(match['span'], doc))
    matched_paragraphs_tuples = list(zip(section_titles, l_matched_paragraphs))
    unique_matched_paragraphs_tuples = list(dict.fromkeys(matched_paragraphs_tuples).keys())
    return unique_matched_paragraphs_tuples


def create_text_from_paragraph_list(paragraph_tuples):
    tuple_text = []
    for tuple_ in paragraph_tuples:
        tuple_text.append('[{title}] {text}'.format(title=tuple_[0], text=tuple_[1]))
    return ' [...] '.join(tuple_text)
# docs = list(nlp.pipe(df_covid19['body_text'], batch_size=50, n_process=3))
docs = list(nlp.pipe(df_covid19['body_text'], batch_size=50, n_process=1))

# Add spacy doc objects to our covid-19 df
df_covid19['spacy_doc'] = docs
matcher = Matcher(nlp.vocab)
incubation_pattern = [{"LOWER": {'IN': ['incubation', 'latency', 'latent','window']}}, 
                      {"LOWER": 'period'},
                      {'OP':'*'},
                      {'POS': 'NUM'},
                      {'OP':'*'},
                      {"LOWER": {'IN': ['week', 'weeks', 'days','day','months','month']}}]

incubation_pattern2 = [{"LOWER": 'period'},
                       {"LOWER": {'IN': ['of']}},
                       {"LOWER": {'IN': ['incubation', 'latency']}},
                       {'OP':'*'},
                       {'POS': 'NUM'},
                       {'OP':'*'},
                       {"LOWER": {'IN': ['week', 'weeks', 'days','day','month','month']}}]

matcher.add("incubation_matcher", None, incubation_pattern)
matcher.add("incubation_matcher2", None, incubation_pattern2)
incubation_matchers = ['incubation_matcher', 'incubation_matcher2']
relevant_extract = []
for index, row in df_covid19.iterrows():
    mp_tuples = matched_paragraphs(row['spacy_doc'], matcher, incubation_matchers, row['sections'])
    relevant_extract.append(create_text_from_paragraph_list(mp_tuples))

df_covid19['incubation_extract'] = relevant_extract
df_covid19['incubation_extract'] = df_covid19['incubation_extract'].apply(lambda x: np.nan if len(x) == 0 else x)

# Create a dataframe with the results
df_incubation = df_covid19.loc[df_covid19['incubation_extract'].notna(), ['paper_id','title','authors', 'incubation_extract']]
print('Number of papers related to Covid-19 that mention the incubation period: ' + str(df_incubation.shape[0]))
df_incubation.head()
get_matches(df_covid19.loc[129, 'spacy_doc'], matcher, incubation_matchers)
df_incubation.loc[129, 'incubation_extract']
matcher = Matcher(nlp.vocab)
transmission_pattern = [{"LOWER": {'IN': ['transmission', 'transmissibility','reproduction']}},
                        {"LOWER":{'IN' : ['number','rate']}},
                       {'OP':'*'},
                       {'POS': 'NUM'}
                       #,{"LOWER": {'IN': ['percent', '%']}}
                       ]

transmission_pattern2 = [{"LOWER": 'reproductive'}, 
                         {"LOWER": 'rate'},
                         {'OP':'*'},
                         {'POS': 'NUM'}
                         #,{"LOWER": {'IN': ['percent', '%']}}
                        ]


transmission_pattern3 = [{"LOWER": 'environmental'}, 
                         {"LOWER": 'stability'},
                         {'OP':'*'},
                         {'POS': 'NUM'}]


matcher.add("transmission_matcher", None, transmission_pattern)
matcher.add("transmission_matcher2", None, transmission_pattern2)
matcher.add("transmission_matcher3", None, transmission_pattern3)
transmission_matchers = ['transmission_matcher', 'transmission_matcher2','transmission_matcher3']
# Extract relevant paragraphs

relevant_extract = []
for index, row in df_covid19.iterrows():
    mp_tuples = matched_paragraphs(row['spacy_doc'], matcher, transmission_matchers, row['sections'])
    relevant_extract.append(create_text_from_paragraph_list(mp_tuples))

df_covid19['transmission_extract'] = relevant_extract
df_covid19['transmission_extract'] = df_covid19['transmission_extract'].apply(lambda x: np.nan if len(x) == 0 else x)

# Create a dataframe with the results
df_transmission = df_covid19.loc[df_covid19['transmission_extract'].notna(), ['paper_id','title','authors', 'transmission_extract']]


print('Number of papers related to Covid-19 that mention the transmission rate: ' + str(df_transmission.shape[0]))
df_transmission.head()
get_matches(df_covid19.loc[2164, 'spacy_doc'], matcher, transmission_matchers)
df_transmission.loc[2164, 'transmission_extract']
# Pattern for which you want to find matches

mortality_pattern = [{"LOWER": {'IN': ['mortality', 'fatality']}},
                     #{'LOWER': {'IN':['rate','rates']}},
                     {'OP': '*'},
                     {'POS': 'NUM'},
                     {"LOWER": {'IN': ['percent', '%']}}]



matcher.add("mortality_matcher", None, mortality_pattern)
relevant_extract = []
for index, row in df_covid19.iterrows():
    mp_tuples = matched_paragraphs(row['spacy_doc'], matcher, ['mortality_matcher'], row['sections'])
    relevant_extract.append(create_text_from_paragraph_list(mp_tuples))

df_covid19['mortality_extract'] = relevant_extract
df_covid19['mortality_extract'] = df_covid19['mortality_extract'].apply(lambda x: np.nan if len(x) == 0 else x)

# Create a dataframe with the results
df_mortality = df_covid19.loc[df_covid19['mortality_extract'].notna(), ['paper_id','title','authors', 'mortality_extract']]
print('Number of papers related to Covid-19 that mention the mortality rate: ' + str(df_mortality.shape[0]))
df_mortality.head()
get_matches(df_covid19.loc[3047, 'spacy_doc'], matcher, ['mortality_matcher'])
# Inspect some results
df_mortality.loc[3047, 'mortality_extract']
columns_to_include = ['paper_id','abstract', 'title','publication_date']
df = df_covid19[columns_to_include]
for i in range(len(df)):
    if df['abstract'].iloc[i] == "":
        df['abstract'].iloc[i] = np.nan
    if df['title'].iloc[i] == "":
        df['title'].iloc[i] = np.nan
sars_synonyms = [r'\bsars\b',
                 'severe acute respiratory syndrome']
df = tagger(df, sars_synonyms, 'sars')
mers_synonyms = [r'\bmers\b',
                 'middle east respiratory syndrome']
df = tagger(df, mers_synonyms, 'mers')
df = tagger(df, corona_synonyms, 'corona')
corona_synonyms = ['corona', r'\bcov\b']
ards_synonyms = ['acute respiratory distress syndrome',
                 r'\bards\b']
df = tagger(df, ards_synonyms, 'ards')
riskfac_synonyms = [
    'risk factor analysis',
    'cross sectional case control',
    'prospective case control',
    'matched case control',
    'medical records review',
    'seroprevalence survey',
    'syndromic surveillance'
]
df = tagger(df, riskfac_synonyms, 'design_riskfac')
risk_factor_synonyms = ['risk factor',
                        'risk model',
                        'risk by',
                        'comorbidity',
                        'comorbidities',
                        'coexisting condition',
                        'co existing condition',
                        'clinical characteristics',
                        'clinical features',
                        'demographic characteristics',
                        'demographic features',
                        'behavioural characteristics',
                        'behavioural features',
                        'behavioral characteristics',
                        'behavioral features',
                        'predictive model',
                        'prediction model',
                        'univariate', # implies analysis of risk factors
                        'multivariate', # implies analysis of risk factors
                        'multivariable',
                        'univariable',
                        'odds ratio', # typically mentioned in model report
                        'confidence interval', # typically mentioned in model report
                        'logistic regression',
                        'regression model',
                        'factors predict',
                        'factors which predict',
                        'factors that predict',
                        'factors associated with',
                        'underlying disease',
                        'underlying condition']
df = tagger(df, risk_factor_synonyms, 'generic_risk_factors')
age_synonyms = ['median age',
                'mean age',
                'average age',
                'elderly',
                r'\baged\b',
                r'\bold',
                'young',
                'teenager',
                'adult',
                'child'
               ]
df = tagger(df, age_synonyms, 'risk_age')
sex_synonyms = ['sex',
                'gender',
                r'\bmale\b',
                r'\bfemale\b',
                r'\bmales\b',
                r'\bfemales\b',
                r'\bmen\b',
                r'\bwomen\b'
               ]
df = tagger(df, sex_synonyms, 'risk_gender')
bodyweight_synonyms = [
    'overweight',
    'over weight',
    'obese',
    'obesity',
    'bodyweight',
    'body weight',
    r'\bbmi\b',
    'body mass',
    'body fat',
    'bodyfat',
    'kilograms',
    r'\bkg\b', # e.g. 70 kg
    r'\dkg\b'  # e.g. 70kg
]
df = tagger(df, bodyweight_synonyms, 'risk_bodyweight')
smoking_synonyms = ['smoking',
                    'smoke',
                    'cigar', # this picks up cigar, cigarette, e-cigarette, etc.
                    'nicotine',
                    'cannabis',
                    'marijuana']
df = tagger(df, smoking_synonyms, 'risk_smoking')
diabetes_synonyms = [
    'diabet', # picks up diabetes, diabetic, etc.
    'insulin', # any paper mentioning insulin likely to be relevant
    'blood sugar',
    'blood glucose',
    'ketoacidosis',
    'hyperglycemi', # picks up hyperglycemia and hyperglycemic
]
df = tagger(df, diabetes_synonyms, 'risk_diabetes')
hypertension_synonyms = [
    'hypertension',
    'blood pressure',
    r'\bhbp\b', # HBP = high blood pressure
    r'\bhtn\b' # HTN = hypertension
]
df = tagger(df, hypertension_synonyms, 'risk_hypertension')
immunodeficiency_synonyms = [
    'immune deficiency',
    'immunodeficiency',
    r'\bhiv\b',
    r'\baids\b'
    'granulocyte deficiency',
    'hypogammaglobulinemia',
    'asplenia',
    'dysfunction of the spleen',
    'spleen dysfunction',
    'complement deficiency',
    'neutropenia',
    'neutropaenia', # alternate spelling
    'cell deficiency' # e.g. T cell deficiency, B cell deficiency
]
df = tagger(df, immunodeficiency_synonyms, 'risk_immunodeficiency')
cancer_synonyms = [
    'cancer',
    'malignant tumour',
    'malignant tumor',
    'melanoma',
    'leukemia',
    'leukaemia',
    'chemotherapy',
    'radiotherapy',
    'radiation therapy',
    'lymphoma',
    'sarcoma',
    'carcinoma',
    'blastoma',
    'oncolog'
]
df = tagger(df, cancer_synonyms, 'risk_cancer')
chronicresp_synonyms = [
    'chronic respiratory disease',
    'asthma',
    'chronic obstructive pulmonary disease',
    r'\bcopd',
    'chronic bronchitis',
    'emphysema'
]
df = tagger(df, chronicresp_synonyms, 'risk_chronic_respiratory_disease')
asthma_synonyms = ['asthma']
df = tagger(df, asthma_synonyms, 'risk_asthma')
immunity_synonyms = [
    'immunity',
    r'\bvaccin',
    'innoculat'
]
df = tagger(df, immunity_synonyms,'immunity_generic')
climate_synonyms = [
    'climate',
    'weather',
    'humid',
    'sunlight',
    'air temperature',
    'meteorolog', # picks up meteorology, meteorological, meteorologist
    'climatolog', # as above
    'dry environment',
    'damp environment',
    'moist environment',
    'wet environment',
    'hot environment',
    'cold environment',
    'cool environment'
]
df = tagger(df, climate_synonyms,'climate_generic')
df['publication_date']= pd.to_datetime(df['publication_date'])
date_sars = pd.to_datetime('2003-04-01')
date_mers = pd.to_datetime('2012-09-01')
date_covid_19 = pd.to_datetime('2019-12-31')
def tagger_date(row):
    date_sars = pd.to_datetime('2003-04-01')
    date_mers = pd.to_datetime('2012-09-01')
    date_covid_19 = pd.to_datetime('2019-12-31')
    
    if row['publication_date'] < date_sars :
      return 'publication date before SARS'
    if ((row['publication_date'] >= date_sars) & (row['publication_date'] < date_mers)):
      return 'publication date after SARS (but before MERS)'
    if ((row['publication_date'] >= date_mers) & (row['publication_date'] < date_covid_19)):
      return 'publication date after MERS (but before COVID 19)'
    if row['publication_date'] >= date_covid_19:
      return 'publication date after COVID_19'
df['tag_publication_date'] = df.apply (lambda row: tagger_date(row), axis=1)
df = df.drop(columns=['abstract', 'title','publication_date'])
df_covid19 = df_covid19.merge(df,on='paper_id',how='left')
df_mortality = df_mortality.merge(df,on='paper_id',how='left')
df_incubation = df_incubation.merge(df,on='paper_id',how='left')
df_transmission = df_transmission.merge(df,on='paper_id',how='left')
df_covid19.head()
df_incubation.head()
df_transmission.head()
df_mortality.head()