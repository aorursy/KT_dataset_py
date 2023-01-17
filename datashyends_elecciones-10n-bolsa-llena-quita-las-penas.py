!pip install spacy

!python -m spacy download es_core_news_sm

!python -m spacy link es_core_news_sm es
import pandas as pd

import string



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import plotly

import plotly.graph_objects as go



import os

from os import listdir

from os.path import isfile, join



import spacy

from spacy.tokens import Doc



import re



from sklearn.feature_extraction.text import CountVectorizer



import nltk

from nltk.tokenize import RegexpTokenizer



from nltk.corpus import stopwords

nltk.download('stopwords')

nltk.download('wordnet')



# Function to return a list of stop words to consider

def create_stop_words():

    # We create a stop word list

    stops = set(stopwords.words("spanish"))



    # We define individual numbers and letters as stop words

    all_letters_numbers = string.digits + string.ascii_letters



    stops = stops.union([])  # add some stopwords

    

    stops = stops.union(list(all_letters_numbers))

    

    return stops



g_parties_dict = {"Cs": "../input/programas10n/Programas/Cs/",

                  "PP": "../input/programas10n/Programas/PP/",

                  "PSOE": "../input/programas10n/Programas/PSOE/",

                  "UP": "../input/programas10n/Programas/UP/", 

                  "VOX": "../input/programas10n/Programas/VOX/"}



g_parties_color_dict = {"Cs": ["#ffd9b3", "#ff8000"],

                        "PP": ["#b3d9ff", "#0080ff"],

                        "PSOE": ["#ffb3b3", "#ff0000"],

                        "UP": ["#e6b3ff", "#aa00ff"],

                        "VOX": ["#c2f0c2", "#33cc33"]}



g_nlp = spacy.load('es')

g_stop_words = create_stop_words() | g_nlp.Defaults.stop_words

g_not_category_name='Resto de partidos'

g_charts_root_dir = './Charts'



def list_dirs(directory):

    """Returns all directories in a given directory

    """

    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]





def list_files(directory, ext):

    """Returns all files in a given directory

    """

    onlyfiles = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(ext)]

    return sorted(onlyfiles)



# Function to remove numbers and small words (1 or 2 letters) from a document

def num_and_short_word_preprocessor(tokens):

    # Regular expression for numbers

    no_numbers = re.sub('(\d)+', '', tokens.lower())

    

    # Regular expression for 1-letter and 2-letter words

    no_short_words = re.sub(r'\b\w{1,2}\b', '', no_numbers)

    

    return no_short_words



# Function to tokenize a document wihthout using stopwords

def custom_tokenizer(doc):

    word_tokenizer = RegexpTokenizer(r'\w+')

    tokens = word_tokenizer.tokenize(doc)



    return tokens



# Function to tokenize a document using stopwords

def custom_tokenizer_filtering_stopwords(doc):

    word_tokenizer = RegexpTokenizer(r'\w+')

    tokens = word_tokenizer.tokenize(doc)



    filtered_tokens = [token for token in tokens if token not in g_stop_words]

    

    return filtered_tokens



def create_Doc(row):

    doc = Doc(g_nlp.vocab, words=row)

    doc.is_parsed = True

    

    return doc



def create_party_programs_corpus_with_sections():



    all_party_program_section_name_l = []

    all_party_corpus_lc = []

    all_party_name = list()



    for party_name, party_program_location in g_parties_dict.items():

        # List of files and their corresponding names that make up the party program

        party_program_section_file_l = list_files(party_program_location, ".txt")

        party_program_section_name_l = list(map(lambda x: os.path.split(x)[1], party_program_section_file_l))



        # List that contains the program document corpus

        party_program_corpus = list()



        # Fill out the corpus

        for section_file_name in party_program_section_file_l:



            with open(section_file_name, 'r', encoding="utf8") as section_file:

                section_file_data = section_file.read().replace('\n', '. ')



            party_program_corpus.append(section_file_data)

            all_party_name.append(party_name)



        # Clean up sections

        party_corpus_lc = [section.lower() for section in party_program_corpus] # to lowercase

        party_corpus_lc = [re.sub("(\d)+", '', section) for section in party_corpus_lc] # no numbers

        party_corpus_lc = [re.sub("[€º”—“«»>•‘’!""#$%&'()*+,-./:;?@[\]^_`{|}~]+", ' ', 

                                  section) for section in party_corpus_lc] # no strange characters

        party_corpus_lc = [re.sub('\s\s+', ' ', section) for section in party_corpus_lc] # no multiple spaces

        party_corpus_lc = [re.sub(r'\b\w{1}\b', '', section) for section in party_corpus_lc]



        all_party_corpus_lc += party_corpus_lc



        # Cleanup section names

        all_party_program_section_name_l += party_program_section_name_l

        all_party_program_section_name_l = [re.sub(".txt", '', 

                                                section_name) for section_name in all_party_program_section_name_l]

        all_party_program_section_name_l = [re.sub("(\d)+[-]", '',

                                                section_name) for section_name in all_party_program_section_name_l]  

        

        

    token_raw_party_section = [custom_tokenizer(section) for section in all_party_corpus_lc]

    token_filtered_party_section = [custom_tokenizer_filtering_stopwords(section) for section in all_party_corpus_lc]

    

    # Create DataFrame

    party_corpus_df = pd.DataFrame(data = { 

                                    "Party": all_party_name,

                                    "Section_name": all_party_program_section_name_l,

                                    "Section_text": all_party_corpus_lc,

                                    "Tokenized_raw_section_text": token_raw_party_section, 

                                    "Tokenized_filtered_section_text": token_filtered_party_section, 

                                    })



    parsed = party_corpus_df.Tokenized_filtered_section_text.apply(create_Doc)



    party_corpus_df["Parsed_section_text"] = parsed



    return party_corpus_df



def merge_party_sections(party_corpus_df):

    corpus_groupby_party = party_corpus_df.groupby("Party", as_index=False)



    party_aggregated_sections = corpus_groupby_party["Section_text"].apply(lambda x: ' '.join(x))



    party_sections_df = pd.DataFrame(data = {"Section_text": party_aggregated_sections.values, 

                                             "Party": list(corpus_groupby_party["Party"].groups.keys())})   

    

    return party_sections_df



def create_corpora_metric(corpora, doc_names, metric, stop, ngram = 1, num_freq_words=25, num_freq_words_section = 5, 

                     debug = False): 

    

    if metric == "bow":

        vectorizer = CountVectorizer(stop_words = stop, ngram_range = (ngram, ngram), 

                                 preprocessor = num_and_short_word_preprocessor, tokenizer=custom_tokenizer)

    else:

        raise Exception('Metric not supported')

    

    vec = vectorizer.fit(corpora)

    doc_term_mat = vec.transform(corpora)

    

    # START: Calculate the most frequent words per section (using word ID)

    freq_words_set = set()

    for section in range(len(corpora)):

        section_info = doc_term_mat[section,]

        

        freq_words_section = [(idx, section_info[0, idx]) for word, idx in vec.vocabulary_.items()]

        sorted_freq_words_section = sorted(freq_words_section, key = lambda x: x[1], reverse=True)

        

        top_freq_words_section = [word[0] for word in sorted_freq_words_section]

        

        freq_words_set.update(top_freq_words_section[0:num_freq_words_section])

        

    # END

     

    # START: Calculate the most frequest words overall (using word ID)

    all_sections_info = doc_term_mat.sum(axis=0)

    all_freq_words_section = [(idx, all_sections_info[0, idx]) for word, idx in vec.vocabulary_.items()]

    all_sorted_freq_words_section = sorted(all_freq_words_section, key = lambda x: x[1], reverse=True)

    

    all_top_freq_words_section = [word[0] for word in all_sorted_freq_words_section]

    freq_words_set.update(all_top_freq_words_section[0:num_freq_words])

    # END

        

    

    freq_words_key_l = list(freq_words_set) # list of most frequent word IDs

    reverse_dictionary = {v: k for k, v in vec.vocabulary_.items()} # dictionary ID: word

    freq_words_value_l = [reverse_dictionary[word_key] for word_key in freq_words_key_l] # list of most frequent words

    

    #DataFrame including the most frequent words per section and overall

    freq_words_df = pd.DataFrame((doc_term_mat[:,freq_words_key_l]).todense(), 

                                  columns=freq_words_value_l, 

                                  index=doc_names)

    

    return freq_words_df



def create_metric_dict_by_party(party_corpus_df, metric = "bow", 

                                ngram = 1, num_freq_words=10, num_freq_words_section = 5):

    

    parties = party_corpus_df.Party.unique()

    metric_by_party = dict()



    # METRIC (BOW or TF-IDF) for individual parties

    for party in parties:

        party_corpora = party_corpus_df[party_corpus_df.Party == party]



        party_section_texts = party_corpora.Section_text.tolist()

        party_section_names = party_corpora.Section_name.tolist()



        freq_words_df = create_corpora_metric(party_section_texts, 

                                                        party_section_names, 

                                                        metric,

                                                        g_stop_words, 

                                                        ngram = ngram, 

                                                        num_freq_words=num_freq_words, 

                                                        num_freq_words_section = num_freq_words_section)

        

        metric_by_party[party] = freq_words_df

    

    # METRIC (BOW or TF-IDF) for the overall corpora including all parties    

    party_sections_df = merge_party_sections(party_corpus_df)

    

    overall_section_texts = party_sections_df.Section_text.tolist()

    overall_section_names = party_sections_df.Party.tolist()

    

    overall_freq_words_df = create_corpora_metric(overall_section_texts, 

                                                        overall_section_names, 

                                                        metric,

                                                        g_stop_words, 

                                                        ngram = ngram, 

                                                        num_freq_words=num_freq_words, 

                                                        num_freq_words_section = num_freq_words_section)

    

    metric_by_party["All"] = overall_freq_words_df

        

    return metric_by_party



def normalize_word_df(word_df):

    norm_word_df = word_df.copy()

    total_per_party = norm_word_df.sum(axis=1)

    grand_total = sum(total_per_party)



    for party in norm_word_df.index:

        norm_word_df.loc[party] = (norm_word_df.loc[party] / total_per_party[party]) * 100

        

    return norm_word_df



def show_df_as_bubble(word_df,

                      num_words = 2,

                      normalize = False,

                      title = 'Palabras más frecuentes por partido', 

                      xtitle = 'Palabras', 

                      ytitle = 'Partido',

                      sub_dir = 'bow', 

                      file_name = 'bow_bubble',

                      to_file = False):

   

    if normalize:

        word_df = normalize_word_df(word_df)

        title = title + " (normalizado)"



    # Keep the top 'num_words' words

    word_df.reindex(word_df.sum().sort_values(ascending=False).index, axis=1)

    

    num_words = min(num_words, word_df.shape[1])

    word_df = word_df.iloc[:, :num_words]

    

    # Sort in alphabetical order

    word_df = word_df.reindex(sorted(word_df.columns), axis=1)



    # Get the word labels

    word_l = word_df.columns.values

    num_words = len(word_l)

    # List that contains every bubble row. Each element is a party.

    bubble_data = []



    xs = list(range(word_df.shape[1]))



    for num_row in range(len(word_df.values)):

        index_name = word_df.index[num_row]

        index_color = g_parties_color_dict[index_name][1]

        index_values = word_df.values[num_row]

        

        if normalize:

            sizeref = 0.02 * max(index_values)

            hover_text = ["{:.2f}%".format(index_value) for index_value in index_values]

        else:

            sizeref = 1

            hover_text = index_values

        

        section_bubble = go.Scatter(

            x = word_l, 

            y = [num_row + 1] * num_words,

            name = index_name,

            showlegend = True,

            mode = 'markers',

            hovertext = hover_text,

            hoverinfo = "x+text",

            marker_size = index_values,

            marker = dict(color = index_color, sizeref = sizeref),

        )

        bubble_data.append(section_bubble)



    fig = go.Figure(data=bubble_data)

    

    fig.update_layout(

        title = dict(text = title, xanchor = 'center', x = 0.5, font = dict(size=20)),

        yaxis = dict(

            tickmode = 'array',

            tickvals = list(range(1, num_words + 1)),

            ticktext = word_df.index

        ),

        legend = dict(itemsizing = 'constant')

    )

    

    fig.update_xaxes(tickangle=315, title_text=xtitle, title_font=dict(size=18))

    fig.update_yaxes(title_text=ytitle, title_font=dict(size=18))

    

    if to_file:

        dest_dict = os.path.join(g_charts_root_dir, sub_dir)

        if not os.path.exists(dest_dict):

                        os.makedirs(dest_dict, exist_ok=True)

                

        complete_file_name = '{}/{}/{}.html'.format(g_charts_root_dir, sub_dir, file_name)

        plotly.offline.plot(fig, filename = complete_file_name, auto_open=False)

        

    fig.show()

    

party_corpus_df = create_party_programs_corpus_with_sections()



bow_by_party = create_metric_dict_by_party(party_corpus_df, metric = "bow", ngram = 1, 

                                           num_freq_words=20, num_freq_words_section = 5)



two_gram_bow_by_party = create_metric_dict_by_party(party_corpus_df, metric = "bow", ngram = 2, 

                                                 num_freq_words=10, 

                                                 num_freq_words_section = 5)



all_party_words = bow_by_party['All']

all_party_2_words = two_gram_bow_by_party['All']



show_df_as_bubble(all_party_words, num_words = 100, normalize = False,

                  title = 'Palabras más usadas por partido',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_partidos')



show_df_as_bubble(all_party_words, num_words = 100, normalize = True,

                  title = 'Palabras más usadas por partido',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_partidos_norm')



show_df_as_bubble(all_party_2_words, num_words = 100, normalize = False,

                  title = 'Binomios de palabras más usados por partido', 

                  xtitle = 'Binomios de palabras', 

                  ytitle = 'Partido',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_2_partidos')



show_df_as_bubble(all_party_2_words, num_words = 100, normalize = True,

                  title = 'Binomios de palabras más usados por partido', 

                  xtitle = 'Binomios de palabras', 

                  ytitle = 'Partido',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_2_partidos_norm')