!mkdir Charts Charts/bow Charts/tfidf
!pip install spacy

!python -m spacy download es_core_news_sm

!python -m spacy link es_core_news_sm es
import pandas as pd

import string



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

from os import listdir

from os.path import isfile, join



import spacy

from spacy.tokens import Doc



import re



import plotly

import plotly.graph_objects as go



import scattertext as st

from scattertext import CorpusFromPandas, produce_scattertext_explorer



from IPython.display import IFrame

from IPython.core.display import display, HTML



import nltk

from nltk.tokenize import RegexpTokenizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from nltk.corpus import stopwords

nltk.download('stopwords')

nltk.download('wordnet')



# Function to return a list of stop words to consider

def create_stop_words():

    # We create a stop word list

    stops = set(stopwords.words("spanish"))



    # We define individual numbers and letters as stop words

    all_letters_numbers = string.digits + string.ascii_letters



    stops = stops.union(["señor", "abascal", "casado", 

                         "partido", "popular", "vox", "señorías", "moción",

                         "censura"

                        ])  # add some stopwords

    

    stops = stops.union(list(all_letters_numbers))

    

    return stops



g_parties_dict = {

                  "Casado": "../input/mocion/Casado/",

                  "Abascal": "../input/mocion/Abascal/"

}



g_parties_color_dict = {

                        "Casado": ["#b3d9ff", "#0080ff", "interpolateBlues"],

                        "Abascal": ["#c2f0c2", "#33cc33", "interpolateGreens"]}



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



def get_word_breakdown(party_corpus_df):

    

    def get_num_tokens(values):

        flat_list = [item for sublist in values for item in sublist]

        return len(flat_list)





    total_words = party_corpus_df.groupby("Party").Tokenized_raw_section_text.agg(get_num_tokens)  

    useful_words = party_corpus_df.groupby("Party").Tokenized_filtered_section_text.agg(get_num_tokens)

    

    words_per_party = pd.DataFrame(total_words).assign(x=useful_words.values)

    words_per_party.reset_index(level=0, inplace=True)

    words_per_party.columns = ['Party', 'Total words', 'Useful words']

    

    return words_per_party



def create_wordcount_pie(words_per_party):



    import locale

    #locale.setlocale(locale.LC_NUMERIC, 'es_ES')

    

    words_per_party = words_per_party.sort_values(by=['Total words'], ascending=False).reset_index(drop=True)



    legends_l = []

    for party in words_per_party['Party']:

        legends_l.append("{} - totales".format(party))

        legends_l.append("{} - útiles".format(party))



    sns.set(color_codes=True)

    sns.set_style("whitegrid")



    figure = plt.figure(figsize=(7, 7))

    ax_pie = figure.add_subplot(1, 1, 1)   



    startingRadius = 1.7 + (0.3* (len(words_per_party)-1))

    for index, row in words_per_party.iterrows():



        party_name = row["Party"]

        total_words = row["Total words"]

        useful_words = row["Useful words"]



        useless_words = row["Total words"] - row["Useful words"]



        party_colors = g_parties_color_dict[party_name]



        useful_words_pct = (useful_words / total_words) * 100

        

        localized_useful_words_pct = locale.format_string('%.2f', useful_words_pct, grouping=True)



        label_text = "{}: {}% útiles".format(party_name, localized_useful_words_pct)

        label_text_2 = "{}".format(total_words)



        remaining_pct = 100 - useful_words_pct



        donut_sizes = [remaining_pct, useful_words_pct]



        ax_pie.text(-0.48, startingRadius - 0.26, label_text, 

                 horizontalalignment='left', verticalalignment='center',

                    fontsize=12, fontweight='bold',color='black')



        ax_pie.text(0.1 - startingRadius, 0, useful_words, 

                 horizontalalignment='left', verticalalignment='center',

                    fontsize=12, fontweight='normal',color='black', rotation = 45)



        ax_pie.text(-0.3 + startingRadius, 0, total_words, 

                 horizontalalignment='left', verticalalignment='center',

                    fontsize=12, fontweight='normal',color='black', rotation = 45)



        donut = ax_pie.pie(donut_sizes, 

                radius = startingRadius, 

                startangle = 270, 

                colors=party_colors,

                labels = [total_words, useful_words],

                rotatelabels = False,

                counterclock = True,

                labeldistance = 0.95 - 0.025*index,

                wedgeprops = {"edgecolor": "white", 'linewidth': 1},

                textprops = dict(rotation_mode = 'default', 

                                 va='center', 

                                 ha='center', 

                                 #rotation=45,

                                 wrap=True,

                                 #position=(2,0),

                                 visible=False

                )

        )



        startingRadius-=0.80



    # create circle and place onto pie chart

    circle = ax_pie.add_patch(plt.Circle(xy=(0, 0), radius=0.35, facecolor='white'))



    horiz_offset = 0.8

    vert_offset = -0.30



    _ = ax_pie.legend(legends_l,

                 bbox_to_anchor=(horiz_offset, vert_offset),

                 ncol=2)



    _ = ax_pie.set_title('Distribución de palabras (útiles/totales) en el discurso', 

                         fontsize=18, x=0.5, y=1.35)



def create_wordcount_stacked_bar(words_per_party):

    words_per_party = words_per_party.sort_values(by=['Total words'], ascending=False).reset_index(drop=True)



    sns.set(color_codes=True)

    sns.set_style("whitegrid")



    figure = plt.figure(figsize=(15, 5))

    ax_1 = figure.add_subplot(1, 1, 1)  

    ax_1.set_ymargin(0.15)





    parties = words_per_party["Party"]



    total_words_colors = [g_parties_color_dict[key][0] for key in parties.values]

    useful_words_colors = [g_parties_color_dict[key][1] for key in parties.values]



    total_words_legends = ["{} - no útiles".format(key) for key in parties.values]

    useful_words_legends = ["{} - útiles".format(key) for key in parties.values]



    useless_words = words_per_party["Total words"] - words_per_party["Useful words"]





    pie_1 = ax_1.barh(parties, words_per_party["Total words"], 

             color=total_words_colors)



    pie_2 = ax_1.barh(parties, words_per_party["Useful words"], 

             color=useful_words_colors)  



    for i, v in enumerate(words_per_party["Total words"]):

        plt.text(v/1.5, i + .45, str(useless_words[i]), color=total_words_colors[i], fontweight='bold', fontsize=16)

        plt.text(v/1.0, i + .05, str(v), color='black', fontweight='normal', fontsize=13)



    for i, v in enumerate(words_per_party["Useful words"]):    

        plt.text(v/4, i + .45, str(v), color=useful_words_colors[i], fontweight='bold', fontsize=16)





    # we also need to switch the labels

    ax_1.set_xlabel('Número de palabras en el discurso', fontsize=18)  



    horiz_offset = 0.98

    vert_offset = 1





    _ = ax_1.legend(pie_1 + pie_2, total_words_legends + useful_words_legends, 

                    bbox_to_anchor=(horiz_offset, vert_offset),

                   ncol=2)







def merge_party_sections(party_corpus_df):

    corpus_groupby_party = party_corpus_df.groupby("Party", as_index=False)



    party_aggregated_sections = corpus_groupby_party["Section_text"].apply(lambda x: ' '.join(x))



    party_sections_df = pd.DataFrame(data = {"Section_text": [i[1] for i in party_aggregated_sections.values], 

                                             "Party": list(corpus_groupby_party["Party"].groups.keys())})   

    

    return party_sections_df



def create_corpora_metric(corpora, doc_names, metric, stop, ngram = 1, num_freq_words=25, num_freq_words_section = 5, 

                     debug = False): 

    

    if metric == "bow":

        vectorizer = CountVectorizer(stop_words = stop, ngram_range = (ngram, ngram), 

                                 preprocessor = num_and_short_word_preprocessor, tokenizer=custom_tokenizer)

    elif metric =="tf-idf":

        vectorizer = TfidfVectorizer(stop_words = stop, ngram_range = (ngram, ngram), 

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

                      standard_sizeref = True,

                      title = 'Palabras con mayor score por partido', 

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

            sizeref = 0.01 * max(index_values)

            hover_text = ["{:.2f}%".format(index_value) for index_value in index_values]

        else:

            if standard_sizeref:

                sizeref = 1

            else:

                sizeref = 0.01 * max(index_values)

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

        complete_file_name = '{}/{}/{}.html'.format(g_charts_root_dir, sub_dir, file_name)

        plotly.offline.plot(fig, filename = complete_file_name, auto_open=False)

        

    fig.show()



# Read files

party_corpus_df = create_party_programs_corpus_with_sections()



# Word count analysis

words_per_party = get_word_breakdown(party_corpus_df)



create_wordcount_pie(words_per_party)

create_wordcount_stacked_bar(words_per_party)



# BOW analysis

party_corpus_df = create_party_programs_corpus_with_sections()

bow_by_party = create_metric_dict_by_party(party_corpus_df, metric = "bow", ngram = 1, 

                                           num_freq_words=15, num_freq_words_section = 5)



two_gram_bow_by_party = create_metric_dict_by_party(party_corpus_df, metric = "bow", ngram = 2, 

                                                 num_freq_words=10, 

                                                 num_freq_words_section = 10)



all_party_words = bow_by_party['All']

all_party_2_words = two_gram_bow_by_party['All']



show_df_as_bubble(all_party_words, num_words = 15, normalize = False,

                  title = 'Palabras más usadas por político',ytitle = 'Político',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_partidos')





show_df_as_bubble(all_party_2_words, num_words = 15, normalize = True,

                  title = 'Binomios de palabras por político', 

                  xtitle = 'Binomios de palabras', 

                  ytitle = 'Político',

                  to_file = True, sub_dir = 'bow', file_name = 'bow_2_partidos')



# TF-IDF analysis



td_idf_by_party = create_metric_dict_by_party(party_corpus_df, metric = "tf-idf", ngram = 1, 

                                           num_freq_words=15, num_freq_words_section = 15)



two_gram_tf_idf_by_party = create_metric_dict_by_party(party_corpus_df, metric = "tf-idf", ngram = 2, 

                                                 num_freq_words=10, 

                                                 num_freq_words_section = 10)



tdidf_all_party_words = td_idf_by_party['All']

tdidf_all_party_2_words = two_gram_tf_idf_by_party['All']



show_df_as_bubble(tdidf_all_party_words, num_words = 250, normalize = False, standard_sizeref = False,

                  title = 'Palabras con mayor score por político', ytitle = 'Político',

                  to_file = True, sub_dir = 'tfidf', file_name = 'tfidf_partidos')



show_df_as_bubble(tdidf_all_party_2_words, num_words = 250, normalize = False, standard_sizeref = False,

                  title = 'Binomios de palabras con mayor score por político', ytitle = 'Político',

                  to_file = True, sub_dir = 'tfidf', file_name = 'tfidf_2_partidos')