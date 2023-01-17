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



g_parties_dict = {"Cs": "../input/Programas/Cs/",

                  "PP": "../input/Programas/PP/",

                  "PSOE": "../input/Programas/PSOE/",

                  "UP": "../input/Programas/UP/", 

                  "VOX": "../input/Programas/VOX/"}



g_parties_color_dict = {"Cs": ["#ffe0b3", "#ff9900"],

                        "PP": ["#b3d9ff", "#0080ff"],

                        "PSOE": ["#ffc2b3", "#ff3300"],

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

    #locale.setlocale(locale.LC_NUMERIC, 'en')

    

    words_per_party = words_per_party.sort_values(by=['Total words'], ascending=False).reset_index(drop=True)



    legends_l = []

    for party in words_per_party['Party']:

        legends_l.append("{} - totales".format(party))

        legends_l.append("{} - útiles".format(party))



    sns.set(color_codes=True)

    sns.set_style("whitegrid")



    figure = plt.figure(figsize=(10, 5))

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

                                 rotation=45,

                                 wrap=True,

                                 #position=(2,0),

                                 visible=False

                )

        )



        startingRadius-=0.55



    # create circle and place onto pie chart

    circle = ax_pie.add_patch(plt.Circle(xy=(0, 0), radius=0.35, facecolor='white'))



    horiz_offset = 1.6

    vert_offset = -0.75



    _ = ax_pie.legend(legends_l,

                 bbox_to_anchor=(horiz_offset, vert_offset),

                 ncol=5)



    _ = ax_pie.set_title('Distribución de palabras (útiles/totales) en el programa electoral', 

                         fontsize=18, x=0.5, y=1.8)



def create_wordcount_stacked_bar(words_per_party):

    words_per_party = words_per_party.sort_values(by=['Total words'], ascending=False).reset_index(drop=True)



    sns.set(color_codes=True)

    sns.set_style("whitegrid")



    figure = plt.figure(figsize=(15, 10))

    ax_1 = figure.add_subplot(1, 1, 1)   



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

    ax_1.set_xlabel('Número de palabras en el programa electoral', fontsize=18)  



    horiz_offset = 0.98

    vert_offset = 1





    _ = ax_1.legend(pie_1 + pie_2, total_words_legends + useful_words_legends, 

                    bbox_to_anchor=(horiz_offset, vert_offset),

                   ncol=2)



party_corpus_df = create_party_programs_corpus_with_sections()

words_per_party = get_word_breakdown(party_corpus_df)



create_wordcount_pie(words_per_party)

create_wordcount_stacked_bar(words_per_party)