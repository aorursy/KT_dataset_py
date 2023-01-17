%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



# More "not interesting" words:

ADDITIONAL_STOPWORDS = ["im", "know", "dont", "will", "got", "now", "nan", "ive", "whats", "hes", 

                        "shes", "its", "cant", "mr", "mrs", "ill", "let", "lets", "youre", "oh", "one",

                        "thats", "theres", "ye", "go", "day", "say", "didnt", "wont", "aw", "uh","ah", "oh","two",

                        "isnt","ol", "youll","yes", "well", "us", "see", "hey", "ho", "look","call", "gonna", "youve",

                        "us", "take"]
df_char_index = pd.read_csv("../input/simpsons_characters.csv")

print("Nb. of distinct characters: %d " % df_char_index.shape[0])
custom_char_index = {"Maggie": 105, "Marge": 1, "Bart": 8, "Lisa": 9, "Moe": 17,  "Seymour": 3,

                     "Ned": 11, "Grampa": 31, "Wiggum": 71, "Milhouse": 25, "Smithers": 14,

                     "Nelson": 101, "Edna": 40, "Selma": 22, "Barney": 18, "Patty": 10, "Martin": 38,

                     "Todd": 5, "Rod": 121, "Homer": 2, "Cletus": 1413, "Gil": 2369,

                     "Moleman": 963, "Duffman": 2277, "Apu": 208, "Burns": 15, "Dr. Nick": 349,

                     "Dr. Hibbert": 332, "Sideshow Bob": 153,"Krusty": 139, "Fat Tony": 568, "Snake": 518,

                     "Ralph": 119}
data_script_lines = pd.read_csv("../input/simpsons_script_lines.csv",

                    error_bad_lines=False,

                    warn_bad_lines=False,

                    low_memory=False)
def draw_wc(character, data, add_stopwds):

    """

    Draw a nice wordcloud representing the most frequent words for a given character.

    

    Parameters

    ----------

    character (str): the character name (must be a key from custom_char_index)

    data (DataFrame): the script lines data

    add_stopwds (list): additional stopwords for pre-processing

    

    Returns

    -------

    wc (WordCloud)

    

    """

    # Get all the lines from the character of interest:

    chosen_char_id = custom_char_index[character]

    df_charac = data[data["character_id"]==str(chosen_char_id)]

    charac_lines = list(df_charac["normalized_text"].values.astype(str))

    # Transform into one big string:

    charac_lines_one_str = ' '.join(charac_lines)

    # Build the stopwords set:

    stopwords = set(STOPWORDS)

    for w in add_stopwds:

        stopwords.add(w)

    # Instanciate the Wordcloud object:

    wc = WordCloud(background_color="black",

                   max_words=200,

                   stopwords=stopwords,

                   relative_scaling=0.5,

                   width=500,

                   height=350)

    # Generate the wordcloud using the big string:

    wc.generate(charac_lines_one_str)

    return wc

    
def plot_wordcloud(character):

    char_wc = draw_wc(character, data_script_lines, ADDITIONAL_STOPWORDS)

    fig, ax = plt.subplots(figsize=(8,8))

    plt.imshow(char_wc)

    plt.title(character, fontsize=32)

    plt.axis("off")
plot_wordcloud("Homer")

plot_wordcloud("Marge")
plot_wordcloud("Bart")

plot_wordcloud("Lisa")
plot_wordcloud("Nelson")

plot_wordcloud("Martin")
plot_wordcloud("Moe")

plot_wordcloud("Barney")

plot_wordcloud("Duffman")
plot_wordcloud("Moleman")

plot_wordcloud("Gil")
plot_wordcloud("Burns")

plot_wordcloud("Smithers")
plot_wordcloud("Sideshow Bob")

plot_wordcloud("Snake")

plot_wordcloud("Fat Tony")
plot_wordcloud("Ned")

plot_wordcloud("Rod")

plot_wordcloud("Todd")
plot_wordcloud("Seymour")

plot_wordcloud("Edna")
plot_wordcloud("Dr. Nick")

plot_wordcloud("Dr. Hibbert")
plot_wordcloud("Ralph")