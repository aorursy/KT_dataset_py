# Import packages

import spacy

import pandas as pd

import seaborn as sns

from matplotlib.pyplot import figure

import matplotlib.pyplot as plt

from collections import Counter

from spacy.lang.en.stop_words import STOP_WORDS



# Custom settings

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

sns.set_style("whitegrid")

sns.set_context("talk", font_scale=.9)



# Load data

nlp = spacy.load('en')

df = pd.read_csv('../input/all-injuries-in-cinematography-19142019/movie_injury.csv', index_col = 0)
df.head()
df["Decade"] = [str(x)[0:3] + "0" for x in df["Year"]]



figure(figsize=(10, 5))

ax=sns.countplot(x="Decade", data=df, color="grey");

ax.set_ylabel('Count');
# Add words related to filming to stop words

film_words = ["set","crew","scene","filming","film","actor","director",

              "character","performing","member","extras","shooting"]



for word in film_words:

    nlp.vocab[word].is_stop = True
df_text = nlp(df["Description"].str.cat(sep=' '))



# Remove stop words, punctuation, and spaces

df_words = [token.text.lower() for token in df_text

            if token.is_stop != True 

            and token.is_punct != True 

            and token.text != ' ' 

            and token.text != '  '

            and token.text != '\n' 

            and token.text != '\n\n']



# Create subsets that include only nouns or only verbs

df_nouns = [token.text.lower() for token in df_text if token.is_stop != True and token.pos_ == "NOUN"]

df_verbs = [token.text.lower() for token in df_text if token.is_stop != True and token.pos_ == "VERB"]
# Find the most common words, nouns, and verbs

common_words = pd.DataFrame(Counter(df_words).most_common(20), columns = ["Word", "Frequency"])

common_nouns = pd.DataFrame(Counter(df_nouns).most_common(20), columns = ["Word", "Frequency"])

common_verbs = pd.DataFrame(Counter(df_verbs).most_common(20), columns = ["Word", "Frequency"])
figure(figsize=(16, 6))



plt.subplot(1, 3, 1)

ax=sns.barplot(x=common_words.Frequency, y=common_words.Word, color="grey");

ax.set_title('Most Common Words');

ax.set_ylabel('');



plt.subplot(1, 3, 2)

ax=sns.barplot(x=common_nouns.Frequency, y=common_nouns.Word, color="grey");

ax.set_title('Most Common Nouns');

ax.set_ylabel('');



plt.subplot(1, 3, 3)

ax=sns.barplot(x=common_verbs.Frequency, y=common_verbs.Word, color="grey");

ax.set_title('Most Common Verbs');

ax.set_ylabel('');



plt.tight_layout(pad=1)