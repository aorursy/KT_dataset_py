# Import statements



import pandas as pd

import random

import nltk as nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.probability import FreqDist

import matplotlib.pyplot as plt

from wordcloud import WordCloud
# Load the metal band names



df_metal = pd.read_csv("../input/metal-by-nation/metal_bands_2017.csv", encoding="latin-1")

df_metal.head()
# De-duplicate and tokenize the metal band names



metal_bands = list(df_metal["band_name"])

metal_bands = set(metal_bands)

metal_bands = [word_tokenize(i) for i in metal_bands]

metal_bands[0:6]
# Load the mushrooms names



df_mushrooms = pd.read_csv("../input/english-names-for-fungi/English names for fungi (BMS 2019).csv")

df_mushrooms.head()
# De-duplicate and tokenize the mushroom names



mushrooms = list(df_mushrooms["English Name(s)"])

mushrooms = set(mushrooms)

mushrooms = [word_tokenize(i) for i in mushrooms]

mushrooms[0:6]
# Define the stop words



stop_words = set(stopwords.words('english'))
# Create a list of all terms used in metal band names (drops any non-alphabetical terms, preserves term frequency)



lem = WordNetLemmatizer()



metal_terms = []



for i in range(0, len(metal_bands)):

    for w in metal_bands[i]:

        if w.lower() not in stop_words:

            if w.isalpha():

                w = w.lower()

                w = lem.lemmatize(w)

                metal_terms.append(w)



# Create a set of metal band terms

                

metal_set = set(metal_terms)
# Create a list of all terms used in mushroom names (drops any non-alphabetical terms, preserves term frequency)



mushroom_terms = []



for i in range(0, len(mushrooms)):

    for w in mushrooms[i]:

        if w not in stop_words:

            if w.isalpha():

                w = w.lower()

                w = lem.lemmatize(w)

                mushroom_terms.append(w)



mushroom_set = set(mushroom_terms)
# Print a metal band data summary



print("Number of metal band names: " + str(len(metal_bands)))

print("Number of unique terms in metal band names: " + str(len(metal_set)))

print("Number of metal band term occurances: " + str(len(metal_terms)))



# Print a mushroom data summary



print("Number of mushroom names: " + str(len(mushrooms)))

print("Number of unique terms in mushroom names: " + str(len(mushroom_set)))

print("Number of mushroom name term occurances: " + str(len(mushroom_terms)))
# Calculate term frequencies



metal_freq = FreqDist(metal_terms)

mushroom_freq = FreqDist(mushroom_terms)
# Define a colour function for the word clouds



def colour_select(word, font_size, position, orientation, random_state=10, **kwargs):

    colours = ["#4094e6", "#5f60f2", "#21c8d9", "#00ffcc", "#7f2aff"]

    return random.choice(colours)
# Generate the metal band word cloud



wordcloud = WordCloud(width=800, height=800, 

                      background_color='black', 

                      min_font_size=12,

                      max_font_size=96).generate_from_frequencies(metal_freq)

                        

plt.figure(figsize=(10, 10), facecolor="k", edgecolor='k') 

plt.imshow(wordcloud.recolor(color_func=colour_select, random_state=3), interpolation="bilinear") 

plt.axis("off") 

plt.tight_layout(pad=0) 

plt.show() 
# Generate the mushroom name word cloud



wordcloud = WordCloud(width=800, height=800, 

                      background_color="black", 

                      min_font_size=12,

                      max_font_size=96).generate_from_frequencies(mushroom_freq)

                        

plt.figure(figsize=(10, 10), facecolor="k", edgecolor="k")

plt.imshow(wordcloud.recolor(color_func=colour_select, random_state=3), interpolation="bilinear") 

plt.axis("off") 

plt.tight_layout(pad=0) 

plt.show() 
# Add the most frequent terms for each category to a dataframe



top_metal = pd.DataFrame(metal_freq.most_common(25), columns=["Terms", "Frequency"])

top_mushroom = pd.DataFrame(mushroom_freq.most_common(25), columns=["Terms", "Frequency"])
# Plot term frequency for the most common terms



fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,6))



# Metal band most common terms



ax1.barh(top_metal["Terms"], top_metal["Frequency"], color="#21c8d9")

ax1.set_ylabel("Terms", labelpad=15, size=12)

plt.yticks(size=12)

ax1.set_xlabel("Frequency", labelpad=15, size=12)

ax1.set_yticklabels(top_metal["Terms"], size=12)

plt.xticks(size=12)

ax1.set_title("Most frequent metal band terms", size=14, pad=15)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.spines['bottom'].set_color('#DDDDDD')

ax1.spines['left'].set_color('#DDDDDD')

ax1.tick_params(bottom=False, left=False)



# Mushroom most common terms



ax2.barh(top_mushroom["Terms"], top_mushroom["Frequency"], color="#5f60f2")

ax2.set_ylabel("Terms", labelpad=15, size=12)

plt.yticks(size=12)

ax2.set_xlabel("Frequency", labelpad=15, size=12)

ax2.set_yticklabels(top_mushroom["Terms"], size=12)

plt.xticks(size=12)

ax2.set_title("Most frequent mushroom terms", size=14, pad=15)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

ax2.spines['bottom'].set_color('#DDDDDD')

ax2.spines['left'].set_color('#DDDDDD')

ax2.tick_params(bottom=False, left=False)



fig.tight_layout()

plt.show()
# Identify the set of terms which occur in both categories



shared_terms = mushroom_set.intersection(metal_set)

print("Number of terms common to both sets: " + str(len(shared_terms)))



# Calculate the union of the mushroom and metal band term sets



all_terms = mushroom_set.union(metal_set)

print("Total number of terms across both sets : " + str(len(all_terms)))



# Calculate the Jaccard coefficient



jaccard_coef = len(shared_terms) / len(all_terms)

print("Jaccard coefficient : " + str(round(jaccard_coef, 3)))
# Create a dataframe of the shared terms and their frequencies



df_shared = pd.DataFrame(shared_terms, columns=["Terms"])

df_shared["Metal"], df_shared["Mushroom"] = 0, 0  # initialise count columns



for i in range(0,len(df_shared)):

    df_shared.iloc[i, 1] = int(metal_terms.count(df_shared.iloc[i, 0]))

    df_shared.iloc[i, 2] = int(mushroom_terms.count(df_shared.iloc[i, 0]))



# Adjusted term frequencies (accounting for the different dataset sizes by looking at frequency per 1,000 terms)    

 

df_shared["Metal"] = (df_shared["Metal"] / len(metal_terms)) * 1000

df_shared["Mushroom"] = (df_shared["Mushroom"] / len(mushroom_terms)) * 1000



df_shared.describe()
# Generate boxplots to understand term frequency distributions



ax = df_shared.plot(kind="box", legend=False, color="#4094e6", vert=0, figsize=(8,4))

ax.set_xlabel("Frequency (per 1,000 terms)", labelpad=15, size=12)

plt.yticks(size=12)

plt.xticks(size=12)

ax.set_title("Adjusted term frequency distributions", size=16, pad=15)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_color('#DDDDDD')

ax.spines['left'].set_color('#DDDDDD')

ax.tick_params(bottom=False, left=False)

plt.tight_layout()

plt.show()
# Select terms which occur at least 0.5 times per 1,000 terms



most_common = []



for i in range(0,len(df_shared)):

    if (df_shared.iloc[i, 1] >= 0.5) and (df_shared.iloc[i, 2] >= 0.5):

            most_common.append(list(df_shared.iloc[i,0:3]))



most_common = pd.DataFrame(most_common, columns=["Terms", "Metal", "Mushroom"])
# Plot the terms which occur at least three times in each list 



ax = most_common.plot(x="Terms", y=["Metal", "Mushroom"], 

                    kind="barh", figsize=(8,12), width=0.7, 

                    color=["#5f60f2", "#21c8d9"])

ax.set_ylabel("Terms", labelpad=15, size=14)

plt.yticks(size=12)

ax.set_xlabel("Frequency (per 1,000 terms)", labelpad=15, size=14)

plt.xticks(size=12)

ax.set_title("Most common shared terms", size=16, pad=15)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_color('#DDDDDD')

ax.spines['left'].set_color('#DDDDDD')

ax.tick_params(bottom=False, left=False)

plt.show()
print(str(list(df_shared["Terms"])))