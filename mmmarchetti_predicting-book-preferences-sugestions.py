# Import library

import glob



# The books files are contained in this folder

folder = "../input/"



# List all the .txt files and sort them alphabetically

files = glob.glob(folder + '*.txt')

# ... YOUR CODE FOR TASK 1 ...

files.sort()

print(files)
# Import libraries

import re, os



# Initialize the object that will contain the texts and titles

txts = []

titles = []



for n in files:

    # Open each file

    f = open(n, encoding='utf-8-sig')

    

    # Remove all non-alpha-numeric characters

    data = re.sub('[\W_]+', ' ', f.read())

    

    # Store the texts and titles of the books in two separate lists

    txts.append(data)

    titles.append(os.path.basename(n).replace(".txt", ""))



# Print the length, in characters, of each book

[len(t) for t in txts]
# Browse the list containing all the titles

for i in range(len(titles)):

    # Store the index if the title is "OriginofSpecies"

    if titles[i] == "OriginofSpecies":

        ori = i



# Print the stored index

print(ori)
# Define a list of stop words

stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())



# Convert the text to lower case 

txts_lower_case = [t.lower() for t in txts]



# Transform the text into tokens 

txts_split = [t.split() for t in txts_lower_case]



# Remove tokens which are part of the list of stop words

texts = [[word for word in txt if word not in stoplist] for txt in txts_split]



# Print the first 20 tokens for the "On the Origin of Species" book

print(texts[15][:20])
import pickle



# Load the stemmed tokens list from the pregenerated pickle file

texts_stem = texts



# Print the 20 first stemmed tokens from the "On the Origin of Species" book

print(texts_stem[15][:20])
# Load the functions allowing to create and use dictionaries

from gensim import corpora



# Create a dictionary from the stemmed tokens

dictionary = corpora.Dictionary(texts_stem)





# Create a bag-of-words model for each book, using the previously generated dictionary

bows = [dictionary.doc2bow(t) for t in texts_stem]



# Print the first five elements of the On the Origin of species' BoW model

print(bows[15][:5])
# Import pandas to create and manipulate DataFrames

import pandas as pd



# Convert the BoW model for "On the Origin of Species" into a DataFrame

df_bow_origin = pd.DataFrame(bows[ori])



# Add the column names to the DataFrame

df_bow_origin.columns = ['index', 'occurrences']



# Add a column containing the token corresponding to the dictionary index

df_bow_origin['token'] = df_bow_origin['index'].apply(lambda x: dictionary[x])



# Sort the DataFrame by descending number of occurrences and print the first 10 values

df_bow_origin = df_bow_origin.sort_values('occurrences', ascending=False)

df_bow_origin.head(10)
# Load the gensim functions that will allow us to generate tf-idf models

from gensim.models import TfidfModel



# Generate the tf-idf model

model = TfidfModel(bows)



# Print the model for "On the Origin of Species"

model[bows[ori]][:5]
# Convert the tf-idf model for "On the Origin of Species" into a DataFrame

df_tfidf = pd.DataFrame(model[bows[ori]])



# Name the columns of the DataFrame id and score

df_tfidf.columns = ['id', 'score']



# Add the tokens corresponding to the numerical indices for better readability

df_tfidf['token'] = df_tfidf['id'].apply(lambda x: dictionary[x])



# Sort the DataFrame by descending tf-idf score and print the first 10 rows.

df_tfidf = df_tfidf.sort_values('score', ascending=False)

df_tfidf.head(10)
# Load the library allowing similarity computations

from gensim import similarities



# Compute the similarity matrix (pairwise distance between all texts)

sims = similarities.MatrixSimilarity(model[bows])



# Transform the resulting list into a dataframe

sim_df = pd.DataFrame(list(sims))



# Add the titles of the books as columns and index of the dataframe

sim_df.columns = titles

sim_df.index = titles



# Print the resulting matrix

sim_df
# This is needed to display plots in a notebook

%matplotlib inline



# Import libraries

import matplotlib.pyplot as plt



# Select the column corresponding to "On the Origin of Species" and 

v = sim_df['OriginofSpecies']



# Sort by ascending scores

v_sorted = v.sort_values()



# Plot this data has a horizontal bar plot

v_sorted.plot.barh(x='lab', y='val', rot=0).plot()



# Modify the axes labels and plot title for a better readability

plt.xlabel("Score")

plt.ylabel("Book")

plt.title("Similarity")
# Import libraries

from scipy.cluster import hierarchy



# Compute the clusters from the similarity matrix,

# using the Ward variance minimization algorithm

Z = hierarchy.linkage(sims, 'ward')



# Display this result as a horizontal dendrogram

hierarchy.dendrogram(Z, leaf_font_size=8, labels=sim_df.index, orientation='left')