#Import python libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import re

from collections import defaultdict
# List the files in the directory

import os

print(os.listdir("../input/"))
#Import the US based dataset

data_us = pd.read_csv("../input/data-scientist-job-market-in-the-us/alldata.csv")

#Import the UK based dataset

data_uk = pd.read_csv("../input/50000-job-board-record-from-reed-uk/reed_uk.csv")
#Select only the position and the associated description for the US based dataset

select_data_us = data_us[["position","description"]]

select_data_uk = data_uk[["job_title","job_description"]]

# rename UK columns

select_data_uk = select_data_uk.rename(index=str, columns={"job_title": "position", "job_description": "description"})
# Concatenate resulting dataframes

select_dat = pd.concat([select_data_us,select_data_uk],axis=0)

# Convert to strings

select_dat = select_dat.applymap(str)

# Replace certain strings

select_dat["description"] = select_dat["description"].replace(to_replace='Apply', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='apply', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='now', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='apply now', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='Apply Now', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='Job Description', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='job description', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='changes everything', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='everything', value="",regex=True)

select_dat["description"] = select_dat["description"].replace(to_replace='data scientist', value="Data Scientist",regex=True)
#View the resulting concatenated dataframe

select_dat.head()
#Check the resulting shape of the dataframe

select_dat.shape
#Select Data Analyst postings from the listings

Analyst = select_dat[select_dat['position'].str.contains("Data Analyst")] 

#View the slice

Analyst.head()
#Select Data Scientist postings from the listings

Scientist = select_dat[select_dat['position'].str.contains("Data Scientist")] 

#View the slice

Scientist.head()
#Select Machine learning postings from the listings

ML = select_dat[select_dat['position'].str.contains("Machine Learning")] 

#View the slice

ML.head()
#Select Big Data postings from the listings

BD = select_dat[select_dat['position'].str.contains("Big Data")] 

#View the slice

BD.head()
#Code for wordcloud (adapted for removal of stop words)



#Code adpted from : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc



#import the wordcloud package

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



#Define the word cloud function with a max of 200 words

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10,10), 

                   title = None, title_size=20, image_color=False):

    stopwords = set(STOPWORDS)

    #define additional stop words that are not contained in the dictionary

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)

    #Generate the word cloud

    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    #set the plot parameters

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

#ngram function

def ngram_extractor(text, n_gram):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



# Function to generate a dataframe with n_gram and top max_row frequencies

def generate_ngrams(df, n_gram, max_row):

    temp_dict = defaultdict(int)

    for question in df:

        for word in ngram_extractor(question, n_gram):

            temp_dict[word] += 1

    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)

    temp_df.columns = ["word", "wordcount"]

    return temp_df



#Function to construct side by side comparison plots

def comparison_plot(df_1,df_2,col_1,col_2, space):

    fig, ax = plt.subplots(1, 2, figsize=(20,10))

    

    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")

    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")



    ax[0].set_xlabel('Word count', size=14)

    ax[0].set_ylabel('Words', size=14)

    ax[0].set_title('Top 20 Bi-grams in Descriptions', size=18)



    ax[1].set_xlabel('Word count', size=14)

    ax[1].set_ylabel('Words', size=14)

    ax[1].set_title('Top 20 Tri-grams in Descriptions', size=18)



    fig.subplots_adjust(wspace=space)

    

    plt.show()
#Select descriptions from Analyst

Analyst_desc = Analyst["description"]

Analyst_desc.replace('--', np.nan, inplace=True) 

Analyst_desc_na = Analyst_desc.dropna()

#convert list elements to lower case

Analyst_desc_na_cleaned = [item.lower() for item in Analyst_desc_na]

#remove html links from list 

Analyst_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in Analyst_desc_na_cleaned]

#remove special characters left

Analyst_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in Analyst_desc_na_cleaned]

#convert to dataframe

Analyst_desc_na_cleaned = pd.DataFrame(np.array(Analyst_desc_na_cleaned).reshape(-1))

#Squeeze dataframe to obtain series

Analyst_cleaned = Analyst_desc_na_cleaned.squeeze()
#run the function on the Data Analyst headlines and Remove NA values for clarity of visualisation

plot_wordcloud(Analyst_cleaned, title="Word Cloud of Data Analyst Descriptions")
#Select descriptions from Scientist

Scientist_desc = Scientist["description"]

Scientist_desc.replace('--', np.nan, inplace=True) 

Scientist_desc_na = Scientist_desc.dropna()

#convert list elements to lower case

Scientist_desc_na_cleaned = [item.lower() for item in Scientist_desc_na]

#remove html links from list 

Scientist_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in Scientist_desc_na_cleaned]

#remove special characters left

Scientist_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in Scientist_desc_na_cleaned]

#convert to dataframe

Scientist_desc_na_cleaned = pd.DataFrame(np.array(Scientist_desc_na_cleaned).reshape(-1))

#Squeeze dataframe to obtain series

Scientist_cleaned = Scientist_desc_na_cleaned.squeeze()
#run the function on the Data Analyst headlines and Remove NA values for clarity of visualisation

plot_wordcloud(Scientist_cleaned, title="Word Cloud of Data Scientist Descriptions")
#Select descriptions from ML

ML_desc = ML["description"]

ML_desc.replace('--', np.nan, inplace=True) 

ML_desc_na = ML_desc.dropna()

#convert list elements to lower case

ML_desc_na_cleaned = [item.lower() for item in ML_desc_na]

#remove html links from list 

ML_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in ML_desc_na_cleaned]

#remove special characters left

ML_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in ML_desc_na_cleaned]

#convert to dataframe

ML_desc_na_cleaned = pd.DataFrame(np.array(ML_desc_na_cleaned).reshape(-1))

#Squeeze dataframe to obtain series

ML_cleaned = ML_desc_na_cleaned.squeeze()
#run the function on the Machine learning headlines and Remove NA values for clarity of visualisation

plot_wordcloud(ML_cleaned, title="Word Cloud of Machine learning positions Descriptions")
#Select descriptions from BD_US

BD_desc = BD["description"]

BD_desc.replace('--', np.nan, inplace=True) 

BS_desc_na = BD_desc.dropna()

#convert list elements to lower case

BD_desc_na_cleaned = [item.lower() for item in BS_desc_na]

#remove html links from list 

BD_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in BD_desc_na_cleaned]

#remove special characters left

BD_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in BD_desc_na_cleaned]

#convert to dataframe

BD_desc_na_cleaned = pd.DataFrame(np.array(BD_desc_na_cleaned).reshape(-1))

#Squeeze dataframe to obtain series

BD_cleaned = BD_desc_na_cleaned.squeeze()
#run the function on the Big Data headlines and Remove NA values for clarity of visualisation

plot_wordcloud(BD_cleaned, title="Word Cloud of Big Data positions Descriptions")
#Generate unigram for data analyst

Analyst_1gram = generate_ngrams(Analyst_cleaned, 1, 20)

#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(Analyst_1gram["wordcount"],Analyst_1gram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Data Analyst Descriptions")

plt.show()
#Obtain bi-grams and tri-grams (top 20)

Analyst_2gram = generate_ngrams(Analyst_cleaned, 2, 20)

Analyst_3gram = generate_ngrams(Analyst_cleaned, 3, 20)

#compare the bar plots

comparison_plot(Analyst_2gram,Analyst_3gram,'word','wordcount', 0.5)
#Generate unigram for data analyst

Scientist_1gram = generate_ngrams(Scientist_cleaned, 1, 20)

#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Data Scientist Descriptions")

plt.show()
#Obtain bi-grams and tri-grams (top 20)

Scientist_2gram = generate_ngrams(Scientist_cleaned, 2, 20)

Scientist_3gram = generate_ngrams(Scientist_cleaned, 3, 20)

#compare the bar plots

comparison_plot(Scientist_2gram,Scientist_3gram,'word','wordcount', 0.5)
#Generate unigram for ML positions

Scientist_1gram = generate_ngrams(ML_cleaned, 1, 20)

#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Machine Learning positions descriptions")

plt.show()
#Obtain bi-grams and tri-grams (top 20)

ML_2gram = generate_ngrams(ML_cleaned, 2, 20)

ML_3gram = generate_ngrams(ML_cleaned, 3, 20)

#compare the bar plots

comparison_plot(ML_2gram,ML_3gram,'word','wordcount', 0.5)
#Generate unigram for ML positions

BD_1gram = generate_ngrams(BD_cleaned, 1, 20)

#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Big Data positions descriptions")

plt.show()
#Obtain bi-grams and tri-grams (top 20)

BD_2gram = generate_ngrams(BD_cleaned, 2, 20)

BD_3gram = generate_ngrams(BD_cleaned, 3, 20)

#compare the bar plots

comparison_plot(BD_2gram,BD_3gram,'word','wordcount', 0.5)