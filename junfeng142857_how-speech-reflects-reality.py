# All Dependencies are added here

import pandas as pd

import os

from collections import Counter

import nltk

import string

import matplotlib.pyplot as plt

import numpy as np

from wordcloud import WordCloud, STOPWORDS

from scipy import misc
path = r'../input/state-of-the-union-corpus-1989-2017'

dirs = os.listdir(path)
path1 = r'../input'

dirs1 = os.listdir(path1)

dirs1
dirs
# define a function to count words in a give text file in the directory

def count_words(word, filename):

    file = open(path + "/"+ filename, encoding='utf8')

    text = file.read().lower()

    words = nltk.word_tokenize(text)

    word_counter = Counter(words)

    word_count = word_counter[word]

    return word_count
# Create a list of dictionaries from proccing the text files. These dictionaries will be used to create Pandas DataFrames

file_dict_list = []

for filename in dirs:

    file_dict = {}

    job_word_count = count_words('job',filename) + count_words('jobs', filename)

    file_dict['year'] = int(filename[-8:-4])

    file_dict['job_word_count'] = job_word_count

    file_dict_list.append(file_dict)

print(file_dict_list)
# Create a DataFrame from the list of dictionaries

df = pd.DataFrame(file_dict_list)

df
df.set_index('year', inplace=True)

df
# Plot of the job word counts

years = df.index

job_word_count = df['job_word_count'].values

plt.bar(years,job_word_count)

plt.show()
#create another DataFrame from the CSV

jobless_rate = pd.read_csv('../input/usa-unemployment-rate-from-1989-to-2017/unemployment_rate.csv', sep=',')

jobless_rate.set_index('Year', inplace=True)

jobless_rate['Annual'] = jobless_rate.mean(axis = 1)

jobless_rate
#Plot the unemployment trend

years = jobless_rate.index

joblessness = jobless_rate['Annual'].values

plt.plot(years, joblessness)

plt.xlabel('Year')

plt.ylabel('Unemployment Rate (%)')

plt.title("Unemployment Rate Trend")

plt.show()
#Merge the two DFs

final_df = pd.merge(jobless_rate, df, left_index=True, right_index=True)

final_df
#Plotting the two trends in the same plot.



fig, ax1 = plt.subplots(figsize=(8,5))

final_df['job_word_count'].plot(kind='bar', stacked=False, ax=ax1, label='No. of times "job" is mentioned')

ax2 = ax1.twinx()

ax2.plot(ax1.get_xticks(), final_df['Annual'].values, linestyle='-', marker='o', color='k', linewidth=1.0, label='Unemployment Rate (%)')



lines, labels = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2, labels + labels2, loc='best')



ax1.set_title('How State of the Union Addresses Reflect the Reality',fontweight="bold", size=15)

ax1.set_ylabel('"Job" Word Count', fontsize=12)

ax1.set_xlabel("Year", fontsize=12)

ax2.set_ylabel("Unemployment Rate (%)", fontsize=12)



plt.show()
# A closer look at the text of 2003

bush_2003_filename = "Bush_2003.txt"

bush_2003_file = open(path + "/"+ "Bush_2003.txt", encoding='utf8')

bush_2003_text = bush_2003_file.read().lower()

bush_2003_words = nltk.word_tokenize(bush_2003_text)



useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation) + ["will", "americans", "america", "american", "—", "'s"]



bush_2003_words_filtered = [word for word in bush_2003_words if word not in useless_words]

bush_2003_word_counter = Counter(bush_2003_words_filtered)

most_common_words_2003 = bush_2003_word_counter.most_common()

most_common_words_2003

# Generate a word cloud



useless_word_set = set(useless_words)

word_cloud = WordCloud(background_color="white", stopwords=useless_word_set)

word_cloud.generate(bush_2003_text)



plt.figure(figsize=(20, 10))

plt.imshow(word_cloud,interpolation='bilinear')

plt.axis("off")



plt.show()
