import re

import os

from nltk.corpus import stopwords

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



stop_words = stopwords.words('english')



la_words = ['los','angeles','city','california']



for word in la_words:

    stop_words.append(word)



file_names = os.listdir("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/")



file_paths = []



for name in file_names:

    file_paths.append("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/" + name)



print(file_paths[0:5])
full_text = {}



bag_of_words = {}



no_punct = re.compile(r'[\?\.\,\>\<\[\]\{\(\)\'\"\;\:\_\-\+\=\%\#\@\!\&\*\$\^]')

no_nums = re.compile(r'[0-9]+')

for path in file_paths:

    with open(path,'r',encoding='ascii',errors='replace') as my_file:

        temp = my_file.read()

        full_text[path] = (temp)

        temp = no_punct.sub(string=temp,repl='')

        temp = no_nums.sub(string=temp,repl='')

        for word in temp.lower().split():

            if not word in stop_words:

                if not word in bag_of_words.keys():

                    bag_of_words[word] = 1

                else:

                    bag_of_words[word] += 1

    my_file.close()
bag_of_words_df = pd.DataFrame({'words': list(bag_of_words.keys()), 'count':list(bag_of_words.values())})
print(bag_of_words_df.describe())

print('median: '+ str(bag_of_words_df['count'].median()))
top_25 = bag_of_words_df.nlargest(25, 'count')

top_25
sns.barplot(x='count',y='words',data = top_25)

plt.title('Top 25 Most Frequent Words in Job Bulletins')

plt.show()
path_to_remove = re.compile(r'../input/cityofla/CityofLA/Job Bulletins/|[0-9]+|\.txt')

job_bullitens = []



for job in full_text.keys():

    job_bullitens.append(path_to_remove.sub(string=job,repl='').lower())





full_text_df = pd.DataFrame({'Job Bulletin':job_bullitens,'Job Bulletin Text':list(full_text.values())})
full_text_df.head()



full_pdf_text = pd.read_csv('../input/datascienceforgoodlapdftextuncleaned/all_pdf_text.csv')

print(full_pdf_text.shape)

full_pdf_text = full_pdf_text.dropna()

print(full_pdf_text.shape)
full_text_pdf = list(full_pdf_text['pdf_text_all'])

spaces = re.compile(r' {2,}|(\uf0a7)')

pdf_bag_of_words = {}

for text in full_text_pdf:

    text = spaces.sub(string = text, repl = ' ')

    text = no_punct.sub(string=text,repl='')

    text = no_nums.sub(string=text,repl='')

    for word in text.lower().split():

        if not word in stop_words:

            if not word in pdf_bag_of_words.keys():

                pdf_bag_of_words[word] = 1

            else:

                pdf_bag_of_words[word] += 1
pdf_bag_of_words_df = pd.DataFrame({'words':list(pdf_bag_of_words.keys()), 'count':list(pdf_bag_of_words.values())})

pdf_top_25 = pdf_bag_of_words_df.nlargest(25, 'count')

pdf_top_25
sns.barplot(x='count',y='words',data = pdf_top_25)

plt.title('Top 25 Most Frequent Words in PDF Job Bulletins')

plt.show()
full_text_df.to_csv('job_bulletin_text.csv')

bag_of_words_df.to_csv('from_txt_job_bulletins_bag_of_words.csv')

pdf_bag_of_words_df.to_csv('from_pdf_job_bulletins_bag_of_words.csv')
sentences = []

for text in full_text.values():

    temp = text.split('\n\n')

    for line in temp:

        temp2 = line.split('.')

        for sent in temp2:

            sent = sent.strip()

            sentences.append(sent)

len(sentences)
sentences = [i for i in sentences if i]

len(sentences)