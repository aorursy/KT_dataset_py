# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the CSV files into respective dataframes

questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

otr = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')
#Converting the string characters to lower case to bring in uniformity for the field Q26_OTHER_TEXT

text_lower_cv = pd.DataFrame(otr['Q26_OTHER_TEXT'].str.lower())

text_cv = text_lower_cv.Q26_OTHER_TEXT.unique()

# Installing NLTK to analyse the textual content of the field Q27_OTHER_TEXT

!pip install nltk
#Load nltk

import nltk
#Creating a list with all the responses

sentence_cv = ['']

i = 0

for i in range(len(text_cv)):

    sentence_cv.append(text_cv[i])

sentence_cv

#Deleting some of the responses that doesn't give any relevant methods

del sentence_cv[0:3]

sentence_cv.remove('am learning this')

del sentence_cv[11]

del sentence_cv[12]





#Replacing spaces with hyphen to get full names

sentence_cv[0] = 'time-based, lstm, i3d'

sentence_cv[3] ='video-analysis'

sentence_cv[10] ='glcm-wavelet'

sentence_cv[12] ='triplet-loss'

sentence_cv[13] ='triplet-loss'

sentence_cv[18] ='pose-estimation'

sentence_cv[19] ='pose-estimation'

sentence_cv[23] ='triplet-loss'

sentence_cv[29] ='i3d'

sentence_cv
#Creating onse single sentence with the unique value for further tokenization purpose

one_sentence_cv = "" 

for index, value in enumerate(sentence_cv):

    one_sentence_cv += (str(value)+",")

print(one_sentence_cv)
#text.dropna()

#Implement Word Tokenization

from nltk.tokenize import word_tokenize

tokenized_word_cv = word_tokenize(one_sentence_cv)



#Stopwords

from nltk.corpus import stopwords

stop_words_cv =  stopwords.words('english')

newStopWords_cv = [',','.','(',')','?','-']

stop_words_cv.extend(newStopWords_cv)



#Removing StopWords

filtered_sent_cv = []

for w in tokenized_word_cv:

    if w not in stop_words_cv:

        filtered_sent_cv.append(w)

        

#Frequency Distribution

from nltk.probability import FreqDist

fdist_cv = FreqDist(filtered_sent_cv)

print(fdist_cv)



fdist_cv.most_common(65)

    
#Creating a frequency distribution dataframe of the most used CV Methods

freq_words_cv = pd.DataFrame(filtered_sent_cv)

freq_words_cv.columns =['words']

freq_words_cv.head()

type(freq_words_cv)
#install wordcloud package using pip

! pip install wordcloud
#Importing Matplotlib for plotting purpose

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
#import sub features

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS
#WordCloud figure parameters

mpl.rcParams['figure.figsize']=(10.0,14.0)    #(6.0,4.0)

mpl.rcParams['font.size']=24              #10 

mpl.rcParams['savefig.dpi']=250           #72 

mpl.rcParams['figure.subplot.bottom']=.1
#Generating the worldcloud with the website name

wordcloud_cv = WordCloud(

                          background_color='white',

                          max_words=100,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(freq_words_cv['words']))
#Printing the wordcloud and storing in a png file

print(wordcloud_cv)

fig = plt.figure(1)

plt.imshow(wordcloud_cv)

plt.axis('off')

plt.show()
#Frequency Distribution plot

import matplotlib.pyplot as plt

plt.tick_params(axis='x', which='major', labelsize=10)

plt.title('Frequently used CV Methods')

fdist_cv.plot(60,cumulative = False)

plt.show()
#Converting the string characters to lower case to bring in uniformity for the field Q27_OTHER_TEXT

text_lower = pd.DataFrame(otr['Q27_OTHER_TEXT'].str.lower())

text = text_lower.Q27_OTHER_TEXT.unique()
#Creating a list with all the response

sentence_nlp = ['']

i = 0

for i in range(len(text)):

    sentence_nlp.append(text[i])

sentence_nlp

 
#Deleting the sentences that are noise and doesn't give any relevant methods

del sentence_nlp[0:3]

del sentence_nlp[3]

sentence_nlp.remove('am learning this ')

sentence_nlp
#Replacing few spaces with hyphens to preserve the full form of the methods

sentence_nlp[23] = 'topic-modeling'

sentence_nlp[25] = 'stopwords, lemmatization, tfidf, bow'

sentence_nlp[27] = 'text-mining by r and python libraries only'

sentence_nlp
#Converting the unque responses into one sentence for further tokenization purpose

one_sentence_nlp = "" 

for index, value in enumerate(sentence_nlp):

    one_sentence_nlp += (str(value)+",")

print(one_sentence_nlp)
#text.dropna()

#Implement Word Tokenization

from nltk.tokenize import word_tokenize

tokenized_word_nlp = word_tokenize(one_sentence_nlp)



#Stopwords

from nltk.corpus import stopwords

stop_words_nlp =  stopwords.words('english')

newStopWords_nlp = [',','.','(',')','?','-']

stop_words_nlp.extend(newStopWords_nlp)



#Removing StopWords

filtered_sent_nlp = []

for w in tokenized_word_nlp:

    if w not in stop_words_nlp:

        filtered_sent_nlp.append(w)

        

#Frequency Distribution

from nltk.probability import FreqDist

fdist_nlp = FreqDist(filtered_sent_nlp)

print(fdist_nlp)



fdist_nlp.most_common(65)

    
#Creating a Dataframe with the frequency of each word detected

freq_words_nlp = pd.DataFrame(filtered_sent_nlp)

freq_words_nlp.columns =['words']

freq_words_nlp.head()

type(freq_words_nlp)
#Generating the worldcloud with the website name

wordcloud_nlp = WordCloud(

                          background_color='white',

                          max_words=100,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(freq_words_nlp['words']))
#Printing the wordcloud and storing in a png file

print(wordcloud_nlp)

fig = plt.figure(1)

plt.imshow(wordcloud_nlp)

plt.axis('off')

plt.show()

#Frequency Distribution plot

import matplotlib.pyplot as plt

plt.tick_params(axis='x', which='major', labelsize=14)

plt.title('Frequently used NLP Methods')

fdist_nlp.plot(25,cumulative = False)

plt.show()

#Identify number of NON NULL responses for CV and NLP Question

otr.count(axis = 0)
# initialize list of lists 

data_count = [['Kaggle',mcr['Q17_Part_1'].count()]

               ,['Colab',mcr['Q17_Part_2'].count()]

               ,['GCloud',mcr['Q17_Part_3'].count()]

               ,['MAzure',mcr['Q17_Part_4'].count()]

               ,['Paperspace',mcr['Q17_Part_5'].count()]

               ,['FloydHub',mcr['Q17_Part_6'].count()]

               ,['Bynder',mcr['Q17_Part_7'].count()]

               ,['IBMWatson',mcr['Q17_Part_8'].count()]

               ,['CodeOcean',mcr['Q17_Part_9'].count()]

               ,['AWS',mcr['Q17_Part_10'].count()]

               ,['None',mcr['Q17_Part_11'].count()]

               ,['Other',mcr['Q17_Part_12'].count()]]

notebook_type_count = pd.DataFrame(data_count, columns = ['Notebook', 'Users_Count'])        

notebook_type_count.head()
# x-coordinates of left sides of bars  

left = notebook_type_count['Notebook']

  

# heights of bars 

height = notebook_type_count['Users_Count']

  

# labels for bars 

plt.tick_params(axis='x', which='major', labelsize=8)

  

# plotting a bar chart 

plt.bar(left, height, 

        width = 0.8) 

  

# naming the x-axis 

plt.xlabel('Notebook') 

# naming the y-axis 

plt.ylabel('Height') 

# plot title 

plt.title('Popularity of Hosted Notebooks') 

  

# function to show the plot 

plt.show() 

import seaborn as sns

pt1 = mcr[['Q5']]

pt1 = pt1.rename(columns={"Q5": "Title"})

pt1.drop(0, axis=0, inplace=True)



# plotting to create pie chart 

plt.figure(figsize=(38,36))

plt.subplot(221)

pt1["Title"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",5),startangle = 60,wedgeprops={"linewidth":2,"edgecolor":"k"},shadow =True)

plt.title("Title Distribution")
