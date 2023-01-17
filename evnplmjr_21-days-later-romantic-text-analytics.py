import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from matplotlib import pyplot as plt

from textblob import TextBlob

import seaborn as sns

import numpy as np

import pandas as pd

import nltk
# Loading the text file and organizing it into a structured dataframe

raw_text = pd.read_csv('../input/text-analysis-keep-private/emily_texts.txt', sep='\n', header=None)

raw_text_df = pd.DataFrame(raw_text.iloc[::,0].str.split('\t', expand=True))



# Organizing the columns and encoded dummy data for categorical variables

organized_text_df = raw_text_df.rename(columns={0:'date',1:'time',2:'in/out',3:'number',4:'contact',5:'text'})

io_data = pd.get_dummies(data=organized_text_df['in/out'], prefix='text')



# Cleaned the dataframe by dropping the unnecessary columns

text_df = organized_text_df.drop(columns=['number','contact','in/out'])

text_df['sent_evan'],text_df['sent_emily'] = io_data['text_in'],io_data['text_out']



# Deconstructed the date column to only get what I needed and added to the final dataframe

expanded_time_df = text_df['time'].str.split(':', expand=True)

expanded_time_df.rename(columns={0:'hour', 1:'minutes', 2:'seconds'}, inplace=True)

text_df['time'] = expanded_time_df['hour']
# Plotting the number of text messages sent by each person during the 21 day period

text_by_person = pd.DataFrame(text_df.groupby('date')[['sent_evan','sent_emily']].sum())

plt.style.use('seaborn-whitegrid')

text_by_person.plot(kind='bar', figsize=(20,5), title='Who Texted Who More?')

plt.legend(labels=['Evan','Emily'])

plt.xticks(rotation=0)

plt.xlabel('Date')

plt.ylabel('Number of Text Messages')

plt.savefig('TextCountPerPerson.png')



# Plotting the average peak times we messaged each other during the 21 day period

sns.set_style('whitegrid')

plt.figure(figsize=(20,5))

sns.distplot(text_df['time'], kde=False)

plt.title('What Time Did We Text The Most?')

plt.xticks(range(24))

plt.xlabel('Hour of the Day')

plt.ylabel('Number of Texts Exchanged')

plt.savefig('TextsPerHour.png')
# Tokenizing the text data using the NLTK module

token_list = []

for raw_text in text_df['text']:

    tokens = nltk.word_tokenize(raw_text)

    token_list.append(tokens)



text_df['tokens'] = token_list

token_df = text_df.drop(columns='text')



# Computing the sentiment scores (polarity) and categorizing the results using the TextBlob module

sentiment_scores_tb = [round(TextBlob(str(texts)).sentiment.polarity, 3) for texts in token_df['tokens']]

sentiment_category_tb = ['positive' if score > 0 

                             else 'negative' if score < 0 

                                 else 'neutral' 

                                     for score in sentiment_scores_tb]



df = token_df.drop(columns='tokens')

sender_list = []

for x in token_df['sent_evan']:

    if x == 1:

        sender_list.append('Evan')

    else:

        sender_list.append('Emily')



# Creating a dataframe for the sentiment data analyzed       

df['Sent By'], df['Sentiment Score'], df['Sentiment Category'], df['TextBlob'] = sender_list,sentiment_scores_tb,sentiment_category_tb,token_list

results = df.drop(columns=['sent_evan','sent_emily'])

final_results = results.rename(columns={'date':'Date','time':'Hour of the Day'})

results_stats = final_results.describe(include='all')



# Plotting a chart showing the sentiment scores

plt.figure(figsize=(20,6))

sns.lineplot(x=final_results['Date'], y=final_results['Sentiment Score'])

plt.title('The Different Levels of Kilig')

plt.savefig('KiligLevels.png')