import numpy as np 

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt 
data = pd.read_csv('../input/ds5230usml-project/Reviews.csv')
print('Shape:',data.shape)



data.head()
#Sorting the data with Product ID now

sorted_data = data.sort_values('ProductId',axis = 0, inplace = False, kind = 'quicksort',ascending = True,na_position='last')

sorted_data.head()

print(sorted_data.shape)
filtered_data = sorted_data.drop_duplicates(subset = {'UserId','ProfileName','Time','Text'} ,keep = 'first', inplace = False)

print('The values dropped from (568454, 11) -->',filtered_data.shape)

print('The percentage of data remaining is -->',(filtered_data.shape[0]/sorted_data.shape[0])*100,'%')
# Creating the copy of the data here 

final = filtered_data.copy()

final.shape

final.head()
# To check if the helpfulness numerator is greater than the denominator anywhere

final[final.HelpfulnessNumerator > final.HelpfulnessDenominator]
# We cannot have these two as its not possible here

final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]

final.shape
print(final.Text.values[0])

print("")

print(final.Text.values[900])

print('')

print(final.Text.values[4900])

print('')

print(final.Text.values[25000])

print('-------------------')

print('We see that we have lots of other things like html tags and stuff like that so we have to remove it')

import re

import nltk

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from nltk.corpus import wordnet

import string

from nltk import pos_tag

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer
final.head(3)
data = final.copy()





# Define the function to implement POS tagging:

def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN





# Define the main function to clean text in various ways:

def clean_text(text):

    

    # Apply regex expressions first before converting string to list of tokens/words:

    # 1. remove @usernames

    text = re.sub('@[^\s]+', '', text)

    

    # 2. remove URLs

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)

    

    # 3. remove hashtags entirely i.e. #hashtags

    text = re.sub(r'#([^\s]+)', '', text)

    

    # 4. remove emojis

    # text = emoji_pattern.sub(r'', text)

    

    # 5. Convert text to lowercase

    text = text.lower()

    

    # 6. tokenize text and remove punctuation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    

    # 7. remove numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    

    # 8. remove stop words

    # stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    

    # 9. remove empty tokens

    text = [t for t in text if len(t) > 0]

    

    # 10. pos tag text and lemmatize text

    pos_tags = pos_tag(text)

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    

    # 11. remove words with only one letter

    text = [t for t in text if len(t) > 1]

    

    # join all

    text = " ".join(text)

    

    return(text)
# Apply function on the column 'text':

data['cleaned_text'] = data['Text'].apply(lambda x: clean_text(x))
# Saving Progress to csv

# data.to_csv('Clean_Text_1.csv', index=False)

# Check out the shape again and reset_index

print(data.shape)

data.reset_index(inplace = True, drop = True)

 

# Check out data.tail() to validate index has been reset

data.tail()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Create a sid object called SentimentIntensityAnalyzer()

sid = SentimentIntensityAnalyzer()



# Apply polarity_score method of SentimentIntensityAnalyzer()

data['sentiment'] = data['cleaned_text'].apply(lambda x: sid.polarity_scores(x))



data.head()
# Keep only the compound scores under the column 'Sentiment'

data = pd.concat([data.drop(['sentiment'], axis = 1), data['sentiment'].apply(pd.Series)], axis = 1)
data.head(5)
# New column: number of characters in 'review'

data['numchars'] = data['cleaned_text'].apply(lambda x: len(x))



# New column: number of words in 'review'

data['numwords'] = data['cleaned_text'].apply(lambda x: len(x.split(" ")))



# Check the new columns:

data.tail(5)
data.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color = 'white',

        max_words = 100,

        max_font_size = 40, 

        scale = 3,

        random_state = 42

    ).generate(str(data))



    fig = plt.figure(1, figsize = (20, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(wordcloud)

    plt.show()

    

# print wordcloud

show_wordcloud(data['cleaned_text'])
# Focusing only  on 'compound' scores here...

sentimentclass_list = []

for i in range(0, len(data)):

 # current 'compound' score:

    curr_compound = data.iloc[i,:]['compound']

    if (curr_compound <= 1.0 and curr_compound >= 0.55):

        sentimentclass_list.append(5)

    elif (curr_compound < 0.55 and curr_compound >= 0.10):

        sentimentclass_list.append(4)

    elif (curr_compound < 0.10 and curr_compound > -0.10):

        sentimentclass_list.append(3)

    elif (curr_compound <= -0.10 and curr_compound > -0.55):

        sentimentclass_list.append(2)

    elif (curr_compound <= -0.55 and curr_compound >= -1.00):

        sentimentclass_list.append(1)

# Creating a new column here to add the sentiments classification

data['sentiment_class'] = sentimentclass_list
# Backing up the data frame

data.iloc[0:5, :][['compound', 'sentiment_class']]

data.tail()

data.to_csv('Clean_Text_SentimentRating.csv', index=False)
import seaborn as sns

# Distribution of sentiment_class

plt.figure(figsize = (10,5))

sns.set_palette('PuBuGn_d')

sns.catplot(x="sentiment_class", kind="count", palette="ch:.25", data=data)

plt.title('Countplot of sentiment_class by Sentiment Analysis')

plt.xlabel('sentiment class')

plt.ylabel('No. of classes')

plt.show()



plt.figure(figsize = (10,5))

sns.set_palette('PuBuGn_d')

sns.catplot(x="Score", kind="count", palette="ch:.25", data=data)

plt.title('Countplot of Scores Given in the Dataset')

plt.xlabel('Score Classes')

plt.ylabel('No. of classes')

plt.show()
# Display full text:

pd.set_option('display.max_colwidth', -1)
# Filter 10 negative reviews:

print("10 random negative original reviews and their sentiment classes and their Scores from data set:")

data[(data['sentiment_class'] == 1) | (data['sentiment_class'] == 2)].sample(n=10)[['Text', 'sentiment_class','Score']]





# Filter 10 neutral reviews:

print("10 random neutral original reviews and their sentiment classes and their score values:")

data[(data['sentiment_class'] == 3)].sample(n=10)[['Text', 'sentiment_class','Score']]

# Filter 20 positive reviews:

print("10 random positive Revies and their sentiment classes mapped against their score values:")

data[(data['sentiment_class'] == 4) | (data['sentiment_class'] == 5)].sample(n=10)[['Text', 'sentiment_class','Score']]
# !pip3 install datetime 
import datetime

from datetime import datetime



data['Time_converted']=data['Time'].apply(lambda col: datetime.utcfromtimestamp(int(col)).strftime('%Y-%m-%d'))



#strftime('%Y-%m-%d %H:%M:%S'))

data['Time_converted_ym']=data['Time'].apply(lambda col: datetime.utcfromtimestamp(int(col)).strftime('%Y-%m')) 



data.head(5)
data.isnull().sum()
data['Text_lenght']=data['Text'].apply(lambda col: len(col.split(' ')))

data.tail(2)
data['Text_lenght'].hist(bins=1000)

plt.xlim(0,400)

plt.title('Review length distribution',fontsize=14)
data.Text_lenght.describe()
def score_pos_neg(col):

    

    ''' To club the sentiments into positive neutral and negative sentiments

    '''

    

    if col == 4 or col == 5:

        

        return 'positive'

    elif col == 1 or col == 2:

        

        return 'negative'

    

    else:

        return 'neutral'



data['score_pos_neg']=data['sentiment_class'].apply(score_pos_neg)
# Checking correct assignments

data[data['score_pos_neg']=='neutral'].head()



#plotting count of positive "1" and negative "0" reviews

sns.countplot(x='score_pos_neg', data=data.sample(500));

plt.xlabel('Score 0/1',fontsize=14);

plt.ylabel('Count',fontsize=14);

plt.title('Review Score 0/1 count 1999-2012',fontsize=14);
# TOP Reviewers with count > 10

user_number_review=data.groupby(by=['UserId']).count().sort_values(by=['Text'],ascending=False)



user_number_review.head()



user_top_reviewer=user_number_review[user_number_review['Id']>10][['Id']]

user_top_reviewer.head()
# distribution of review count per user

user_top_reviewer['Id'].hist(bins=350,label='Number of reviews for Top (>10) reviewer',alpha=0.7)

plt.xlabel('Review count of top reviewers',fontsize=14)

plt.ylabel('count',fontsize=14)



plt.grid(linewidth=0.5,alpha=0.75)



plt.xlim(10,100)

plt.title('Review count for top reviewers',fontsize=14)
# getting average data per user

user_average_info=data.groupby(by=['UserId']).mean()

# top_reviewer_avg_data=pd.merge(user_average_info,user_top_reviewer,how='inner',on='UserId')



top_reviewer_avg_data=pd.merge(user_average_info,user_top_reviewer,how='inner',on='UserId')
top_reviewer_avg_data['ratio_helpful']=top_reviewer_avg_data['HelpfulnessNumerator']/top_reviewer_avg_data['HelpfulnessDenominator']
top_reviewer_avg_data.head()

top_reviewer_avg_data['Time_converted']=top_reviewer_avg_data['Time'].apply(lambda col: datetime.utcfromtimestamp(int(col)).strftime('%Y-%m'))

top_reviewer_avg_data=top_reviewer_avg_data[(top_reviewer_avg_data['ratio_helpful']>0.0) & (top_reviewer_avg_data['ratio_helpful']<1.0)]



colors = np.random.rand(top_reviewer_avg_data.shape[0])

top_reviewer_avg_data.head()
# review helpfulness distribution

top_reviewer_avg_data['ratio_helpful'].hist(bins=100,label='review helpfullness top reviewers',alpha=0.7);



plt.xlabel('Review Helpfullness ratio',fontsize=14);

plt.ylabel('count',fontsize=14);



#plt.legend()

plt.grid(linewidth=0.5,alpha=0.75)





plt.title('Review helpfullness for top reviewers',fontsize=14);

plt.savefig('helpfullness_top_reviewer_dist.png')
# review length vs helpfulness

plt.scatter(top_reviewer_avg_data['Text_lenght'],top_reviewer_avg_data['ratio_helpful'],alpha=0.7);

plt.xlim(0,600);

plt.xlabel('Review Length',fontsize=14);

plt.ylabel('Review Helpfullness ratio',fontsize=14);

#plt.legend()

plt.grid(linewidth=0.5,alpha=0.75)





plt.title('Review length for top reviewers',fontsize=14);

plt.savefig('helpfullness_top_reviewer_length.png')
#revire count of top reviewers vs. review length

plt.scatter(top_reviewer_avg_data['Id_y'],top_reviewer_avg_data['Text_lenght'],alpha=0.7); #,c=top_reviewer_avg_data['ratio_helpful'])

plt.xlim(0,175);



plt.xlabel('Top reviewer review count',fontsize=14);

plt.ylabel('Review length for top reviewers',fontsize=14);



#plt.legend()

plt.grid(linewidth=0.5,alpha=0.75)





plt.title('Review count vs. length top reviewers',fontsize=14);

plt.savefig('helpfullness_top_reviewer_length_count.png')
pos=data[data['score_pos_neg']=='positive']

neg=data[data['score_pos_neg']=='negative']

neu=data[data['score_pos_neg']=='neutral']





grp_date_pos=pos.groupby(by=['Time_converted_ym']).count();

grp_date_neg=neg.groupby(by=['Time_converted_ym']).count();

grp_date_neu=neu.groupby(by=['Time_converted_ym']).count();





grp_date_pos.reset_index(inplace=True);

grp_date_neg.reset_index(inplace=True);

grp_date_neu.reset_index(inplace=True);

# review count by score for each month from 2000 to 2012

plt.figure(figsize=(24,15))



plt.plot_date(x=grp_date_pos['Time_converted_ym'],y=grp_date_pos['score_pos_neg'],label='Score=Positive');

plt.plot_date(x=grp_date_neg['Time_converted_ym'],y=grp_date_neg['score_pos_neg'],label='Score=Negative');

plt.plot_date(x=grp_date__neu['Time_converted_ym'],y=grp_date_neu['score_pos_neg'],label='Score=Neutral');

plt.xticks(rotation=90);

plt.legend()

plt.grid(linewidth=0.7,alpha=0.75)

plt.xlim('2000-01','2012-10');

plt.xlabel('Date',fontsize=22)

plt.ylabel('Number of review',fontsize=22)

plt.title('Number of review trend from 2000 to 2012',fontsize=22);