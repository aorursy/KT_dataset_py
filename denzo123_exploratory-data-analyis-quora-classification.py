# import all needed packages
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud as wc
import matplotlib.pyplot as plt
from collections import Counter
from afinn import Afinn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# get the training data
# downloaded from https://www.kaggle.com/c/quora-insincere-questions-classification/data
# since this is merely a shot exploratory look at the data, the test data is not needed
# and will therefore not be loaded
data = pd.read_csv("../input/quora-insincere-questions-classification/train.csv", sep = ',')

# drop id column
data = data.drop('qid', axis = 1)

# check for missing values
# none here
data.isna().sum()
## the following is used to clean up the data and prepare it for further analysis
# remove useless special characters
replaceing = ['"', '.', ',', '?', '!']
for word in replaceing:
    data['question_text'] = data['question_text'].str.replace(word, '')
    
# expand contractions
# dict to expand contractions
# taken from https://gist.github.com/nealrs/96342d8231b75cf4bb82
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

# replace ´ with ' to get every contraction expanded
data["question_text"] = data["question_text"].str.replace('’', "'")

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expand_contractions(text, c_re = c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())

# call the function on the data
data['question_text'] = data["question_text"].apply(lambda x: expand_contractions(x))

# make everything lower case text
data['question_text'] = data['question_text'].str.lower()

# split into words
data['question_text'] = data['question_text'].str.split(' ')

# word count
# calculate now because stop words might be relevant here
data['word_count'] = data['question_text'].str.len()

# remove stopwords
stop_words = set(stopwords.words('english'))
data['question_text'] = data['question_text'].apply(lambda x: [w for w in x if not w in stop_words])

# if question is now empty, add string to symbolize NA
data['question_text'] = data['question_text'].apply(lambda x: x if x else ['_NA_'])

# lemmatize the words
lemma = WordNetLemmatizer()
data['question_text'] = data['question_text'].apply(lambda x: [lemma.lemmatize(word, pos = 'v') for word in x])
## perform some sentiment analysis using different unsupervised methods

# function to calculate sentinent of sentence with sentiwordnet  
def get_senti_score_sentence(sentence, score_type):
    # function to calculate sentinent of word with sentiwordnet
    def get_senti_score_word(word, score_type):
        try:
            word = list(swn.senti_synsets(word, 'n'))[0]
        except IndexError:
            return 999
        if score_type == 'pos':
            return word.pos_score()
        elif score_type == 'neg':
            return word.neg_score()
        else:
            return word.obj_score()
        
    if sentence == []:
        raise ValueError('Empty Sentences not allowed!')
    score = 0
    n = len(sentence)
    for word in sentence:
        if get_senti_score_word(word, score_type) == 999:
            n = n - 1
        else:
            score += get_senti_score_word(word, score_type)
    try:
        avg = score / n
        return avg
    except ZeroDivisionError:
        return np.nan

# apply function to data
data['sent_pos'] = data['question_text'].apply(lambda x: get_senti_score_sentence(x, 'pos'))
data['sent_neg'] = data['question_text'].apply(lambda x: get_senti_score_sentence(x, 'neg'))

# fill NA Values with 0
data['sent_pos'].fillna(0, inplace = True)
data['sent_neg'].fillna(0, inplace = True)

# make it a string again to apply affinity analyis
data['question_text'] = data['question_text'].apply(lambda x: ' '.join(x))

## map out simple afinnity scores
afn = Afinn(emoticons = True)
data['afinn_score'] = data['question_text'].apply(lambda x: afn.score(x))

## vader sentiment scores
# because vader is highly correlated with afinn score (and also very slow)
# it won't be used here any further
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

# function to get vader compound score
#analyzer = SentimentIntensityAnalyzer()

#def get_vader_compound_score(sentence, analyzer):
#    scores = analyzer.polarity_scores(sentence)
#    return scores['compound']

#data['vader'] = data['question_text'].apply(lambda x: get_vader_compound_score(x))
## take a sneak peak at the data
data[:10]
## graphical analysis

# first lets plot the frequency percentages of the target variable
data['target'].value_counts().plot(kind = 'pie', labels = ['real', 'insincere'],
     startangle = 90, autopct = '%1.0f%%')
# as you can see it's clearly not equally balanced
# accuracy won't be the best metric to evaluate your classifiers in this case

# now lets look at the distributions of the different calculated variables
# first we will take a look at the afinnity scores
sns.boxplot(x = 'target', y = 'afinn_score', data = data)
# The afinnity score seems to be lower in insincere questions
# Since these scores have a range from -4 (being a word with negative connotations) and +4 (the opposite),
# it seems that people writing insincere questions use more negatively connotated words.
# However, the variance also seems to be a lot bigger in the group of insincere questions.

# let's see if there is a similar pattern in the sentiwordnet scores
sns.boxplot(x = 'target', y = 'sent_pos', data = data)
# Overall the average positive sentinent of a sentence is close to zero in both groups
# However, the mean of the insincere questions scores is slightly higher (which is rather unexpected)
# The difference is very small, but with a sample of this size statistically significant

# The negative sentinent scores offer a similar picture
sns.boxplot(x = 'target', y = 'sent_neg', data = data)
# Generally these plots indicate that the insincere questions are more polarizing in general,
# both in the negative and in the positive direction. Normal questions are expectedly more neutral.

# Let's also take a look at the distribution of the number of words in a sentence
sns.boxplot(x = 'target', y = 'word_count', data = data)
# This plot shows that the sentences of insincere questions is on average about 5 words longer than
# the normal questions. This might be a usefull variable to add if you're going to use shallow learning

# Now let's look at the actual frequency of the words.
# For starters we'll plot a wordcloud for each group

# split by target variable
normal = data.loc[data['target'] == 0]
troll = data.loc[data['target'] == 1]
 
# make it one big string
normal = ' '.join(question for question in normal['question_text'].astype(str))
troll = ' '.join(question for question in troll['question_text'].astype(str))

# We will start with the normal questions
# call the wordcloud function
wordcloud = wc.WordCloud(max_font_size = 160, max_words = 150, 
                         background_color = 'white', width = 800,
                         height = 500).generate(normal)
# plot it
plt.figure(figsize = (15, 10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()
# Words like work, help, book, make, use etc. are all used very frequently in the normal questions.
# They are mostly neutral and indicate someone actually looking for advice or similar things.

# Now let's contrast this with the most frequent words in the insincere questions
wordcloud2 = wc.WordCloud(max_font_size = 160, max_words = 150,
                          background_color = 'white', width = 800,
                          height = 500).generate(troll)
 
# plot it
plt.figure(figsize = (15, 10))
plt.imshow(wordcloud2, interpolation = 'bilinear')
plt.axis("off")
plt.show()

# you can also save these images if you want to using the following two lines of code
# wordcloud.to_file("Normal_Wordcloud.png")
# wordcloud2.to_file("Troll_Wordcloud.png")
# The difference is very obvious. Instead of the generic advice topics (like relationships, work, educatione etc.),
# the most frequently used words are all very political (for example the frequent use of donald trumps name).
# This also shows why a simple sentiment analysis of words is not sufficient in this case. The words in general are mostly
# not negative. The word people for example is neither positive or negative. But coupled with other words it 
# should be possible to perform useful topic analysis

# Wordclouds are a great tool to get a first impression of the word frequencies, but without looking at the
# actual values as well they might be misleading. The max_font_size parameter for example might cover up some
# valuable information.

# For that reason, let's look at the actual frequencies of the 15 most common words.
# convert huge strings to lists
normal = normal.split(' ')
troll = troll.split(' ')

# get the 15 most common words and their frequency values of each group using Counter
normal_count = Counter(normal).most_common(15)
troll_count = Counter(troll).most_common(15)

# function to generate input for plotting
def get_words_and_values(counter):
    words = []
    values = []
    for i in range(0, len(counter)):
        words.append(counter[i][0])
        values.append(counter[i][1])
    return words, values

# simple plot function
def freq_plot(words, values, tit = ''):  
    plt.bar(words, values)
    plt.ylabel('Absolute Frequency')
    plt.title(tit)
    plt.xticks(rotation = 45)
    
# call the functions for normal questions first
words, values = get_words_and_values(normal_count)
freq_plot(words, values, tit = 'Frequency of the 15 most common words in normal questions')
# And now we'll use the same functions to generate a similar image for the insincere questions
words, values = get_words_and_values(troll_count)
freq_plot(words, values, tit = 'Frequency of the 15 most common words in troll questions')
# Apart from the different words it is also noteworthy that the word 'people' has a relatively
# very high percentage of occurence. This might be usefull knowledge for further modelling

# Since there aren't equal numbers of insincere and normal questions, the absolute frequencies might not
# be the best metric to compare the two groups.

# Instead, we will now plot the relative frequencies of the 15 most frequent words in each group
# and compare these using grouped bar plots.
# function to calculate the relative frequency of a word
def relative_frequency(lst, element):
    return lst.count(element) / float(len(lst))

# function to get lists for plotting
def get_rel_troll_and_normal(words, document):
    rel_troll = []
    rel_normal = []
    for word in words:
        freq = relative_frequency(troll, word)
        rel_troll.append(freq)
        freq = relative_frequency(normal, word)
        rel_normal.append(freq)
    return rel_troll, rel_normal    

# simple plot function for grouped bar charts
def subcategorybar(X, vals, width = 0.8, ylab = '', tit = '', labs = []):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width / 2. + i / float(n )* width, vals[i], 
                width = width / float(n), align = 'edge')   
    plt.xticks(_X, X, rotation = 90)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.legend(labs)
    
# Let's start with the 15 most frequent normal words again
words, values = get_words_and_values(normal_count)

rel_troll, rel_normal = get_rel_troll_and_normal(words, normal)

subcategorybar(words, [rel_normal, rel_troll], ylab = 'Relative Frequency',
               tit = 'Relative Frequency of most common words in normal questions',
               labs = ['normal', 'troll'])  
# and for the insincere questions
words, values = get_words_and_values(troll_count)
rel_troll, rel_normal = get_rel_troll_and_normal(words, troll)

subcategorybar(words, [rel_troll, rel_normal], ylab = 'Relative Frequency',
               tit = 'Relative Frequency of most common words in troll questions',
               labs = ['troll', 'normal'])
# Some differences are clearly obvious. If you want to use shallow learning and plan to use n-grams for that,
# you might also want to make these kind of plots for the most frequent n-grams. The code should work for that as well.

# Overall the preceeding exploratory analysis shows some interesting patterns in the data.
# I hope it was usefull to some of you guys. If you have further suggestions or advice, let me know.