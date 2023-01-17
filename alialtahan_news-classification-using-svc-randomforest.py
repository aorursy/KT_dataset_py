#

# Import Libraries

#

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import math

import time

import nltk

#nltk.download('wordnet')



from scipy import stats



from textblob import TextBlob, Word



from wordcloud import WordCloud, STOPWORDS



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import xgboost





from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer



from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning



start_time = time.time()



#

# Read data from CSV file and save to DataFrame then print the data frame

#

file_path_1 = r'../input/fake-and-real-news-dataset/Fake.csv'

df_data_1 =  pd.read_csv(file_path_1)  





#

# Inspect Data Set

#

#print('\n\nFake News Data: Display')

#display(df_data_1)



#print('\n\nFake News Data: Dimensions')

#display(df_data_1.shape)



print('\n\nFake News Data: Information')

display(df_data_1.info())



print('\n\nFake News Data: Description')

display(df_data_1.describe())



print('\n\nFake News Data: Head')

display(df_data_1.head())





#

# Read data from CSV file and save to DataFrame then print the data frame

#

file_path_2 = r'../input/fake-and-real-news-dataset/True.csv'

df_data_2 =  pd.read_csv(file_path_2)  





#

# Inspect Data Set

#

#print('\n\nReal News Data: Display')

#display(df_data_2)



#print('\n\nReal News Data: Dimensions')

#display(df_data_2.shape)



print('\n\nReal News Data: Information')

display(df_data_2.info())



print('\n\nReal News Data: Description')

display(df_data_2.describe())



print('\n\nReal News Data: Head')

display(df_data_2.head())
#

# Function to clean HTML characters that might still be in the text

# Source: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

#

def CleanHTMLText(Text):

    Text = Text.str.replace('(<br/>)', '')

    Text = Text.str.replace('(<a).*(>).*(</a>)', '')

    Text = Text.str.replace('(&amp)', '')

    Text = Text.str.replace('(&gt)', '')

    Text = Text.str.replace('(&lt)', '')

    Text = Text.str.replace('(\xa0)', ' ')

    return Text





#

# Convert title and text columns to string

#

df_data_1['title'] = df_data_1['title'].astype(str)

df_data_1['text'] = df_data_1['text'].astype(str)



df_data_2['title'] = df_data_2['title'].astype(str)

df_data_2['text'] = df_data_2['text'].astype(str)





#

# Remove leading and trailing white charcaters

#

df_data_1['title'] = df_data_1['title'].str.strip()

df_data_1['text'] = df_data_1['text'].str.strip()



df_data_2['title'] = df_data_2['title'].str.strip()

df_data_2['text'] = df_data_2['text'].str.strip()





#

# Remove duplicate spaces, tabs, new line characters and conver them to single space

#

df_data_1['title'] = df_data_1['title'].apply(lambda t: ' '.join(t.split()))

df_data_1['text'] = df_data_1['text'].apply(lambda t: ' '.join(t.split()))



df_data_2['title'] = df_data_2['title'].apply(lambda t: ' '.join(t.split()))

df_data_2['text'] = df_data_2['text'].apply(lambda t: ' '.join(t.split()))





#

# Clean HTML characters / tags from title and text

#

df_data_1['title'] = CleanHTMLText(df_data_1['title'])

df_data_1['text'] = CleanHTMLText(df_data_1['text'])



df_data_2['title'] = CleanHTMLText(df_data_2['title'])

df_data_2['text'] = CleanHTMLText(df_data_2['text'])





#

# Add a new column to record the label (Fake / real).  1 for Fake and 0 for Real

#

df_data_1['label'] = 1  # Fake



df_data_2['label'] = 0  # Real





#

# Check if there are any duplicate records based on title and text

#

len_orig = len(df_data_1.index)

df_data_1 = df_data_1.drop_duplicates(subset = ['title','text'])

len_new = len(df_data_1.index)

if len_orig != len_new:

    print('\n\nFake News Data: No. of duplicate records that were removed based on title and text columns = ', 

          len_orig - len_new)

else:

    print('\n\nFake News Data: No duplicate records based on title and text columns')



len_orig = len(df_data_2.index)

df_data_2 = df_data_2.drop_duplicates(subset = ['title','text'])

len_new = len(df_data_2.index)

if len_orig != len_new:

    print('\n\nReal News Data: No. of duplicate records that were removed based on title and text columns = ', 

          len_orig - len_new)

else:

    print('\n\nReal News Data: No duplicate records based on title and text columns')



    

#

# Merge both data frames into one

#

df_data_all = pd.concat([df_data_1, df_data_2])





#

# Convert title and text columns to string

#

df_data_all['title'] = df_data_all['title'].astype(str)

df_data_all['text'] = df_data_all['text'].astype(str)





#    

# Check if there are any duplicate records after merger

#

len_orig = len(df_data_all.index)

df_data_all = df_data_all.drop_duplicates(subset = ['title','text'])

len_new = len(df_data_all.index)

if len_orig != len_new:

    print('\n\nAll News Data: No. of duplicate records that were removed based on title and text columns = ', 

          len_orig - len_new)

else:

    print('\n\nAll News Data: No duplicate records based on title and text column')





#

# Check if there are any records with NaN (text column)

#                           

len_orig = len(df_data_all.index)

df_data_all = df_data_all.dropna(subset = ['text'])

len_new = len(df_data_all.index)

if len_orig != len_new:

    print('\n\nAll News Data: No. of NaN records that were removed from text column = ', 

          len_orig - len_new)

else:

    print('\n\nAll News Data: No records with NaN in text column')





#

# Check if there are any records with empty string

#

nan_value = float('NaN')

df_data_all['text'] = df_data_all['text'].replace('', nan_value)                             

len_orig = len(df_data_all.index)

df_data_all = df_data_all.dropna(subset = ['text'])

len_new = len(df_data_all.index)

if len_orig != len_new:

    print('\n\nAll News Data: No. of empty string records that were removed from text column = ', 

          len_orig - len_new)

else:

    print('\n\nAll News Data: No empty string records in text column')





#

# Inspect Data Set

#

#print('\n\nAll News Data: Display')

#display(df_data_all)



#print('\n\nAll News Data: Dimensions')

#display(df_data_all.shape)



print('\n\nAll News Data: Information')

display(df_data_all.info())



print('\n\nAll News Data: Description')

display(df_data_all.describe())



print('\n\nAll News Data: Head')

display(df_data_all.head())
#

# Function to get count of upper case words in a string

# Source: https://github.com/Adarsh4052/100daysofmlcode/blob/master/Day%205-6%20-%20Fake%20%26%20Real%20News%20Classifier.ipynb

#

def CountAllUpperCaseLetterWords(t):

    upper_list = []

    for word in t.split():

        if word.isupper():

            upper_list.append(word)

    return len(upper_list)





#

# Function to get count of upper case words in a string

#

def CountAllLowerCaseLetterWords(t):

    lower_list = []

    for word in t.split():

        if word.islower():

            lower_list.append(word)

    return len(lower_list)





#

# Function to get average word length in a string

# Source: https://github.com/Adarsh4052/100daysofmlcode/blob/master/Day%205-6%20-%20Fake%20%26%20Real%20News%20Classifier.ipynb

#

def AvgWordLength(t):

    words = t.split()

    return ( sum( len(word) for word in words ) / len(words))





#    

# Check if there are any duplicate records after merger and clean up based on text and label

#    

len_orig = len(df_data_all.index)

df_data_all = df_data_all.drop_duplicates(subset = ['text','label'])

len_new = len(df_data_all.index)

if len_orig != len_new:

    print('\n\nAll News Data: No. of duplicate records that were removed based on text and label columns = ', 

          len_orig - len_new)

else:

    print('\n\nAll News Data: No duplicate records based on text and label columns')





#

# Merge title and text into one column

#

df_data_all['titleandtext'] = df_data_all['title'] + ' ' + df_data_all['text']

df_data_all['titleandtext'] = df_data_all['titleandtext'].astype(str)





#

# Add columns for word count in title and text

#

df_data_all['title_word_count'] = df_data_all['title'].str.split().str.len()

df_data_all['text_word_count'] = df_data_all['text'].str.split().str.len()





#

# Calculate the length for title and text

#

df_data_all['title_length'] = df_data_all['title'].apply(len)

df_data_all['text_length'] = df_data_all['text'].apply(len)





#

# Calculate the number of sentences

#

df_data_all['title_sentence_count'] = df_data_all['title'].str.split('.').str.len()

df_data_all['text_sentences_count'] = df_data_all['text'].str.split('.').str.len()





#

# Calculate the average word count per sentence

#

df_data_all['title_sentence_avg_words'] = df_data_all['title_word_count'] / df_data_all['title_sentence_count']

df_data_all['text_sentences_avg_words'] = df_data_all['text_word_count'] / df_data_all['text_sentences_count']





#

# Calculate the number of question marks

#

df_data_all['title_question_marks'] = df_data_all['title'].str.count('\?')

df_data_all['text_question_marks'] = df_data_all['text'].str.count('\?')





#

# Calculate the number of exclamation marks

#

df_data_all['title_exclamation_marks'] = df_data_all['title'].str.count('!')

df_data_all['text_exclamation_marks'] = df_data_all['text'].str.count('!')





#

# Use TextBlob to calculate sentiment polarity of title and text

# The polarity has a range of [-1,1]

# Source: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

#

df_data_all['title_polarity'] = df_data_all['title'].map(lambda text: TextBlob(text).sentiment.polarity)

df_data_all['text_polarity'] = df_data_all['text'].map(lambda text: TextBlob(text).sentiment.polarity)





#

# Add count of words of capitalized words, all lower case words, all upper case words and other words

# Count all other words (that are no all uppercase or all lowercase)

# Also calculate % of each category

#

df_data_all['title_lcase_count'] = df_data_all['title'].apply(lambda t: CountAllLowerCaseLetterWords(t))

df_data_all['text_lcase_count'] = df_data_all['text'].apply(lambda t: CountAllLowerCaseLetterWords(t))

df_data_all['title_ucase_count'] = df_data_all['title'].apply(lambda t: CountAllUpperCaseLetterWords(t))

df_data_all['text_ucase_count'] = df_data_all['text'].apply(lambda t: CountAllUpperCaseLetterWords(t))

df_data_all['title_other_case_count'] = df_data_all['title_word_count'] - df_data_all['title_lcase_count'] - df_data_all['title_ucase_count'] 

df_data_all['text_other_case_count'] = df_data_all['text_word_count'] - df_data_all['text_lcase_count'] - df_data_all['text_ucase_count'] 



df_data_all['title_lcase_pct'] = df_data_all['title_lcase_count'] / df_data_all['title_word_count']

df_data_all['text_lcase_pct'] = df_data_all['text_lcase_count'] / df_data_all['text_word_count']

df_data_all['title_ucase_pct'] = df_data_all['title_ucase_count'] / df_data_all['title_word_count']

df_data_all['text_ucase_pct'] = df_data_all['text_ucase_count'] / df_data_all['text_word_count']

df_data_all['title_other_case_pct'] = df_data_all['title_other_case_count'] / df_data_all['title_word_count']

df_data_all['text_other_case_pct'] = df_data_all['text_other_case_count'] / df_data_all['text_word_count']





#

# Add average word length for titles and text

#

df_data_all['title_avg_word_length'] = df_data_all['title'].apply(lambda t: AvgWordLength(t))

df_data_all['text_avg_word_length'] = df_data_all['text'].apply(lambda t: AvgWordLength(t))





# 

# Rearrange data frame columns

#

df_data_all = df_data_all[['title', 'text', 'titleandtext', 'title_word_count', 'text_word_count', 'title_length', 'text_length', 

                           'title_avg_word_length', 'text_avg_word_length', 'title_polarity', 'text_polarity',

                           'title_lcase_count', 'text_lcase_count', 'title_ucase_count', 'text_ucase_count', 'title_other_case_count', 

                           'text_other_case_count', 'title_lcase_pct', 'text_lcase_pct', 'title_ucase_pct', 'text_ucase_pct', 

                           'title_other_case_pct', 'text_other_case_pct', 'title_sentence_count', 'text_sentences_count', 

                           'title_sentence_avg_words', 'text_sentences_avg_words', 'title_question_marks', 'text_question_marks', 

                           'title_exclamation_marks', 'text_exclamation_marks', 'label']]





#

# Describe the data set.

#

print('\n\nAll News Data: Description')

display(df_data_all[['title','text', 'titleandtext']].describe())

display(df_data_all[['title_word_count', 'text_word_count', 'title_length', 'text_length', 

                     'title_avg_word_length', 'text_avg_word_length', 'title_polarity', 'text_polarity']].describe())

display(df_data_all[['title_lcase_count', 'text_lcase_count', 'title_ucase_count', 'text_ucase_count', 'title_other_case_count', 

                     'text_other_case_count', 'title_lcase_pct', 'text_lcase_pct', 'title_ucase_pct', 'text_ucase_pct', 

                     'title_other_case_pct', 'text_other_case_pct']].describe())

display(df_data_all[['title_sentence_count', 'text_sentences_count', 'title_sentence_avg_words', 'text_sentences_avg_words', 

                     'title_question_marks', 'text_question_marks', 'title_exclamation_marks', 'text_exclamation_marks']].describe())
#

# Clean the text column that will be passed to the TFDIFVectorizer

# Source: https://github.com/Adarsh4052/100daysofmlcode/blob/master/Day%205-6%20-%20Fake%20%26%20Real%20News%20Classifier.ipynb

#





#

# Chane all words to lowercase

#

df_data_all['titleandtext'] = df_data_all['titleandtext'].apply(lambda t: ' '.join(word.lower() for word in t.split()))





#

# Remove stop words from the text

#

df_data_all['titleandtext'] = df_data_all['titleandtext'].apply(lambda t: ' '.join(word for word in t.split() if word not in STOPWORDS))





#

# Remove punctuation from text

#

df_data_all['titleandtext'] = df_data_all['titleandtext'].str.replace('[^\w\s]' , '')





#

# remove numbers

#

df_data_all['titleandtext'] = df_data_all['titleandtext'].apply(lambda t: ' '.join(word for word in t.split() if not word.isnumeric()))



# Remove frequent words

# all_words = ' '.join(df_data_all['titleandtext'] ).split()

# # let's keep the threshold of 28 K which almost equal to number of data instances in the dataset

# freq_words = pd.Series(all_words).value_counts()[:20]

# # remove freq_words

# df_data_all['titleandtext'] = df_data_all['titleandtext'].apply( lambda t: ' '.join( word for word in t.split() if word not in freq_words)) 



# remove rare words

# all_words = ' '.join( df_data_all['titleandtext'] ).split()

# rare_words = pd.Series( all_words ).value_counts()[ -200000 : ]

# rare_words.sort_values

# # remove rare_words

# df_data_all['titleandtext'] = df_data_all['titleandtext'].apply( lambda t: ' '.join( word for word in t.split() if word not in rare_words))





#

# lemmatization

#

df_data_all['titleandtext'] = df_data_all['titleandtext'].apply(lambda t: ' '.join([Word(word).lemmatize() for word in t.split()]))





#

# Create new DF for titletext and label.  This will be used to TFDIF

#

df_titletext = pd.DataFrame({'titleandtext': df_data_all['titleandtext'], 

                             'label': df_data_all['label']})
#

# Data Exploration - display count for each label (dependent variable)

# Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# Source: https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn

#

def DisplayLabelBreakdown(header, data, label_list, label_col_name):

    print('\n\n' + header + ': Record count by Label')

    print(data[label_col_name].value_counts())



    plt.figure(figsize = (7, 5), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    ax = sns.countplot(x = label_col_name, data = data, alpha = 0.7)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width() / 2.,

                height + 3,

                '{:1.2f}'.format(height / len(data) * 100) + '%',

                ha = 'center')

    ax.set_xticklabels(label_list)

    plt.suptitle(header + ': Record count by Label', fontsize = 16)

    plt.xlabel('Label', fontsize = 14)

    plt.ylabel('Count', fontsize = 14)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()





label_dict = {0: 'Real',

              1: 'Fake'}





#

# Plot record breakdown by label

#

DisplayLabelBreakdown('All Newa Data', df_data_all, label_dict.values(), 'label')





#

# Print numeric feature description by label

#

print('\n\nAll Newa Data: Description')

display(df_data_all[['title_word_count', 'text_word_count', 'label']].groupby('label').describe())

display(df_data_all[['title_length', 'text_length', 'label']].groupby('label').describe())

display(df_data_all[['title_polarity', 'text_polarity', 'label']].groupby('label').describe())

display(df_data_all[['title_avg_word_length', 'text_avg_word_length', 'label']].groupby('label').describe())

display(df_data_all[['title_lcase_count', 'text_lcase_count', 'label']].groupby('label').describe())

display(df_data_all[['title_ucase_count', 'text_ucase_count', 'label']].groupby('label').describe())

display(df_data_all[['title_other_case_count', 'text_other_case_count', 'label']].groupby('label').describe())

display(df_data_all[['title_lcase_pct', 'text_lcase_pct', 'label']].groupby('label').describe())

display(df_data_all[['title_ucase_pct', 'text_ucase_pct', 'label']].groupby('label').describe())

display(df_data_all[['title_other_case_pct', 'text_other_case_pct', 'label']].groupby('label').describe())

display(df_data_all[['title_sentence_count', 'text_sentences_count', 'label']].groupby('label').describe())

display(df_data_all[['title_sentence_avg_words', 'text_sentences_avg_words', 'label']].groupby('label').describe())

display(df_data_all[['title_question_marks', 'text_question_marks', 'label']].groupby('label').describe())

display(df_data_all[['title_exclamation_marks', 'text_exclamation_marks', 'label']].groupby('label').describe())
#

# Function to Plot histogram, density plot and box plot for one independat variable with binary classes

# Source: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

# Source: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

#

def PlotBinaryClassCharts(header, data, feature_col_name, label_col_name, x_axis_label, label_list):

    #

    # Plot bivariate histogram by label

    #

    fig, ax = plt.subplots(ncols = 3, figsize = (21, 5), facecolor = 'lightgrey')

    fig.suptitle(header, fontsize = 18)

    sns.set(style = 'darkgrid', palette = 'hls')

    data[data[label_col_name] == 0][feature_col_name].hist(ax = ax[0], alpha = 0.7, bins = 30, label = label_list[0])

    data[data[label_col_name] == 1][feature_col_name].hist(ax = ax[0], alpha = 0.7, bins = 30, label = label_list[1])

    ax[0].set_title('Histogram by Label', size = 16)

    ax[0].set_xlabel(x_axis_label, fontsize = 14)

    ax[0].set_ylabel('Count', fontsize = 14)

    ax[0].tick_params(axis = 'both', labelsize = 12)

#    ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

#    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(float(x), ',.1f')))

    ax[0].legend()

    

    

    #

    # Plot bivariate density plot by label

    #

    for key in label_list:

        try:

            sns.distplot(data[feature_col_name][data[label_col_name] == key], 

                         hist = True, 

                         kde = True,

                         kde_kws = {'linewidth': 1},

                         label = label_list[key],

                         bins = 30,

                         ax = ax[1])

        except Exception as e:

            print('\n\nDensity plot for ' + header + ' error: ' + str(e))

    ax[1].set_title('Density Plot by Label', size = 16)

    ax[1].set_xlabel(x_axis_label, fontsize = 14)

    ax[1].set_ylabel('Density', fontsize = 14)

    ax[1].tick_params(axis = 'both', labelsize = 12)

#    ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(float(x), ',.2f')))

#    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(float(x), ',.1f')))

    ax[1].legend()

    

    

    #

    # Plot bivariate box plot by label

    #

    data_to_plot = [data[data[label_col_name] == 0][feature_col_name], data[data[label_col_name] == 1][feature_col_name]]

    bp = ax[2].boxplot(data_to_plot)

    plt.xticks([1, 2], [label_list[0], label_list[1]])

    ax[2].set_title('Box Plot by Label', size = 16)

    ax[2].set_xlabel('Label', fontsize = 14)

    ax[2].set_ylabel(x_axis_label, fontsize = 14)

    ax[2].tick_params(axis = 'both', labelsize = 12)

#    ax[2].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(float(x), ',.1f')))

    plt.show()

    

    

#

# Plot vivarate chars for original features

#

PlotBinaryClassCharts('Title Word Count', df_data_all, 'title_word_count', 'label', 'Word Count', label_dict)

PlotBinaryClassCharts('Title Length', df_data_all, 'title_length', 'label', 'Length', label_dict)

PlotBinaryClassCharts('Title Average Word Length', df_data_all, 'title_avg_word_length', 'label', 'Average Word Length', label_dict)

PlotBinaryClassCharts('Title Lowercase Word Count', df_data_all, 'title_lcase_count', 'label', 'Lower Case Word Count', label_dict)

PlotBinaryClassCharts('Title Lowercase Word Percentage', df_data_all, 'title_lcase_pct', 'label', 'Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Title Uppercase Word Count', df_data_all, 'title_ucase_count', 'label', 'Uppercase Word Count', label_dict)

PlotBinaryClassCharts('Title Uppercase Word Percentage', df_data_all, 'title_ucase_pct', 'label', 'Uppercase Word Percentage', label_dict)

PlotBinaryClassCharts('Title Non Uppercase/Lowercase Word Count', df_data_all, 'title_other_case_count', 'label', 'Non Uppercase/Lowercase Word Count', label_dict)

PlotBinaryClassCharts('Title Non Uppercase/Lowercase Word Percentage', df_data_all, 'title_other_case_pct', 'label', 'Non Uppercase/Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Title Polarity', df_data_all, 'title_polarity', 'label', 'Polarity', label_dict)

PlotBinaryClassCharts('Title Sentence Count', df_data_all, 'title_sentence_count', 'label', 'Sentence Count', label_dict)

PlotBinaryClassCharts('Title Average Words per Sentence', df_data_all, 'title_sentence_avg_words', 'label', 'Average Words per Sentence', label_dict)

PlotBinaryClassCharts('Title Question Mark Count', df_data_all, 'title_question_marks', 'label', 'Question Mark Count', label_dict)

PlotBinaryClassCharts('Title Exclamation Mark Count', df_data_all, 'title_exclamation_marks', 'label', 'Exclamation Mark Count', label_dict)



PlotBinaryClassCharts('Text Word Count', df_data_all, 'text_word_count', 'label', 'Word Count', label_dict)

PlotBinaryClassCharts('Text Length', df_data_all, 'text_length', 'label', 'Length', label_dict)

PlotBinaryClassCharts('Text Average Word Length', df_data_all, 'text_avg_word_length', 'label', 'Average Word Length', label_dict)

PlotBinaryClassCharts('Text Lowercase Word Count', df_data_all, 'text_lcase_count', 'label', 'Lower Case Word Count', label_dict)

PlotBinaryClassCharts('Text Lowercase Word Percentage', df_data_all, 'text_lcase_pct', 'label', 'Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Text Uppercase Word Count', df_data_all, 'text_ucase_count', 'label', 'Uppercase Word Count', label_dict)

PlotBinaryClassCharts('Text Uppercase Word Percentage', df_data_all, 'text_ucase_pct', 'label', 'Uppercase Word Percentage', label_dict)

PlotBinaryClassCharts('Text Non Uppercase/Lowercase Word Count', df_data_all, 'text_other_case_count', 'label', 'Non Uppercase/Lowercase Word Count', label_dict)

PlotBinaryClassCharts('Text Non Uppercase/Lowercase Word Percentage', df_data_all, 'text_other_case_pct', 'label', 'Non Uppercase/Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Text Polarity', df_data_all, 'text_polarity', 'label', 'Polarity', label_dict)

PlotBinaryClassCharts('Text Sentence Count', df_data_all, 'text_sentences_count', 'label', 'Sentence Count', label_dict)

PlotBinaryClassCharts('Text Average Words per Sentence', df_data_all, 'text_sentences_avg_words', 'label', 'Average Words  per Sentence', label_dict)

PlotBinaryClassCharts('Text Question Mark Count', df_data_all, 'text_question_marks', 'label', 'Question Mark Count', label_dict)

PlotBinaryClassCharts('Text Exclamation Mark Count', df_data_all, 'text_exclamation_marks', 'label', 'Exclamation Mark Count', label_dict)
#

# list of numeric columns excluding label

#

numeric_columns = df_data_all[df_data_all.columns[0:-1]].select_dtypes([np.number]).columns





#

# Function to plot Histogram

#

def PlotHistogramAll(header, data, numeric_col_list):   

    fig = plt.figure(figsize = (21, 21), facecolor = 'lightgrey')

    ax = fig.gca()

    sns.set(style = 'darkgrid', palette = 'hls')

    data[numeric_col_list].hist(ax = ax, alpha = 0.7, bins = 30)

    fig.suptitle(header, fontsize = 18)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.show()





#

# Plot histogram of all features before transforming to normal distribution

#

PlotHistogramAll('All News Data: Histogram Before Normalization and Scaling', df_data_all, numeric_columns)





#

# Create a new dataframe for normalized and scaled data

#

df_data_all_norm = df_data_all.copy()





#

# Loop through each column and determine the skewness to understand whether it is normally 

# distributed or not.

# If skew >= 1 then it is positive highly skewed and it can be converted to normal distribution 

# by applying log10(x + 1)

# If skew <= -1 then it is negative highly skewed and it can be converted to normal distribution 

# by applying log10(max(x+1) - x)

# If skew > -1 and <= -0.5 then it negative modertely skewed and it can be converted to normal 

# distribution by applying sqrt(max(x+1) - x)

# If skew >= 0.5 and < 1 then it is positve moderately skewed and it can be converted to normal 

# distribution by applying sqrt(x)

# Else it is normally distributed

# Source: https://www.datanovia.com/en/lessons/transform-data-to-normal-distribution-in-r/

#

skew_list = df_data_all_norm[numeric_columns].skew(axis = 0, skipna = True)

print('\n\nSkew Summary:')

for key, value in skew_list.items():

    orig_data = df_data_all_norm[key]

    new_data = None

    msg = key

    if value >= 1:

        #Positive highly skewed

        #new_data = np.log10(orig_data)

        new_data = np.log10(orig_data + 1)

        msg = msg + ': is Positive highly skewed. Skew = '

    elif value <= -1:

        #Negative highly skewed

        new_data = np.log10(max(orig_data + 1) - orig_data)

        msg = msg +': is Negative highly skewed. Skew = '

    elif value > -1 and value <= -0.5:

        #Negative moderately skewed

        new_data = np.sqrt(max(orig_data + 1) - orig_data)

        msg = msg +': is Negative moderately skewed. Skew = '

    elif value >= 0.5 and value < 1:

        #Positive moderately skewed

        new_data = np.sqrt(orig_data)

        msg = msg +': is Positive moderately skewed. Skew = '

    else:

        new_data = orig_data

        msg = msg +': No change since it is normally ditributed. Skew = '    

    df_data_all_norm[key] = new_data

    msg = msg + str(round(value, 3))

    print('\t' + msg)



    

#

# Check if there are any records with NaN, Inf, -Inf after transformation and remove them

#                           

len_orig = len(df_data_all_norm.index)

df_data_all_norm = df_data_all_norm[~df_data_all_norm.isin([np.nan, np.inf, -np.inf]).any(1)]

len_new = len(df_data_all_norm.index)

if len_orig != len_new:

    print('\n\nAll News Data - Normalized and Scaled: No. of NaN, INF, -INF records that were removed from numerical columns = ', 

          len_orig - len_new, '\n\n')

else:

    print('\n\nAll News Data - Normalized and Scaled: No records with NaN, INF, -INF records that were removed from numerical columns\n\n')



    

#

# Plot histogram of all features after transforming to normal distribution

#

PlotHistogramAll('All News Data: Histogram After Normalization and Scaling', df_data_all_norm, numeric_columns)





#

# Scale the normalized data

#

scaler = StandardScaler()

df_data_all_norm[numeric_columns] = scaler.fit_transform(df_data_all_norm[numeric_columns])
#

# Remove the outliers from the data set by using stats.zscore

# Source: https://github.com/pandas-dev/pandas/issues/15111

#

def RemoveOutliers(header, data, numeric_col_list):

    # Remove outliers by only keeping records with zscore below 3 only.  Anything >= 3 is an outlier 

    # and will be removed

    df_temp = data[(np.abs(stats.zscore(data[numeric_col_list])) < 3).all(axis = 1)]

    if len(data) != len(df_temp):

        num_outliers_removed = len(data) - len(df_temp)

        print('\n\n' + header + ': No. of outlier records to be removed =', num_outliers_removed)

        data = df_temp

        print('\n\n' + header + ': Description after removing outliers:')

        display(data.describe())

    else:

        print('\n\n' + header + ': No outlier were records')

    return data





#

# Remove outliers

#

df_data_all = RemoveOutliers('All News Data', df_data_all, numeric_columns)

df_data_all_norm = RemoveOutliers('All News Data - Normalized and Scaled', df_data_all_norm, numeric_columns)
#

# Plot bivariate charts for Normalized and Scaled Data

#

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Word Count', df_data_all_norm, 'title_word_count', 'label', 'Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Length', df_data_all_norm, 'title_length', 'label', 'Length', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Average Word Length', df_data_all_norm, 'title_avg_word_length', 'label', 'Average Word Length', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Lowercase Word Count', df_data_all_norm, 'title_lcase_count', 'label', 'Lower Case Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Lowercase Word Percentage', df_data_all_norm, 'title_lcase_pct', 'label', 'Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Uppercase Word Count', df_data_all_norm, 'title_ucase_count', 'label', 'Uppercase Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Uppercase Word Percentage', df_data_all_norm, 'title_ucase_pct', 'label', 'Uppercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Non Uppercase/Lowercase Word Count', df_data_all_norm, 'title_other_case_count', 'label', 'Non Uppercase/Lowercase Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Non Uppercase/Lowercase Word Percentage', df_data_all_norm, 'title_other_case_pct', 'label', 'Non Uppercase/Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Polarity', df_data_all_norm, 'title_polarity', 'label', 'Polarity', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Sentence Count', df_data_all_norm, 'title_sentence_count', 'label', 'Sentence Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Average Words per Sentence', df_data_all_norm, 'title_sentence_avg_words', 'label', 'Average Words per Sentence', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Question Mark Count', df_data_all_norm, 'title_question_marks', 'label', 'Question Mark Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Title Exclamation Mark Count', df_data_all_norm, 'title_exclamation_marks', 'label', 'Exclamation Mark Count', label_dict)



PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Word Count', df_data_all_norm, 'text_word_count', 'label', 'Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Length', df_data_all_norm, 'text_length', 'label', 'Length', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Average Word Length', df_data_all_norm, 'text_avg_word_length', 'label', 'Average Word Length', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Lowercase Word Count', df_data_all_norm, 'text_lcase_count', 'label', 'Lower Case Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Lowercase Word Percentage', df_data_all_norm, 'text_lcase_pct', 'label', 'Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Uppercase Word Count', df_data_all_norm, 'text_ucase_count', 'label', 'Uppercase Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Uppercase Word Percentage', df_data_all_norm, 'text_ucase_pct', 'label', 'Uppercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Non Uppercase/Lowercase Word Count', df_data_all_norm, 'text_other_case_count', 'label', 'Non Uppercase/Lowercase Word Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Non Uppercase/Lowercase Word Percentage', df_data_all_norm, 'text_other_case_pct', 'label', 'Non Uppercase/Lowercase Word Percentage', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Sentence Count', df_data_all_norm, 'text_sentences_count', 'label', 'Sentence Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Average Words per Sentence', df_data_all_norm, 'text_sentences_avg_words', 'label', 'Average Words per Sentence', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Question Mark Count', df_data_all_norm, 'text_question_marks', 'label', 'Question Mark Count', label_dict)

PlotBinaryClassCharts('Normalized and Scaled without outliers: Text Exclamation Mark Count', df_data_all_norm, 'text_exclamation_marks', 'label', 'Exclamation Mark Count', label_dict)
#

# Check the correlation of independent variables by plotting a heat map

# Check if there is any correlation between independaet variables and drop highly correlated variables

# Source: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

# Source: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

#

def RemoveCorrelatedFeatures(header, data, numeric_col_list):

    # Find the correlation matrix for independent variables

    corr_matrix = data[numeric_col_list].corr()

    

    # Plot a heat map for the correlation matrix

    plt.figure(figsize = (14, 14), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    ax = sns.heatmap(

        corr_matrix, 

        vmin = -1, vmax = 1, center = 0,

        cmap = sns.diverging_palette(20, 220, n = 50, s = 50, l = 50),

        square = True)

    ax.set_xticklabels(

        ax.get_xticklabels(),

        rotation = 45,

        horizontalalignment = 'right')

    plt.title(header +': Correlation Heatmap', fontsize = 18)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.show()

    

    # Set a threshold for correlation.  Any feature with corr greater than this threshold will be dropped

    corr_threshold = 0.75



    # Create correlation matrix.  Take absolute values only

    corr_matrix = corr_matrix.abs()



    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    

    # Find index of feature columns with correlation greater than Correlation Threshold

    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    

    if len(to_drop) > 0:

        # Display highly correlated features

        print('\n\n' + header + ': Correlated features to be dropped:')

        print(to_drop)

        

        # Drop features 

        data = data.drop(data[to_drop], axis = 1)



        # Inspect Data Set after removing correlated columns

        print('\n\n' + header + ': Information after removing correlated columns:')

        display(data.info())

        print('\n\n' + header + ': Description after removing correlated columns:')

        display(data.describe())

        print('\n\n' + header + ': Head after removing correlated columns:')

        display(data.head())

    else:

        print('\n\n' + header + ': No correlated features to be dropped')

    return data

    

df_data_all = RemoveCorrelatedFeatures('All News Data without outliers', df_data_all, numeric_columns)

df_data_all_norm = RemoveCorrelatedFeatures('All News Data - Normalized and Scaled without outliers', df_data_all_norm, numeric_columns)



#

# RFE: Recursive Feature Elimination

# SourceL https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

#



# data_final_vars=data_final.columns.values.tolist()

# y=['y']

# X=[i for i in data_final_vars if i not in y]from sklearn.feature_selection import RFE

# from sklearn.linear_model import LogisticRegressionlogreg = LogisticRegression()rfe = RFE(logreg, 20)

# rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)



# cols=[ list columns to be removed ] 

# X=os_data_X[cols]

# y=os_data_y['y']
#

# Display records breakdown per label

#

DisplayLabelBreakdown('All News Data without outliers', df_data_all, label_dict.values(), 'label')

DisplayLabelBreakdown('All News Data - Normalized and Scaled without outliers', df_data_all_norm, label_dict.values(), 'label')
#

# Function to vectorize the words and return the top n words

# Source: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

#

def get_top_n_words(corpus, n = None, stop_words = None, bigram = False):

    if bigram:

        vec = CountVectorizer(stop_words = stop_words, ngram_range = (2, 2)).fit(corpus)

    else:

        vec = CountVectorizer(stop_words = stop_words).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis = 0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    return words_freq[:n]





#

# Create a dictionry to store all word frequencies

#

all_word_dict = {'Real News Title: All words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title']),

                 'Real News Title: All words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'], 

                                                                                   stop_words = 'english'),

                 'Fake News Title: All words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title']),

                 'Fake News Title: All words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'],

                                                                                   stop_words = 'english'),

                 'Real News Text: All words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text']),

                 'Real News Text: All words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'],

                                                                                   stop_words = 'english'),

                 'Fake News Text: All words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text']),

                 'Fake News Text: All words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'],

                                                                                   stop_words = 'english')}





#

# Create a dictionry to store top 30 word frequencies

#

top_word_dict = {'Real News Title: Top 30 words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'], n = 30),

                 'Real News Title: Top 30 words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'], n = 30, 

                                                                                   stop_words = 'english'),

                 'Fake News Title: Top 30 words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'], n = 30),

                 'Fake News Title: Top 30 words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'], n = 30,

                                                                                   stop_words = 'english'),

                 'Real News Text: Top 30 words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'], n = 30),

                 'Real News Text: Top 30 words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'], n = 30,

                                                                                   stop_words = 'english'),

                 'Fake News Text: Top 30 words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'], n = 30),

                 'Fake News Text: Top 30 words without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'], n = 30,

                                                                                   stop_words = 'english')}
#

# Function to display multiple word clouds in one diagram for word frequencies that are stored in a dictionary

# Source: https://www.datacamp.com/community/tutorials/wordcloud-python

# Source: https://stackoverflow.com/questions/54076679/how-to-generate-wordclouds-next-to-each-other-in-python

#

def ShowWordClouds(word_dict, bigram = False):

    n_word_clouds = len(word_dict)

    n_cols = 2

    n_rows = np.ceil(n_word_clouds / n_cols)

    plt.figure(figsize = (21, 5 * n_rows), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    i = 1

    for key in word_dict:

        if 'stop' in key.lower():

            stopwords = STOPWORDS

        else:

            stopwords = None

        if bigram:

            word_cloud = WordCloud(

                width = 1750,

                height = 1000,

                background_color = 'seashell',

                stopwords = STOPWORDS).generate_from_frequencies(dict(word_dict[key]))

        else:

            word_cloud = WordCloud(

                width = 1750,

                height = 1000,

                background_color = 'seashell',

                stopwords = STOPWORDS).generate(str(word_dict[key]))

        plt.subplot(n_rows, n_cols, i).set_title(key, fontdict = {'fontsize': 16})

        plt.tight_layout(pad = 0)

        plt.imshow(word_cloud, interpolation = 'bilinear')

        plt.axis('off')

        i = i + 1

    plt.show()

    

    

#

# Display WordCloud for Top 30 words for all combinations

#

ShowWordClouds(all_word_dict)
#

# Function to display multiple word clouds in one diagram for word frequencies that are stored in a dictionary

# Source: https://www.datacamp.com/community/tutorials/wordcloud-python

# Source: https://stackoverflow.com/questions/54076679/how-to-generate-wordclouds-next-to-each-other-in-python

#

def ShowWordFrequencyCharts(word_dict):

    n_word_clouds = len(word_dict)

    n_cols = 2

    n_rows = int(np.ceil(n_word_clouds / n_cols))

    fig, ax = plt.subplots(ncols = n_cols, nrows = n_rows, figsize = (21, 5 * n_rows), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    i = 1

    for key in word_dict:

        common_words = dict(word_dict[key])

        names = list(common_words.keys())

        values = list(common_words.values())

        plt.subplot(n_rows, n_cols, i).set_title(key, fontdict = {'fontsize': 16})

        plt.tight_layout(pad = 0)

        plt.bar(range(len(common_words)), values, tick_label = names, color = 'skyblue', edgecolor = 'grey')

        plt.xlabel('Word', fontsize = 14)

        plt.ylabel('Count', fontsize = 14)

        plt.xticks(rotation = 60, fontsize = 12)

        plt.yticks(fontsize = 12)

        plt.subplot(n_rows, n_cols, i).get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        i = i + 1

    plt.show()





#

# Display Word Count for Top 30 words for all combinations

#

ShowWordFrequencyCharts(top_word_dict)
#

# Create a dictionry to store all bigram frequencies

#

all_bigram_dict = {'Real News Title: All bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'],

                                                                      bigram = True),

                 'Real News Title: All bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'],

                                                                                       stop_words = 'english',

                                                                                       bigram = True),

                 'Fake News Title: All bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'],

                                                                    bigram = True),

                 'Fake News Title: All bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'],

                                                                                       stop_words = 'english',

                                                                                       bigram = True),

                 'Real News Text: All bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'],

                                                                   bigram = True),

                 'Real News Text: All bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'],

                                                                                      stop_words = 'english',

                                                                                      bigram = True),

                 'Fake News Text: All bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'],

                                                                   bigram = True),

                 'Fake News Text: All bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'],

                                                                                      stop_words = 'english',

                                                                                      bigram = True)}





#

# Create a dictionry to store top 30 bigram frequencies

#

top_bigram_dict = {'Real News Title: Top 30 bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'], 

                                                                         n = 30,

                                                                         bigram = True),

                 'Real News Title: Top 30 bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['title'], 

                                                                                          n = 30,

                                                                                          stop_words = 'english',

                                                                                          bigram = True),

                 'Fake News Title: Top 30 bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'],

                                                                       n = 30,

                                                                       bigram = True),

                 'Fake News Title: Top 30 bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['title'],

                                                                                          n = 30,

                                                                                          stop_words = 'english',

                                                                                          bigram = True),

                 'Real News Text: Top 30 bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'],

                                                                      n = 30,

                                                                      bigram = True),

                 'Real News Text: Top 30 bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 0]['text'],

                                                                                         n = 30,

                                                                                         stop_words = 'english',

                                                                                         bigram = True),

                 'Fake News Text: Top 30 bigrams': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'],

                                                                      n = 30,

                                                                      bigram = True),

                 'Fake News Text: Top 30 bigrams without stop words': get_top_n_words(corpus = df_data_all[df_data_all['label'] == 1]['text'],

                                                                                         n = 30,

                                                                                         stop_words = 'english',

                                                                                         bigram = True)}
#

# Display WordCloud for Top 30 bigrams for all combinations

#

ShowWordClouds(all_bigram_dict, bigram = True)
#

# Display Word Count for Top 30 biagrams for all combinations

#

ShowWordFrequencyCharts(top_bigram_dict)



print('Execution Time: --- %s seconds ---' % round((time.time() - start_time), 2))
#

# Split the data frame into two data frames: One data frame from independent variables and 

# one for the dependant variable

#





#

# Data after outlier removal

#

numeric_columns = df_data_all[df_data_all.columns[0:-1]].select_dtypes([np.number]).columns

df_X = pd.DataFrame(df_data_all[numeric_columns])

df_Y = pd.DataFrame(df_data_all[df_data_all.columns[-1]])





#

# Data after outlier removal, transformation and scaling

#

numeric_columns = df_data_all_norm[df_data_all_norm.columns[0:-1]].select_dtypes([np.number]).columns

df_X_norm = pd.DataFrame(df_data_all_norm[numeric_columns])

df_Y_norm = pd.DataFrame(df_data_all_norm[df_data_all_norm.columns[-1]])
#

# Split the data into train and test data frames 

# Prepare train / test split data (non-stratified and strartified)

#

split_pct = 0.8

# Train/Test: Non-stratified

df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(

                                                    df_X,

                                                    df_Y,

                                                    train_size = split_pct,

                                                    random_state = 10)

# Train/Test: Stratified

df_X_train_strat, df_X_test_strat, df_Y_train_strat, df_Y_test_strat = train_test_split(

                                                    df_X,

                                                    df_Y,

                                                    train_size = split_pct,

                                                    stratify = df_Y,

                                                    random_state = 10)

# Train/Test Normalized and Scaled: Non-stratified

df_X_norm_train, df_X_norm_test, df_Y_norm_train, df_Y_norm_test = train_test_split(

                                                    df_X_norm,

                                                    df_Y_norm,

                                                    train_size = split_pct,

                                                    random_state = 10)

# Train/Test Normalized and Scaled: Stratified

df_X_norm_train_strat, df_X_norm_test_strat, df_Y_norm_train_strat, df_Y_norm_test_strat = train_test_split(

                                                    df_X_norm,

                                                    df_Y_norm,

                                                    train_size = split_pct,

                                                    stratify = df_Y_norm,

                                                    random_state = 10)





#

# Use TFIDF to calculate the Term Frequency and Inverse Document Frequency

# This is to be applied to the titletext

# Source: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

#

df_X_tfidf = df_titletext['titleandtext']

df_Y_tfidf = df_titletext['label']



df_X_tfidf_train, df_X_tfidf_test, df_Y_tfidf_train, df_Y_tfidf_test = train_test_split(

                                                    df_X_tfidf,

                                                    df_Y_tfidf,

                                                    train_size = split_pct,

                                                    random_state = 10)

# Train/Test Normalized and Scaled: Stratified

df_X_tfidf_train_strat, df_X_tfidf_test_strat, df_Y_tfidf_train_strat, df_Y_tfidf_test_strat = train_test_split(

                                                    df_X_tfidf,

                                                    df_Y_tfidf,

                                                    train_size = split_pct,

                                                    stratify = df_Y_tfidf,

                                                    random_state = 10)



# Fit and teansforma all data

tfidf_vect = TfidfVectorizer(stop_words = 'english', max_df = 0.7, analyzer = 'word', sublinear_tf = True, use_idf = True, smooth_idf = True)

df_X_tfidf = tfidf_vect.fit_transform(df_titletext['titleandtext'])



# Fit and transform train set, transform test set - non stratified

df_X_tfidf_train = tfidf_vect.fit_transform(df_X_tfidf_train)

df_X_tfidf_test = tfidf_vect.transform(df_X_tfidf_test)

# Fit and transform train set, transform test set - stratified

df_X_tfidf_train_strat = tfidf_vect.fit_transform(df_X_tfidf_train_strat)

df_X_tfidf_test_strat = tfidf_vect.transform(df_X_tfidf_test_strat)
#

# calculate the fpr and tpr for all thresholds of the classification then plot ROC Curve

# Source: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

#

def PlotROCCurve(TestY, Predicted):

    roc_auc = roc_auc_score(TestY, Predicted)

    fpr, tpr, thresholds = roc_curve(TestY, Predicted)

    plt.figure(figsize = (7, 5), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    plt.plot(fpr, tpr, label = 'AUC = %0.3f' % roc_auc)

    plt.plot([0, 1], [0, 1],'c--')

    plt.xlim([-.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate', fontsize = 14)

    plt.ylabel('True Positive Rate', fontsize = 14)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.title('Receiver Operating Characteristic', fontsize = 16)

    plt.legend(loc = 'lower right')

    plt.show()

    



#

# Function to plot Tuning score (ROC AUC)

#

def PlotTuningAccuracy(name, param_list, score_history_train, score_history_test, opt_param, opt_score):

    plt.figure(figsize = (7, 5), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    if isinstance(opt_param, str):

        plt.plot(param_list, score_history_train, label = 'Train')

        plt.plot(param_list, score_history_test, label = 'Test - AUC = %0.3f' % opt_score + ' @ ' + name + ' = ' + opt_param)

    else:

        plt.plot(param_list, score_history_train,  label = 'Train')

        plt.plot(param_list, score_history_test, label = 'Test - AUC = %0.3f' % opt_score + ' @ ' + name + ' = %0.3f' % opt_param)

    plt.xlabel(name, fontsize = 14)

    plt.ylabel('ROC AUC Score', fontsize = 14)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.plot(opt_param, opt_score, 'or')

    plt.ylim([-0.05, 1.05])

    plt.title('Tuning Hyperparameter: ' + name, fontsize = 16)

    plt.legend(loc = 'lower right')

    plt.show()

    



#

# Function to plot Tconfusion matrix

#  

def PlotConfusionMatrix(TestY, Predicted, label_list):    

    cm = confusion_matrix(TestY, Predicted)

    plt.figure(figsize = (7, 5), facecolor = 'lightgrey')

    sns.set(style = 'darkgrid', palette = 'hls')

    ax = sns.heatmap(cm, annot = True, cmap = sns.diverging_palette(20, 220, n = 200), fmt = 'd')

    ax.set_xlabel('Predicted labels', fontsize = 14)

    ax.set_ylabel('True labels', fontsize = 14)

    ax.xaxis.set_ticklabels(label_list)

    ax.yaxis.set_ticklabels(label_list)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.title('Confusion Matrix', fontsize = 16)

    plt.show()
#

# Create a function to Run Support Vector Machine Classifier and tune it - this is for Train/Test Set

# The hyperparameters to be tuned: C, gamma

# Display Classification Report, ROC Curve, Confusion Matrix 

# Source: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

#

@ignore_warnings(category = ConvergenceWarning)

def RunSVCTestTrain(model_name, TrainX, TestX, TrainY, TestY, SummaryTable, label_list):

    

    func_start_time = time.time()

    lbl = model_name + ': ' + SummaryTable.at['Validation Method', model_name]

    opt_params = {'gamma': None,

                  'C': None}

#                  'kernel': None}

    

    

    #

    # Tune gamma to maximize ROC

    #

    param_values = [0.1, 1, 10, 100]

    score_history_train = []

    score_history_test = []

    for param_value in param_values:

        svc = SVC(gamma = param_value)

        svc.fit(TrainX, TrainY.values.ravel())

        score_history_train.append(roc_auc_score(TrainY, svc.predict(TrainX)))

        Predicted = svc.predict(TestX)

        score_history_test.append(roc_auc_score(TestY, Predicted))

    # Optimal values

    opt_params['gamma'] = param_values[np.argmax(score_history_test)]

    opt_auc_roc = np.amax(score_history_test)

    print('\n\nOptimal gamma value = ', opt_params['gamma'])

    print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

    # Plot the ROC AUC Score

    PlotTuningAccuracy('gamma', param_values, score_history_train, score_history_test, opt_params['gamma'], opt_auc_roc)

    

    

    #

    # Tune C to maximize ROC

    #

    param_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    score_history_train = []

    score_history_test = []

    for param_value in param_values:

        svc = SVC(gamma = opt_params['gamma'],

                  C = param_value)

        svc.fit(TrainX, TrainY.values.ravel())

        score_history_train.append(roc_auc_score(TrainY, svc.predict(TrainX)))

        Predicted = svc.predict(TestX)

        score_history_test.append(roc_auc_score(TestY, Predicted))

    # Optimal values

    opt_params['C'] = param_values[np.argmax(score_history_test)]

    opt_auc_roc = np.amax(score_history_test)

    print('\n\nOptimal C value = ', opt_params['C'])

    print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

    # Plot the ROC AUC Score

    PlotTuningAccuracy('C', param_values, score_history_train, score_history_test, opt_params['C'], opt_auc_roc)

    

    

#     #

#     # Tune kernel to maximize ROC

#     #

#     param_values = ['linear', 'rbf', 'poly']

#     score_history_train = []

#     score_history_test = []

#     for param_value in param_values:

#         svc = SVC(gamma = opt_params['gamma'],

#                   C = opt_params['C'],

#                   kernel = param_value)

#         svc.fit(TrainX, TrainY.values.ravel())

#         score_history_train.append(roc_auc_score(TrainY, svc.predict(TrainX)))

#         Predicted = svc.predict(TestX)

#         score_history_test.append(roc_auc_score(TestY, Predicted))

#     # Optimal values

#     opt_params['kernel'] = param_values[np.argmax(score_history_test)]

#     opt_auc_roc = np.amax(score_history_test)

#     print('\n\nOptimal kernel value = ', opt_params['kernel'])

#     print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

#     # Plot the ROC AUC Score

#     PlotTuningAccuracy('kernel', param_values, score_history_train, score_history_test, opt_params['kernel'], opt_auc_roc)

    

    

    #

    # Create SVC based on optimal hyperparatmers 

    #

    svc = SVC(gamma = opt_params['gamma'],

              C = opt_params['C'])

#              kernel = opt_params['kernel'])

    

    # Train SVC

    svc = svc.fit(TrainX, TrainY.values.ravel())

    

    # Predict the class for the Test data

    Predicted = svc.predict(TestX)



    # Calculate error, MSE, RMSE

#    mse = mean_squared_error(np.array(TestY, dtype = np.float32), np.array(Predicted, dtype = np.float32))

#    rmse = math.sqrt(mse)

    accuracy = accuracy_score(TestY, Predicted)

    precision, recall, fscore, support = precision_recall_fscore_support(TestY, Predicted, average = None)

    roc_auc = roc_auc_score(TestY, Predicted)

    

    # Display classification report

    print('\n\nClassification Report for Optimal Model:')

    cs_r = classification_report(TestY, Predicted, target_names = label_list)

    print(cs_r)



    # Plot ROC AUC 

    PlotROCCurve(TestY, Predicted)

    

    #Display confusion matrix

    PlotConfusionMatrix(TestY, Predicted, label_list)

    

    # Update the summary table by including the results for this model

    SummaryTable.at['Feature Count', model_name] = TrainX.shape[1]

    SummaryTable.at['Total Records', model_name] = TrainX.shape[0] + TestY.shape[0]

    SummaryTable.at['Optimal C', model_name] = opt_params['C']

    SummaryTable.at['Optimal gamma', model_name] = opt_params['gamma']

#    SummaryTable.at['Optimal kernel', model_name] = opt_params['kernel']

    SummaryTable.at['Accuracy Score', model_name] = round(accuracy, 3)

    SummaryTable.at['ROC AUC Score', model_name] = round(roc_auc, 3)

    SummaryTable.at['Precision - Class 0', model_name] = round(precision[0], 3) 

    SummaryTable.at['Precision - Class 1', model_name] = round(precision[1], 3)

    SummaryTable.at['Recall - Class 0', model_name] = round(recall[0], 3) 

    SummaryTable.at['Recall - Class 1', model_name] = round(recall[1], 3) 

    SummaryTable.at['F1-Score - Class 0', model_name] = round(fscore[0], 3) 

    SummaryTable.at['F1-Score - Class 1', model_name] = round(fscore[1], 3) 

    SummaryTable.at['Support - Class 0', model_name] = round(support[0], 3) 

    SummaryTable.at['Support - Class 1', model_name] = round(support[1], 3) 

    SummaryTable.at['Execution Time', model_name] = round((time.time() - func_start_time), 2)

    display(SummaryTable)

    

    print('Function Execution Time: --- %s seconds ---' % round((time.time() - func_start_time), 2))



    

#

# Create a function to Run SVC and tune it - this is for KFolds

# The hyperparameters to be tuned: C, gamma

# Display Classification Report, ROC Curve, Confusion Matrix 

# Source: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

# Source: https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012

# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

#

@ignore_warnings(category = ConvergenceWarning)

def RunSVCKFolds(model_name, DataX, DataY, Stratified, SummaryTable, label_list):

    

    func_start_time = time.time()

    lbl = model_name + ': ' + SummaryTable.at['Validation Method', model_name]

    opt_params = {'gamma': None,

                  'C': None}

#                  'kernel': None}

    

    

    #

    # create cross validation object and get splits

    #

    if Stratified:

        cv = StratifiedKFold(n_splits = 5, random_state = 10, shuffle = True)

    else:

        cv = KFold(n_splits = 5, random_state = 10, shuffle = True)

    

    cv.get_n_splits(DataX, DataY.values.ravel())

        

        

    #

    # Prepare hyperparamter ranges and scoring

    #

    C_List = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    gamma_list = [0.1, 1, 10, 100]

#    kernel_list = ['linear', 'rbf', 'poly']

    parameter_grid = {'C': C_List,

                      'gamma': gamma_list}

#                      'kernel': kernel_list}

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    

    

    #

    # Create classifer object and run grid search to find optimal paramters

    #

    svc = SVC()

    

    grid_search = GridSearchCV(svc, 

                               param_grid = parameter_grid, 

                               cv = cv,  

                               scoring = scoring, 

                               refit = 'AUC', 

                               return_train_score = True)

    

    grid_search.fit(DataX, DataY.values.ravel())

    

    print('\n\nBest parameters: {}'.format(grid_search.best_params_))

    opt_params['C'] = grid_search.best_params_.get('C')

    opt_params['gamma'] = grid_search.best_params_.get('gamma')

#    opt_params['kernel'] = grid_search.best_params_.get('kernel')

    

    

    #

    # Create SVC based on optimal hyperparatmers 

    #

    svc = grid_search.best_estimator_

    

    # Predict the class for the data

    Predicted = svc.predict(DataX)

    

    # Calculate error, MSE, RMSE

#    mse = mean_squared_error(np.array(TestY, dtype = np.float32), np.array(Predicted, dtype = np.float32))

#    rmse = math.sqrt(mse)

    accuracy = accuracy_score(DataY, Predicted)

    precision, recall, fscore, support = precision_recall_fscore_support(DataY, Predicted, average = None)

    roc_auc = roc_auc_score(DataY, Predicted)

    

    # Display classification report

    print('\n\nClassification Report for Optimal Model:')

    cs_r = classification_report(DataY, Predicted, target_names = label_list)

    print(cs_r)



    # Plot ROC AUC 

    PlotROCCurve(DataY, Predicted)

    

    #Display confusion matrix

    PlotConfusionMatrix(DataY, Predicted, label_list)

    

    # Update the summary table by including the results for this model

    SummaryTable.at['Feature Count', model_name] = DataX.shape[1]

    SummaryTable.at['Total Records', model_name] = DataX.shape[0]

    SummaryTable.at['Optimal C', model_name] = opt_params['C']

    SummaryTable.at['Optimal gamma', model_name] = opt_params['gamma']

#    SummaryTable.at['Optimal kernel', model_name] = opt_params['kernel']

    SummaryTable.at['Accuracy Score', model_name] = round(accuracy, 3)

    SummaryTable.at['ROC AUC Score', model_name] = round(roc_auc, 3)

    SummaryTable.at['Precision - Class 0', model_name] = round(precision[0], 3) 

    SummaryTable.at['Precision - Class 1', model_name] = round(precision[1], 3)

    SummaryTable.at['Recall - Class 0', model_name] = round(recall[0], 3) 

    SummaryTable.at['Recall - Class 1', model_name] = round(recall[1], 3) 

    SummaryTable.at['F1-Score - Class 0', model_name] = round(fscore[0], 3) 

    SummaryTable.at['F1-Score - Class 1', model_name] = round(fscore[1], 3) 

    SummaryTable.at['Support - Class 0', model_name] = round(support[0], 3) 

    SummaryTable.at['Support - Class 1', model_name] = round(support[1], 3) 

    SummaryTable.at['Execution Time', model_name] = round((time.time() - func_start_time), 2)

    display(SummaryTable)

    

    print('Function Execution Time: --- %s seconds ---' % round((time.time() - func_start_time), 2))



    

#

# Create a dataframe to store the results of different models

#

df_SVC_summary_1 = pd.DataFrame({

    'SVC_01':['Support Vector Classifier','Train/Test','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'SVC_02':['Support Vector Classifier','Train/Test - Stratified','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'SVC_03':['Support Vector Classifier','Train/Test','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'SVC_04':['Support Vector Classifier','Train/Test - Stratified','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]})



df_SVC_summary_1.index = ['Algorithm Name',

                          'Validation Method',

                          'Outliers Removed',

                          'Normalized and Scaled',

                          'Feature Type',

                          'Feature Count',

                          'Total Records',

                          'Optimal gamma',

                          'Optimal C',

#                          'Optimal kernel',

                          'Accuracy Score',

                          'ROC AUC Score',

                          'Precision - Class 0',

                          'Precision - Class 1',

                          'Recall - Class 0',

                          'Recall - Class 1',

                          'F1-Score - Class 0',

                          'F1-Score - Class 1',

                          'Support - Class 0',

                          'Support - Class 1',

                          'Execution Time']
RunSVCTestTrain(model_name = 'SVC_01',

                TrainX = df_X_train,

                TestX = df_X_test,

                TrainY = df_Y_train,

                TestY = df_Y_test,

                SummaryTable = df_SVC_summary_1,

                label_list = label_dict.values())
RunSVCTestTrain(model_name = 'SVC_02',

                TrainX = df_X_train_strat,

                TestX = df_X_test_strat,

                TrainY = df_Y_train_strat,

                TestY = df_Y_test_strat,

                SummaryTable = df_SVC_summary_1,

                label_list = label_dict.values())
RunSVCTestTrain(model_name = 'SVC_03',

                TrainX = df_X_norm_train,

                TestX = df_X_norm_test,

                TrainY = df_Y_norm_train,

                TestY = df_Y_norm_test,

                SummaryTable = df_SVC_summary_1,

                label_list = label_dict.values())
RunSVCTestTrain(model_name = 'SVC_04', 

                TrainX = df_X_norm_train_strat,

                TestX = df_X_norm_test_strat,

                TrainY = df_Y_norm_train_strat,

                TestY = df_Y_norm_test_strat,

                SummaryTable = df_SVC_summary_1,

                label_list = label_dict.values())
#

# Create a function to Run Random Forest Classifier and tune it - this is for Train/Test Set

# The hyperparameters to be tuned: max_depth, max_features, bootstrap

# Display Classification Report, ROC Curve, Confusion Matrix 

# Source: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

# Source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

#

@ignore_warnings(category = ConvergenceWarning)

def RunRFTTestTrain(model_name, TrainX, TestX, TrainY, TestY, SummaryTable, label_list):

    

    func_start_time = time.time()

    lbl = model_name + ': ' + SummaryTable.at['Validation Method', model_name]

    opt_params = {'max_depth': None,

                  'max_features': None,

                  'bootstrap': None}

#                  'min_samples_leaf': None,

#                  'min_samples_split': None}

    

    

    #

    # Tune max_depth to maximize ROC

    #

    param_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    score_history_train = []

    score_history_test = []

    for param_value in param_values:

        rft = RandomForestClassifier(max_depth = param_value)

        rft.fit(TrainX, TrainY.values.ravel())

        score_history_train.append(roc_auc_score(TrainY, rft.predict(TrainX)))

        Predicted = rft.predict(TestX)

        score_history_test.append(roc_auc_score(TestY, Predicted))

    # Optimal values

    opt_params['max_depth'] = param_values[np.argmax(score_history_test)]

    opt_auc_roc = np.amax(score_history_test)

    print('\n\nOptimal max_depth value = ', opt_params['max_depth'])

    print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

    # Plot the ROC AUC Score

    PlotTuningAccuracy('max_depth', param_values, score_history_train, score_history_test, opt_params['max_depth'], opt_auc_roc)

    

    

    #

    # Tune max_features to maximize ROC

    #

    param_values = ['auto', 'sqrt']

    score_history_train = []

    score_history_test = []

    for param_value in param_values:

        rft = RandomForestClassifier(max_depth = opt_params['max_depth'],

                  max_features = param_value)

        rft.fit(TrainX, TrainY.values.ravel())

        score_history_train.append(roc_auc_score(TrainY, rft.predict(TrainX)))

        Predicted = rft.predict(TestX)

        score_history_test.append(roc_auc_score(TestY, Predicted))

    # Optimal values

    opt_params['max_features'] = param_values[np.argmax(score_history_test)]

    opt_auc_roc = np.amax(score_history_test)

    print('\n\nOptimal max_features value = ', opt_params['max_features'])

    print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

    # Plot the ROC AUC Score

    PlotTuningAccuracy('max_features', param_values, score_history_train, score_history_test, opt_params['max_features'], opt_auc_roc)

    

    

    #

    # Tune bootstrap to maximize ROC

    #

    param_values = [True, False]

    score_history_train = []

    score_history_test = []

    for param_value in param_values:

        rft = RandomForestClassifier(max_depth = opt_params['max_depth'],

                  max_features = opt_params['max_features'],

                  bootstrap = param_value)

        rft.fit(TrainX, TrainY.values.ravel())

        score_history_train.append(roc_auc_score(TrainY, rft.predict(TrainX)))

        Predicted = rft.predict(TestX)

        score_history_test.append(roc_auc_score(TestY, Predicted))

    # Optimal values

    opt_params['bootstrap'] = param_values[np.argmax(score_history_test)]

    opt_auc_roc = np.amax(score_history_test)

    print('\n\nOptimal bootstrap value = ', opt_params['bootstrap'])

    print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

    # Plot the ROC AUC Score

    PlotTuningAccuracy('bootstrap', param_values, score_history_train, score_history_test, opt_params['bootstrap'], opt_auc_roc) 

    



#     #

#     # Tune min_samples_leaf to maximize ROC

#     #

#     param_values = [1, 2, 4]

#     score_history_train = []

#     score_history_test = []

#     for param_value in param_values:

#         rft = RandomForestClassifier(max_depth = opt_params['max_depth'],

#                   max_features = opt_params['max_features'],

#                   bootstrap = opt_params['bootstrap'],

#                   min_samples_leaf = param_value)

#         rft.fit(TrainX, TrainY.values.ravel())

#         score_history_train.append(roc_auc_score(TrainY, rft.predict(TrainX)))

#         Predicted = rft.predict(TestX)

#         score_history_test.append(roc_auc_score(TestY, Predicted))

#     # Optimal values

#     opt_params['min_samples_leaf'] = param_values[np.argmax(score_history_test)]

#     opt_auc_roc = np.amax(score_history_test)

#     print('\n\nOptimal min_samples_leaf value = ', opt_params['min_samples_leaf'])

#     print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

#     # Plot the ROC AUC Score

#     PlotTuningAccuracy('min_samples_leaf', param_values, score_history_train, score_history_test, opt_params['min_samples_leaf'], opt_auc_roc) 





#     #

#     # Tune min_samples_split to maximize ROC

#     #

#     param_values = [2, 5, 10]

#     score_history_train = []

#     score_history_test = []

#     for param_value in param_values:

#         rft = RandomForestClassifier(max_depth = opt_params['max_depth'],

#                   max_features = opt_params['max_features'],

#                   bootstrap = opt_params['bootstrap'],

#                   min_samples_leaf = opt_params['min_samples_leaf'],

#                   min_samples_split = param_value)

#         rft.fit(TrainX, TrainY.values.ravel())

#         score_history_train.append(roc_auc_score(TrainY, rft.predict(TrainX)))

#         Predicted = rft.predict(TestX)

#         score_history_test.append(roc_auc_score(TestY, Predicted))

#     # Optimal values

#     opt_params['min_samples_split'] = param_values[np.argmax(score_history_test)]

#     opt_auc_roc = np.amax(score_history_test)

#     print('\n\nOptimal min_samples_split value = ', opt_params['min_samples_split'])

#     print('ROC AUC at optimal value = ', round(opt_auc_roc, 3))

#     # Plot the ROC AUC Score

#     PlotTuningAccuracy('min_samples_split', param_values, score_history_train, score_history_test, opt_params['min_samples_split'], opt_auc_roc) 





    #

    # Create RFT based on optimal hyperparatmers 

    #

    rft = RandomForestClassifier(max_depth = opt_params['max_depth'],

             max_features = opt_params['max_features'],

             bootstrap = opt_params['bootstrap'])

#             min_samples_leaf = opt_params['min_samples_leaf'],

#             min_samples_split = opt_params['min_samples_split'])

    

    # Train RFT

    rft = rft.fit(TrainX, TrainY.values.ravel())

    

    # Predict the class for the Test data

    Predicted = rft.predict(TestX)



    # Calculate error, MSE, RMSE

#    mse = mean_squared_error(np.array(TestY, dtype = np.float32), np.array(Predicted, dtype = np.float32))

#    rmse = math.sqrt(mse)

    accuracy = accuracy_score(TestY, Predicted)

    precision, recall, fscore, support = precision_recall_fscore_support(TestY, Predicted, average = None)

    roc_auc = roc_auc_score(TestY, Predicted)

    

    # Display classification report

    print('\n\nClassification Report for Optimal Model:')

    cs_r = classification_report(TestY, Predicted, target_names = label_list)

    print(cs_r)



    # Plot ROC AUC 

    PlotROCCurve(TestY, Predicted)

    

    #Display confusion matrix

    PlotConfusionMatrix(TestY, Predicted, label_list)

    

    # Update the summary table by including the results for this model

    SummaryTable.at['Feature Count', model_name] = TrainX.shape[1]

    SummaryTable.at['Total Records', model_name] = TrainX.shape[0] + TestY.shape[0]

    SummaryTable.at['Optimal max_depth', model_name] = opt_params['max_depth']

    SummaryTable.at['Optimal max_features', model_name] = opt_params['max_features']

    SummaryTable.at['Optimal bootstrap', model_name] = opt_params['bootstrap']

#    SummaryTable.at['Optimal min_samples_leaf', model_name] = opt_params['min_samples_leaf']

#    SummaryTable.at['Optimal min_samples_split', model_name] = opt_params['min_samples_split']

    SummaryTable.at['Accuracy Score', model_name] = round(accuracy, 3)

    SummaryTable.at['ROC AUC Score', model_name] = round(roc_auc, 3)

    SummaryTable.at['Precision - Class 0', model_name] = round(precision[0], 3) 

    SummaryTable.at['Precision - Class 1', model_name] = round(precision[1], 3)

    SummaryTable.at['Recall - Class 0', model_name] = round(recall[0], 3) 

    SummaryTable.at['Recall - Class 1', model_name] = round(recall[1], 3) 

    SummaryTable.at['F1-Score - Class 0', model_name] = round(fscore[0], 3) 

    SummaryTable.at['F1-Score - Class 1', model_name] = round(fscore[1], 3) 

    SummaryTable.at['Support - Class 0', model_name] = round(support[0], 3) 

    SummaryTable.at['Support - Class 1', model_name] = round(support[1], 3) 

    SummaryTable.at['Execution Time', model_name] = round((time.time() - func_start_time), 2)

    display(SummaryTable)

    

    print('Function Execution Time: --- %s seconds ---' % round((time.time() - func_start_time), 2))



    

#

# Create a function to Run RFT and tune it - this is for KFolds

# The hyperparameters to be tuned: max_depth, max_features, bootstrap

# Display Classification Report, ROC Curve, Confusion Matrix 

# Source: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

# Source: https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012

# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

#

@ignore_warnings(category = ConvergenceWarning)

def RunRFTKFolds(model_name, DataX, DataY, Stratified, SummaryTable, label_list):

    

    func_start_time = time.time()

    lbl = model_name + ': ' + SummaryTable.at['Validation Method', model_name]

    opt_params = {'max_depth': None,

                  'max_features': None,

                  'bootstrap': None}

#                  'min_samples_leaf': None,

#                  'min_samples_split': None}

    

    

    #

    # create cross validation object and get splits

    #

    if Stratified:

        cv = StratifiedKFold(n_splits = 5, random_state = 10, shuffle = True)

    else:

        cv = KFold(n_splits = 5, random_state = 10, shuffle = True)

    

    cv.get_n_splits(DataX, DataY.values.ravel())

        

        

    #

    # Prepare hyperparamter ranges and scoring

    #

    max_depth_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    max_features_list = ['auto', 'sqrt']

    bootstrap_list = [True, False]

#    min_samples_leaf_list = [1, 2, 4]

#    min_samples_split = [2, 5, 10]

    parameter_grid = {'max_depth': max_depth_list,

                      'max_features': max_features_list,

                      'bootstrap': bootstrap_list}

#                      'min_samples_leaf': min_samples_leaf_list,

#                      'min_samples_split': min_samples_split}

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    

    

    #

    # Create classifer object and run grid search to find optimal paramters

    #

    rft = RandomForestClassifier()

    

    grid_search = GridSearchCV(rft, 

                               param_grid = parameter_grid, 

                               cv = cv,  

                               scoring = scoring, 

                               refit = 'AUC', 

                               return_train_score = True)

    

    grid_search.fit(DataX, DataY.values.ravel())

    

    print('\n\nBest parameters: {}'.format(grid_search.best_params_))

    opt_params['max_depth'] = grid_search.best_params_.get('max_depth')

    opt_params['max_features'] = grid_search.best_params_.get('max_features')

    opt_params['bootstrap'] = grid_search.best_params_.get('bootstrap')

#    opt_params['min_samples_leaf'] = grid_search.best_params_.get('min_samples_leaf')

#    opt_params['min_samples_split'] = grid_search.best_params_.get('min_samples_split')

    

    

    #

    # Create RFT based on optimal hyperparatmers 

    #

    rft = grid_search.best_estimator_

    

    # Predict the class for the data

    Predicted = rft.predict(DataX)

    

    # Calculate error, MSE, RMSE

#    mse = mean_squared_error(np.array(TestY, dtype = np.float32), np.array(Predicted, dtype = np.float32))

#    rmse = math.sqrt(mse)

    accuracy = accuracy_score(DataY, Predicted)

    precision, recall, fscore, support = precision_recall_fscore_support(DataY, Predicted, average = None)

    roc_auc = roc_auc_score(DataY, Predicted)

    

    # Display classification report

    print('\n\nClassification Report for Optimal Model:')

    cs_r = classification_report(DataY, Predicted, target_names = label_list)

    print(cs_r)



    # Plot ROC AUC 

    PlotROCCurve(DataY, Predicted)

    

    #Display confusion matrix

    PlotConfusionMatrix(DataY, Predicted, label_list)

    

    # Update the summary table by including the results for this model

    SummaryTable.at['Feature Count', model_name] = DataX.shape[1]

    SummaryTable.at['Total Records', model_name] = DataX.shape[0]

    SummaryTable.at['Optimal max_depth', model_name] = opt_params['max_depth']

    SummaryTable.at['Optimal max_features', model_name] = opt_params['max_features']

    SummaryTable.at['Optimal bootstrap', model_name] = opt_params['bootstrap']

#    SummaryTable.at['Optimal min_samples_leaf', model_name] = opt_params['min_samples_leaf']

#    SummaryTable.at['Optimal min_samples_split', model_name] = opt_params['min_samples_split']

    SummaryTable.at['Accuracy Score', model_name] = round(accuracy, 3)

    SummaryTable.at['ROC AUC Score', model_name] = round(roc_auc, 3)

    SummaryTable.at['Precision - Class 0', model_name] = round(precision[0], 3) 

    SummaryTable.at['Precision - Class 1', model_name] = round(precision[1], 3)

    SummaryTable.at['Recall - Class 0', model_name] = round(recall[0], 3) 

    SummaryTable.at['Recall - Class 1', model_name] = round(recall[1], 3) 

    SummaryTable.at['F1-Score - Class 0', model_name] = round(fscore[0], 3) 

    SummaryTable.at['F1-Score - Class 1', model_name] = round(fscore[1], 3) 

    SummaryTable.at['Support - Class 0', model_name] = round(support[0], 3) 

    SummaryTable.at['Support - Class 1', model_name] = round(support[1], 3) 

    SummaryTable.at['Execution Time', model_name] = round((time.time() - func_start_time), 2)

    display(SummaryTable)

    

    print('Function Execution Time: --- %s seconds ---' % round((time.time() - func_start_time), 2))



    

#

# Create a dataframe to store the results of different models

#

df_RFT_summary_1 = pd.DataFrame({

    'RFT_01':['Random Forest','Train/Test','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_02':['Random Forest','Train/Test - Stratified','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_03':['Random Forest','Train/Test','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_04':['Random Forest','Train/Test - Stratified','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_05':['Random Forest','KFolds','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_06':['Random Forest','KFolds - Stratified','Yes','No','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_07':['Random Forest','KFolds','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_08':['Random Forest','KFolds - Stratified','Yes','Yes','Engineered Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]})



df_RFT_summary_1.index = ['Algorithm Name',

                          'Validation Method',

                          'Outliers Removed',

                          'Normalized and Scaled',

                          'Feature Type',

                          'Feature Count',

                          'Total Records',

                          'Optimal max_depth',

                          'Optimal max_features',

                          'Optimal bootstrap',

#                          'Optimal min_samples_leaf',

#                          'Optimal min_samples_split',

                          'Accuracy Score',

                          'ROC AUC Score',

                          'Precision - Class 0',

                          'Precision - Class 1',

                          'Recall - Class 0',

                          'Recall - Class 1',

                          'F1-Score - Class 0',

                          'F1-Score - Class 1',

                          'Support - Class 0',

                          'Support - Class 1',

                          'Execution Time']





#

# Create a dataframe to store the results of different models

#

df_RFT_summary_2 = pd.DataFrame({

    'RFT_09':['Random Forest','Train/Test','Yes','N/A','TF-IDF Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],

    'RFT_10':['Random Forest','Train/Test - Stratified','Yes','N/A','TF-IDF Features',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]})



df_RFT_summary_2.index = ['Algorithm Name',

                          'Validation Method',

                          'Outliers Removed',

                          'Normalized and Scaled',

                          'Feature Type',

                          'Feature Count',

                          'Total Records',

                          'Optimal max_depth',

                          'Optimal max_features',

                          'Optimal bootstrap',

 #                         'Optimal min_samples_leaf',

 #                         'Optimal min_samples_split',

                          'Accuracy Score',

                          'ROC AUC Score',

                          'Precision - Class 0',

                          'Precision - Class 1',

                          'Recall - Class 0',

                          'Recall - Class 1',

                          'F1-Score - Class 0',

                          'F1-Score - Class 1',

                          'Support - Class 0',

                          'Support - Class 1',

                          'Execution Time']
RunRFTTestTrain(model_name = 'RFT_01',

                TrainX = df_X_train,

                TestX = df_X_test,

                TrainY = df_Y_train,

                TestY = df_Y_test,

                SummaryTable = df_RFT_summary_1,

                label_list = label_dict.values())
RunRFTTestTrain(model_name = 'RFT_02',

                TrainX = df_X_train_strat,

                TestX = df_X_test_strat,

                TrainY = df_Y_train_strat,

                TestY = df_Y_test_strat,

                SummaryTable = df_RFT_summary_1,

                label_list = label_dict.values())
RunRFTTestTrain(model_name = 'RFT_03',

                TrainX = df_X_norm_train,

                TestX = df_X_norm_test,

                TrainY = df_Y_norm_train,

                TestY = df_Y_norm_test,

                SummaryTable = df_RFT_summary_1,

                label_list = label_dict.values())
RunRFTTestTrain(model_name = 'RFT_04', 

                TrainX = df_X_norm_train_strat,

                TestX = df_X_norm_test_strat,

                TrainY = df_Y_norm_train_strat,

                TestY = df_Y_norm_test_strat,

                SummaryTable = df_RFT_summary_1,

                label_list = label_dict.values())
RunRFTKFolds(model_name = 'RFT_05',

             DataX = df_X,

             DataY = df_Y,

             Stratified = False,

             SummaryTable = df_RFT_summary_1,

             label_list = label_dict.values())
RunRFTKFolds(model_name = 'RFT_06',

             DataX = df_X,

             DataY = df_Y,

             Stratified = True,

             SummaryTable = df_RFT_summary_1,

             label_list = label_dict.values())
RunRFTKFolds(model_name = 'RFT_07',

             DataX = df_X_norm,

             DataY = df_Y_norm,

             Stratified = False,

             SummaryTable = df_RFT_summary_1,

             label_list = label_dict.values())
RunRFTKFolds(model_name = 'RFT_08', 

             DataX = df_X_norm,

             DataY = df_Y_norm,

             Stratified = True,

             SummaryTable = df_RFT_summary_1,

             label_list = label_dict.values())
RunRFTTestTrain(model_name = 'RFT_09',

                TrainX = df_X_tfidf_train,

                TestX = df_X_tfidf_test,

                TrainY = df_Y_tfidf_train,

                TestY = df_Y_tfidf_test,

                SummaryTable = df_RFT_summary_2,

                label_list = label_dict.values())
RunRFTTestTrain(model_name = 'RFT_10',

                TrainX = df_X_tfidf_train_strat,

                TestX = df_X_tfidf_test_strat,

                TrainY = df_Y_tfidf_train_strat,

                TestY = df_Y_tfidf_test_strat,

                SummaryTable = df_RFT_summary_2,

                label_list = label_dict.values())