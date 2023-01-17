# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = "/kaggle/input/jigsaw-comment-classification-notebook-one/"

os.listdir(DATA_PATH)
TRAIN_PATH = DATA_PATH + "notebook_one_train_data.csv"

VAL_PATH = DATA_PATH + "notebook_one_val_data.csv"

TEST_PATH = DATA_PATH + "notebook_one_test_data.csv"



train_data = pd.read_csv(TRAIN_PATH)

val_data = pd.read_csv(VAL_PATH)

test_data = pd.read_csv(TEST_PATH)
train_data.head()
val_data.head()
test_data.head()
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-test-translated/"

os.listdir(DATA_PATH)
TRANSLATED_VAL_PATH = DATA_PATH + "jigsaw_miltilingual_valid_translated.csv"

TRANSLATED_TEST_PATH = DATA_PATH + "jigsaw_miltilingual_test_translated.csv"



trans_val_data = pd.read_csv(TRANSLATED_VAL_PATH)

trans_test_data = pd.read_csv(TRANSLATED_TEST_PATH)
trans_val_data.head()
trans_test_data.head()
val_data["translated_comment"] = trans_val_data.translated

test_data["translated_comment"] = trans_test_data.translated
train_non_eng_sentences = train_data[(train_data.lang_code != 'en')]
print("Total number of comments which are Non-English: ",

      train_non_eng_sentences.shape[0])



print("Total number of languages other than English: ",

      len(train_non_eng_sentences.lang_code.unique()))



print("Average number of comments in Non-English Languages: ",

      train_non_eng_sentences.lang_code.value_counts().mean())
import seaborn as sns

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), )

prob = train_non_eng_sentences.lang_code.value_counts(normalize = True)

threshold = .01

mask = prob > threshold

tail_prob = prob.loc[~mask].sum()

prob = prob.loc[mask]

prob['other'] = tail_prob

ax = sns.barplot(x=prob.index, y=prob.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")

ax.set_xlabel("Languages")

ax.set_ylabel("Comment count")

ax.set_title('Non English language comment count in training data (>1% of total)', fontsize=14)

fig.show()
train_non_eng_toxic = train_data[(train_data.lang_code != 'en') & (train_data.label == 1)]
print("Total toxic comments available in the training dataset: ", train_data.label.value_counts()[1])

print("Total toxic comments that are non english: ", train_non_eng_toxic.shape[0])

print("Percetage contribution: {:.2f} %".format(100*(

    train_non_eng_toxic.shape[0]/train_data.label.value_counts()[1])))
! pip install googletrans
from googletrans import Translator



translator = Translator()



def translate(sentence):

    return translator.translate(sentence).text
translated_train_sents = []



train_non_eng_toxic_comment_list = train_non_eng_toxic.comment_text.to_list()

print("Total length: ", len(train_non_eng_toxic_comment_list))



def clean_text_for_translate(comment):

    if type(comment) == str:

        x = "".join(x for x in comment if x.isprintable())        

        return x.replace("\n", " ")

    else:

        return ""



count = 0 



while count < len(train_non_eng_toxic_comment_list):

    sent = clean_text_for_translate(train_non_eng_toxic_comment_list[count])

    translated_train_sents.append(translate(sent))

    count += 1
train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'comment_text']= translated_train_sents

train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'lang_code'] = 'en'

train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'lang_name'] = 'English'
train_data.drop(train_data[(train_data['lang_code'] != 'en') & (train_data['label'] == 0)].index, inplace = True)
train_data.shape
!pip install beautifulsoup4

!pip install contractions
from bs4 import BeautifulSoup

import re

import contractions

import unicodedata





def remove_html(text):

    soup = BeautifulSoup(text, "html.parser")

    html_free = soup.get_text()

    return html_free



def remove_url(text):

    return re.sub(r'http\S', ' ', text)



def remove_digits_spec_chars(text):

    return re.sub(r'[^a-zA-Z]', " ", text)



def to_lower_case(text):

    return text.lower()



def remove_extra_spaces(text):

    return re.sub(r'\s\s+', " ", text)



def remove_next_line(text):

    return re.sub(r'[\n\r]+', " ", text)

    

def remove_non_ascii(comment):

    """

    Remove non-ASCII characters from list of tokenized words

    """

    ascii_string = unicodedata.normalize('NFKD', comment).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return ascii_string

    

def remove_between_square_brackets(comment):

    result = re.sub('\[[^]]*\]', '', comment)

    return result



def replace_contractions(comment):

    """

    Replace contractions in string of text

    """

    contraction_free = contractions.fix(comment)

    return contraction_free



def clean_comment(comment):

    comment = remove_non_ascii(comment)

    comment = remove_next_line(comment)

    comment = replace_contractions(comment)

    comment = remove_url(comment)

    comment = remove_html(comment)

    comment = remove_between_square_brackets(comment)

    comment = remove_digits_spec_chars(comment)

    comment = remove_extra_spaces(comment)

    comment = to_lower_case(comment)

    return comment.strip()
train_data["cleaned_text"] = train_data["comment_text"].apply(lambda row: clean_comment(row))

val_data["cleaned_text"] = val_data["translated_comment"].apply(lambda row: clean_comment(row))

test_data["cleaned_text"] = test_data["translated_comment"].apply(lambda row: clean_comment(row))
!pip install --upgrade vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



analyzer = SentimentIntensityAnalyzer()



def get_pos_polarity(comment):

    return analyzer.polarity_scores(comment)



def get_comment_length(comment):

    try:

        return len(comment.split())

    except:

        return 0
sentiment = train_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))

train_data = pd.concat([train_data,sentiment.apply(pd.Series)],1)

train_data.columns = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate', 'lang_code', 'lang_name', 'country', 'label', 'cleaned_text',

                      'neg_pol','neutral_pol', 'pos_pol', 'compound_pol']
sentiment = val_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))

val_data = pd.concat([val_data,sentiment.apply(pd.Series)],1)

val_data.columns = ['id', 'comment_text', 'lang', 'toxic', 'lang_name', 'country',

                    'translated_comment', 'cleaned_text','neg_pol', 

                    'neutral_pol', 'pos_pol', 'compound_pol']
sentiment = test_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))

test_data = pd.concat([test_data,sentiment.apply(pd.Series)],1)

test_data.columns = ['id', 'content', 'lang', 'lang_name', 'country', 'translated_comment',

                    'cleaned_text','neg_pol', 'neutral_pol', 'pos_pol', 'compound_pol']
train_data["comment_len"] = train_data["cleaned_text"].apply(lambda row: get_comment_length(row))

val_data["comment_len"] = val_data["cleaned_text"].apply(lambda row: get_comment_length(row))

test_data["comment_len"] = test_data["cleaned_text"].apply(lambda row: get_comment_length(row))
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), )

ax = sns.kdeplot(train_data.comment_len, shade=True, label = "Training")

ax = sns.kdeplot(val_data.comment_len, shade=True, label = "Validation")

ax = sns.kdeplot(test_data.comment_len, shade=True, label = "Testing")

ax.set_title('Density distribution of comment length over different datasets')

ax.set_xlabel("Comment length")

ax.set_ylabel("Density")

f.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

sns.kdeplot(train_data[train_data.label ==1].comment_len, shade=True, label = "Toxic Comment", ax=ax1)

sns.kdeplot(train_data[train_data.label ==0].comment_len, shade=True, label = "Non-Toxic Comment", ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Comment length")

ax1.set_ylabel("Density")



sns.kdeplot(val_data[val_data.toxic ==1].comment_len, shade=True, label = "Toxic Comment", ax=ax2)

sns.kdeplot(val_data[val_data.toxic ==0].comment_len, shade=True, label = "Non-Toxic Comment", ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Comment length")

ax2.set_ylabel("Density")





f.suptitle("Density distribution of comment length over labels", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))



# Training data 

var = 'comment_len'

tmp = pd.concat([train_data['label'], train_data[var]], axis=1)

sns.boxplot(x='label', y=var, data=tmp, fliersize=5, ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Labels")

ax1.set_ylabel("Comment Length")

ax1.set_xticklabels(["Non-Toxic", "Toxic"])



tmp = pd.concat([val_data['toxic'], val_data[var]], axis=1)

sns.boxplot(x='toxic', y=var, data=tmp, fliersize=5, ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Labels")

ax2.set_ylabel("Comment Length")

ax2.set_xticklabels(["Non-Toxic", "Toxic"])



fig = sns.boxplot(y=var, data=test_data, fliersize=5, ax=ax3)

ax3.set_title('Test data')

ax3.set_ylabel("Comment Length")



f.suptitle("Comment count across datasets", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
Q1 = train_data.comment_len.quantile(0.25)

Q3 = train_data.comment_len.quantile(0.75)

IQR = Q3 - Q1



print("Total comment length outliers in training data: ", 

      ((train_data.comment_len < (Q1 - 1.5 * IQR)) | (train_data.comment_len > (Q3 + 1.5 * IQR))).sum())



print("Total comment length outliers in toxic comments of training data: ", 

      ((train_data.loc[train_data.label == 1]['comment_len'] < (Q1 - 1.5 * IQR))|(

          train_data.loc[train_data.label == 1]['comment_len'] > (Q3 + 1.5 * IQR))).sum())



print("Total comment length outliers in non-toxic comments of training data: ", 

      ((train_data.loc[train_data.label == 0]['comment_len'] < (Q1 - 1.5 * IQR))|(

          train_data.loc[train_data.label == 0]['comment_len'] > (Q3 + 1.5 * IQR))).sum())





print("-"*80)



Q1 = val_data.comment_len.quantile(0.25)

Q3 = val_data.comment_len.quantile(0.75)

IQR = Q3 - Q1



print("Total comment length outliers in validation data: ", 

      ((val_data.comment_len < (Q1 - 1.5 * IQR)) | (val_data.comment_len > (Q3 + 1.5 * IQR))).sum())



print("Total comment length outliers in toxic comments of validation data: ", 

      ((val_data.loc[val_data.toxic == 1]['comment_len'] < (Q1 - 1.5 * IQR))|(

          val_data.loc[val_data.toxic == 1]['comment_len'] > (Q3 + 1.5 * IQR))).sum())



print("Total comment length outliers in non-toxic comments of validation data: ", 

      ((val_data.loc[val_data.toxic == 0]['comment_len'] < (Q1 - 1.5 * IQR))|(

          val_data.loc[val_data.toxic == 0]['comment_len'] > (Q3 + 1.5 * IQR))).sum())





print("-"*80)



Q1 = test_data.comment_len.quantile(0.25)

Q3 = test_data.comment_len.quantile(0.75)

IQR = Q3 - Q1



print("Total comment length outliers in testing data: ", 

      ((test_data.comment_len < (Q1 - 1.5 * IQR)) | (test_data.comment_len > (Q3 + 1.5 * IQR))).sum())
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

sns.kdeplot(train_data[train_data.label ==1].neg_pol, shade=True, label = "Toxic Comment", ax=ax1)

sns.kdeplot(train_data[train_data.label ==0].neg_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Negative Polarity")

ax1.set_ylabel("Density")



sns.kdeplot(val_data[val_data.toxic ==1].neg_pol, shade=True, label = "Toxic Comment", ax=ax2)

sns.kdeplot(val_data[val_data.toxic ==0].neg_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Negative Polarity")

ax2.set_ylabel("Density")





f.suptitle("Density distribution of Negative Polarity over labels", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

sns.kdeplot(train_data[train_data.label ==1].pos_pol, shade=True, label = "Toxic Comment", ax=ax1)

sns.kdeplot(train_data[train_data.label ==0].pos_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Positive Polarity")

ax1.set_ylabel("Density")



sns.kdeplot(val_data[val_data.toxic ==1].pos_pol, shade=True, label = "Toxic Comment", ax=ax2)

sns.kdeplot(val_data[val_data.toxic ==0].pos_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Positive Polarity")

ax2.set_ylabel("Density")





f.suptitle("Density distribution of Positive Polarity over labels", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

sns.kdeplot(train_data[train_data.label ==1].neutral_pol, shade=True, label = "Toxic Comment", ax=ax1)

sns.kdeplot(train_data[train_data.label ==0].neutral_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Neutral Polarity")

ax1.set_ylabel("Density")



sns.kdeplot(val_data[val_data.toxic ==1].neutral_pol, shade=True, label = "Toxic Comment", ax=ax2)

sns.kdeplot(val_data[val_data.toxic ==0].neutral_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Neutral Polarity")

ax2.set_ylabel("Density")





f.suptitle("Density distribution of Neutral Polarity over labels", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

sns.kdeplot(train_data[train_data.label ==1].compound_pol, shade=True, label = "Toxic Comment", ax=ax1)

sns.kdeplot(train_data[train_data.label ==0].compound_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)

ax1.set_title('Training data')

ax1.set_xlabel("Compound Polarity")

ax1.set_ylabel("Density")



sns.kdeplot(val_data[val_data.toxic ==1].compound_pol, shade=True, label = "Toxic Comment", ax=ax2)

sns.kdeplot(val_data[val_data.toxic ==0].compound_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)

ax2.set_title('Validation data')

ax2.set_xlabel("Compound Polarity")

ax2.set_ylabel("Density")





f.suptitle("Density distribution of Compound Polarity over labels", fontsize = 22)

f.tight_layout()

f.subplots_adjust(top=0.8)

f.show()
train_data.to_csv("notebook_two_train_data.csv", index = False)

val_data.to_csv("notebook_two_val_data.csv", index = False)

test_data.to_csv("notebook_two_test_data.csv", index = False)