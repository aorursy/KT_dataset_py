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
# Importing libraries & magic functions



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
# Putting Test and Train Set back together



dstest = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.csv')

dstrain = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv')



dstest.shape

dstrain.shape



ds = pd.concat([dstest, dstrain],ignore_index=True)

ds.shape
# First glimpse at data

ds.head()



# Statistics Summary - Numerical variables

ds.describe()



# check types

ds.dtypes



# check for missing data

ds.isna().sum()



# check for duplicates

duplicate_ds = ds[ds.duplicated()]

duplicate_ds

duplicate_ds.shape

# drop rows with missing values

ds = ds.dropna()
ds.isna().sum()

ds.shape
# change date to datetime format



ds.date = pd.to_datetime(ds.date)

ds.dtypes
# set date as index



ds.set_index('date', inplace=True)

ds.info()

ds.head()
ds.condition.value_counts()
duplicate_review = ds[ds.duplicated(['review'])]

duplicate_review
ds.review.nunique()
regex_pattern = r'I wrote my first report in Mid-October(?!$)'

ds[ds['review'].str.contains(regex_pattern)]
# Many reviews are duplicated and have been assigned to 2 drugNames where 1 is usually a broader term for the drug and the second one the brand name. 

# We will therefore remove the duplicates and always keep the first value.



ds = ds.drop_duplicates(subset='review', keep="first")

ds.shape
ds.describe()
# checking for outliers



#sns.boxplot(ds.usefulCount)

sns.boxplot(ds.rating)

# check distribution/correlation/outliers

sns.pairplot(ds)
ds.head()
ds.groupby('condition').agg('sum')
ds.condition.unique()
ds.shape
ds_comment = ds[ds['condition'].str.contains('comment')]

ds_comment
# Dropping rows that contain incorrect information 



ds = ds[~ds['condition'].str.contains('comment')]

ds.shape
ds_clean = ds
ds_clean.condition.nunique()

ds_clean.drugName.nunique()
# Distribution of drugs within conditions



ds_drugs_per_cond = ds_clean.groupby('condition').drugName.nunique().sort_values(ascending=False)

pd.DataFrame (data=ds_drugs_per_cond)

ds_drugs_per_cond = ds_drugs_per_cond.reset_index()
# Plotting Distribution of drugs within conditions

plt.figure(figsize=(10,8))

sns.barplot(x='drugName', y='condition', data=ds_drugs_per_cond[0:10], color='lightblue')

plt.box(False)

plt.xlabel("", fontsize = 12)

plt.ylabel("", fontsize = 14)

plt.title("Top 10 Number of Drugs by Condition", fontsize = 18)

plt.show()
ds_drugs_per_cond.head()

ds_drugs_per_cond['drugName'].describe()
#Plotting Distribution Drugs Numbers per Condition

mean=ds_drugs_per_cond['drugName'].mean()

plt.figure(figsize=(11,5))

#sns.set_style("white")

ax = sns.distplot(ds_drugs_per_cond['drugName'],color='lightblue')

ax.axvline(mean, color='r', linestyle='--')

plt.box(False)

plt.legend({'Mean= 8.65':mean})

plt.xlabel("\n Number of different drugs", fontsize = 12)

plt.ylabel("", fontsize = 12)

plt.title("Distribution Drugs per Condition", fontsize = 16)

plt.show()
# Spread? Standard Deviation
# Displaying number of reviewed drugs by condition



ds_reviews_per_condition = ds_clean.groupby('condition').agg({'review':'count'})

ds_reviews_per_condition = ds_reviews_per_condition.sort_values(by='review', ascending=False)#[0:20]

ds_reviews_per_condition = ds_reviews_per_condition.reset_index()

ds_reviews_per_condition.head()
# Plotting reviews per conditions 

plt.figure(figsize=(12,6))

sns.barplot(x='review', y='condition', data=ds_reviews_per_condition[0:10], color='lightblue')

plt.box(False)

plt.xlabel("", fontsize = 12)

plt.ylabel("", fontsize = 14)

plt.title("Top 10 Number of Reviews by Condition\n", fontsize = 18)

plt.show()
# Group Rating and Condition

ds_rating = ds.groupby('condition').agg({'rating':'mean', 'review':'count'}).sort_values(by='rating',ascending=False)

ds_rating_150 = ds_rating[ds_rating.review>150] # we want to exclude those ratings that only received 1 review so we set the threshold approx. to the mean 

ds_rating_150 = ds_rating_150.reset_index()
ds_rating_150.head()
ds_rating_drug = pd.merge(left=ds_rating_150,right=ds_drugs_per_cond, how='left', left_on='condition', right_on='condition')

ds_rating_drug.head()
# Correlation between rating scores and number of drug - grouped by condition

np.corrcoef(ds_rating_drug["rating"], ds_rating_drug["drugName"])

sns.scatterplot(x='rating', y='drugName', data=ds_rating_drug, color='lightblue')

plt.box(False)

plt.xlabel("Rating", fontsize = 10)

plt.ylabel("Number of Drugs\n", fontsize = 12)

plt.title("Rating and Number of Drugs", fontsize = 15)

plt.show()
ds_rating_150.head()
ds_rating_150['rating'].describe()
# Plotting Rating for more than 150 reviews received

mean=ds_rating_150['rating'].mean()



ax=sns.distplot(ds_rating_150['rating'], color='lightblue')

ax.axvline(mean, color='r', linestyle='--')



plt.legend({'Mean= 7.152':mean})



plt.box(False)

plt.xlabel("Rating", fontsize = 10)

plt.ylabel("", fontsize = 12)

plt.title("Rating Distribution", fontsize = 15)

plt.show()
ds_outlier = ds_rating_150[ds_rating_150["review"]>15000]

# Remove outlier



ds_rating_150 = ds_rating_150.drop([ds_rating_150.index[91]])
# Correlation between rating scores and numbe rof reviews received - grouped by condition

np.corrcoef(ds_rating_150["rating"], ds_rating_150["review"])

sns.scatterplot(x='rating', y='review', data=ds_rating_150, color='lightblue')

plt.box(False)

plt.xlabel("Rating", fontsize = 10)

plt.ylabel("Number of Reviews\n", fontsize = 12)

plt.title("Rating and Number of Reviews", fontsize = 15)

plt.show()
ds_rating_150.head()
#ds_clean['reviews_per_cond'] = ds_rating["review"]

ds_merged_left = pd.merge(left=ds_clean,right=ds_rating, how='left', left_on='condition', right_on='condition')



ds_merged_left.head()

ds_merged_left.shape

ds_merged_left.isna().sum()

ds_merged_left_150 = ds_merged_left[ds_merged_left['review_y'] > 150]

ds_merged_left_150.head()

ds_merged_left_150.shape
# Renaming columns

ds_merged_left_150.columns



ds_merged_left_150 = ds_merged_left_150.rename(columns={'rating_y':'Mean Rating',

                        'review_y':'Number of Reviews per condition'})
ds_merged_left_150.head()
ds_time['Year'] = ds_time.index.year
# Time Series

ds_time= ds_clean.sort_index()

ds_time.head()
import statsmodels.api as sm

from statsmodels.formula.api import ols



mod=ols('rating_x  ~ condition',data=ds_merged_left_150).fit()



aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])

aov_table['EtaSq'] = [esq_sm, 'NaN']

print(aov_table)

aov_table
#plt.boxplot(scores~group,data=data)

mod.summary()


ds_rating_150.head()

ds_rating_150 = ds_rating_150.replace({'mance Anxiety': 'Anxiety'})
# Plotting Ratings per condition 

plt.figure(figsize=(10,5))

sns.barplot(x='condition', y='rating', data=ds_rating_150[0:5].sort_values(by='rating', ascending=False), color='lightgreen')

plt.box(False)

plt.xlabel("\nCondition", fontsize = 14)

plt.ylabel("Average Rating", fontsize = 14)

plt.title("5 best Ratings by Condition", fontsize = 20)

#plt.setp(ax.get_xticklabels(), fontsize=14)



plt.show()

# Plotting Ratings per condition

plt.figure(figsize=(12,5))

sns.barplot(x='condition', y='rating', data=ds_rating_150[-6:-1].sort_values(by='rating', ascending= False), color="darkred")

#sns.set(rc={'figure.figsize':(22,10)})

plt.box(False)

plt.xlabel("\nCondition", fontsize = 14)

plt.ylabel("Average Rating", fontsize = 14)

plt.title("5 worst Ratings by Condition", fontsize = 20)

#plt.setp(ax.get_xticklabels(), fontsize=14)

axes = plt.gca()

#axes.set_xlim([xmin,xmax])

axes.set_ylim([0,10])

plt.show()

ds_rating.head()
ds_clean.head()
ds_time_drug = ds_time.groupby('Year')['drugName'].nunique()

ds_time_drug

ds_time_drug.plot(kind='line')

plt.title('Number of Drugs by Year')
ds_time_review = ds_time.groupby('Year')['review'].agg(['count'])
ds_time_review.plot(kind='line')

plt.title('Number of Reviews collected per Year')
ds_time_rating = ds_time.groupby('Year')['rating'].agg(['mean'])

ds_time_rating
ds_time_rating.plot(kind='line')

plt.title('Average Rating Score by Year')
ds_merged_left_150.head()
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from PIL import Image
def generate_wordcloud(ds_merged_left_150):

    

    stopwords_list = stopwords.words('english') + list(STOPWORDS) 

    

    raw_text = " ".join(ds_merged_left_150['review_x'].values)

    

    #mask = np.array(Image.open("/kaggle/input/lalalaa/pill.png"))

    

    wc = WordCloud(stopwords=stopwords_list, background_color="white",width= 1600, height=800, max_words=150).generate(raw_text)

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    fig = plt.gcf()

    fig.set_size_inches(16, 8)

    plt.show()



reviews_by_comments = ds_merged_left_150.sort_values(by="usefulCount", ascending=False)



top_100_useful_comments = reviews_by_comments.head(100)



generate_wordcloud(ds_merged_left_150)



generate_wordcloud(top_100_useful_comments)

#importing libraries

import spacy

from spacy.lang.en.examples import sentences 

import pandas as pd

import numpy as np

import nltk

from nltk.tokenize.toktok import ToktokTokenizer

import re

from bs4 import BeautifulSoup

import unicodedata
nlp = spacy.load('en_core_web_sm', parse = True, tag=True, entity=True)

#nlp = spacy.load()

#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, entity=True)

tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

stopword_list.remove('no')

stopword_list.remove('not')
# Remove HTML



def strip_html_tags(text):

    soup = BeautifulSoup(text, "html.parser")

    stripped_text = soup.get_text()

    return stripped_text



strip_html_tags('<html><h2>Some important text</h2></html>')
# Remove accented characters



def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



remove_accented_chars('Sómě Áccěntěd těxt')
# Remove special characters



def remove_special_characters(text, remove_digits=False):

    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'

    text = re.sub(pattern, '', text)

    return text



remove_special_characters("Well this was fun! What do you think? 123#@!", 

                          remove_digits=True)
# Text lemmatization



def lemmatize_text(text):

    text = nlp(text)

    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    return text



lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")
# Text stemming



def simple_stemmer(text):

    ps = nltk.porter.PorterStemmer()

    text = ' '.join([ps.stem(word) for word in text.split()])

    return text



simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily")
# Remove stopwords



def remove_stopwords(text, is_lower_case=False):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    if is_lower_case:

        filtered_tokens = [token for token in tokens if token not in stopword_list]

    else:

        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text



remove_stopwords("The, and, if are stopwords, computer is not")
# Text normalizer



def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,

                     accented_char_removal=True, text_lower_case=True, 

                     text_lemmatization=True, special_char_removal=True, 

                     stopword_removal=True, remove_digits=True):

    

    normalized_corpus = []

    # normalize each document in the corpus

    for doc in corpus:

        # strip HTML

        if html_stripping:

            doc = strip_html_tags(doc)

        # remove accented characters

        if accented_char_removal:

            doc = remove_accented_chars(doc)

        # lowercase the text    

        if text_lower_case:

            doc = doc.lower()

        # remove extra newlines

        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

        # lemmatize text

        if text_lemmatization:

            doc = lemmatize_text(doc)

        # remove special characters and\or digits    

        if special_char_removal:

            # insert spaces between special characters to isolate them    

            special_char_pattern = re.compile(r'([{.(-)!}])')

            doc = special_char_pattern.sub(" \\1 ", doc)

            doc = remove_special_characters(doc, remove_digits=remove_digits)  

        # remove extra whitespace

        doc = re.sub(' +', ' ', doc)

        # remove stopwords

        if stopword_removal:

            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

            

        normalized_corpus.append(doc)

        

    return normalized_corpus
# Sampling (takes too long otherwise)



ds_merged_left_150.shape

ds_merged_left_150.head()



ds_merged_left_150_sample = ds_merged_left_150.sample(frac=0.02, replace=False, random_state=5)

ds_merged_left_150_sample = ds_merged_left_150_sample.reset_index()

ds_merged_left_150_sample.shape
# Add column with cleaned text to the dataframe

ds_merged_left_150_sample['clean_text'] = normalize_corpus(ds_merged_left_150_sample['review_x'])

ds_merged_left_150_sample.head()
from textblob import TextBlob

    

def detect_polarity(text):

    return TextBlob(text).sentiment.polarity



ds_merged_left_150_sample['polarity'] = ds_merged_left_150_sample['clean_text'].apply(detect_polarity)

ds_merged_left_150_sample.head()

    

# Adding sentiment scores for raw review text



ds_merged_left_150_sample['polarity_raw'] = ds_merged_left_150_sample['review_x'].apply(detect_polarity)

ds_merged_left_150_sample.head()

     
# Dropping polsub columns (not necessary anymore since we split Polarity and Subjectivity)



#ds_merged_left_150_sample = ds_merged_left_150_sample.drop(['pol_sub','pol_sub2'], axis=1)
# correlation btw polarity scores and rating scores

# Spearman correlation between computed polarity and given rating



from scipy.stats import spearmanr

spearmanr(ds_merged_left_150_sample['polarity'], ds_merged_left_150_sample['rating_x'])



# Testing with raw data

spearmanr(ds_merged_left_150_sample['polarity_raw'], ds_merged_left_150_sample['rating_x'])
np.corrcoef(ds_merged_left_150_sample["rating_x"], ds_merged_left_150_sample["polarity"])



# Testing with raw data

np.corrcoef(ds_merged_left_150_sample["rating_x"], ds_merged_left_150_sample["polarity_raw"])
plt.figure(figsize=(6,6))

sns.scatterplot(x='rating_x', y='polarity', data=ds_merged_left_150_sample)

plt.box(False)

plt.xlabel("Rating", fontsize = 12)

plt.ylabel("Polarity", fontsize = 12)

plt.title("Correlation Rating and Polarity", fontsize = 18)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

sns.boxplot(x=ds_merged_left_150_sample["rating_x"],y=ds_merged_left_150_sample["polarity"])

plt.xlabel("Rating")

plt.ylabel("Polarity")

plt.title("Polarity vs Ratings")

plt.show()
# Distribution of Polarity scores across sample



plt.figure(figsize=(6,6))

sns.set_style('white')

sns.distplot(ds_merged_left_150_sample['polarity'])

plt.box(False)

plt.title("Polarity Distribution", fontsize=18)
# Distribution of Rating scores across sample

plt.figure(figsize=(6,6))

sns.distplot(ds_merged_left_150_sample['rating_x'])

plt.box(False)

plt.xlabel('Rating')

plt.title("Rating Distribution", fontsize=16)
# Adding pos/neg/neutral labels



def f(row):

    if row['polarity'] >= 0.3:

        val = 'positive'

    elif row['polarity'] <=-0.3:

        val = 'negative'

    else:

        val = 'neutral'

    return val



ds_merged_left_150_sample['Sentiment'] = ds_merged_left_150_sample.apply(f, axis=1)

ds_merged_left_150_sample[ds_merged_left_150_sample.polarity == -1].head()

ds_merged_left_150_sample[ds_merged_left_150_sample.polarity == 1].head()

#ds_merged_left_150_sample.tail(20)
sns.countplot(ds_merged_left_150_sample['Sentiment'])#.sort_values(by)
ds_merged_left_150_sample = ds_merged_left_150_sample.drop(['index'], axis=1)

#ds_merged_left_150_sample.head()
ds_merged_left_150_sample_displ = ds_merged_left_150_sample.drop(['usefulCount','Mean Rating','Number of Reviews per condition','polarity_raw'], axis=1) 

type(ds_merged_left_150_sample_displ)

ds_merged_left_150_sample_displ.loc[[336]]
ds_merged_left_150_sample_displ[ds_merged_left_150_sample_displ.polarity == -1].head()

ds_merged_left_150_sample_displ[ds_merged_left_150_sample_displ.polarity == 1].head()