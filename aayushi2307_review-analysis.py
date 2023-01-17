# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings; 

warnings.simplefilter('ignore')



from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, KFold



import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
df_review = pd.read_csv("../input/employee-reviews/employee_reviews.csv", sep=',', error_bad_lines=False)
df_review.replace(to_replace = 'none', value = np.nan, inplace = True)
df_review.rename(columns = {'dates':'date'}, inplace = True)
df_copy = df_review.copy()
df = df_copy.dropna()
df.head(1)
df.info()
df['date'] = df['date'].astype(dtype=np.datetime64, inplace=True)
df['overallratings'] = df['overall-ratings'].astype(dtype=np.float64)

df['work-balance-stars'] = df['work-balance-stars'].astype(dtype=np.float64)

df['culture-values-stars'] = df['culture-values-stars'].astype(dtype=np.float64)

df['carrer-opportunities-stars'] = df['carrer-opportunities-stars'].astype(dtype=np.float64)

df['comp-benefit-stars'] = df['comp-benefit-stars'].astype(dtype=np.float64)

df['senior-mangemnet-stars'] = df['senior-mangemnet-stars'].astype(dtype=np.float64)
df['is_current_employee'] = df['job-title'].apply(lambda x: 1 if 'Current' in x else 0)

df['is_high_Overall'] = df['overall-ratings'].apply(lambda x: 1 if x>3 else 0)

df['is_high_worbalance']= df['work-balance-stars'].apply(lambda x: 1 if x >3 else 0)

df['is_high_culturevalue']= df['culture-values-stars'].apply(lambda x: 1 if x >3 else 0)

df['is_high_careeropp']= df['carrer-opportunities-stars'].apply(lambda x: 1 if x >3 else 0)

df['is_high_compbenefit']= df['comp-benefit-stars'].apply(lambda x: 1 if x >3 else 0)

df['is_high_srmngmt']= df['senior-mangemnet-stars'].apply(lambda x: 1 if x >3 else 0)
sns.factorplot(x = 'overall-ratings', y = 'company',hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
sns.factorplot(x = 'work-balance-stars', y = 'company',hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
sns.factorplot(x = 'culture-values-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
sns.factorplot(x = 'carrer-opportunities-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
sns.factorplot(x = 'comp-benefit-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
sns.factorplot(x = 'senior-mangemnet-stars', y = 'company', hue= 'is_current_employee', data = df, kind ='box', \

               aspect =2)
import re

# Natural Language Tool Kit 

import nltk  

nltk.download('stopwords') 

# nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

nltk.download('punkt')

# to remove stopword 

from nltk.corpus import stopwords 



# for Stemming propose  

from nltk.stem.porter import PorterStemmer 

from nltk.stem.snowball import SnowballStemmer

from nltk import pos_tag

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet
df_review["review"] = df_review["pros"] + ' ' + df_review["cons"] + ' ' + df_review["advice-to-mgmt"]
df_review.dropna(how='any',subset=['review'],inplace = True)
sw = stopwords.words('english')

def clean(text):

    

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [t for t in text if len(t) > 0]

    text = [t for t in text if t not in sw]

    text = ' '.join(text)

    return text
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
# ps = PorterStemmer()

sw = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def lemmatize(text):

    text = nltk.word_tokenize(text)

    pos_tags = pos_tag(text)

    #     text = [ps.stem(word) for word in text if not word in set(sw)]

    text = [lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    text = ' '.join(text)

    return text
clean(df_review.iloc[0].review)
lemmatize(df_review.iloc[0].review)
df_review['review_clean'] = df_review['review'].apply(lambda x: clean(x))
df_review['review_lemmatize'] = df_review['review_clean'].apply(lambda x: lemmatize(x))
df_review.info()
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

extras = ["great","team","work","company","place","good","people","employee","none","make","one","go",\

         "day","call","new","come","think","happen","within","look","store","retail","feel",\

         "life","sometime","environment","move","keep","still","review","group","year","role",\

         "want","try","office","create","look","even","level","many","thing","much","even",\

         "hour","year","always","every","things","project","product","need","time","give",\

          "take","never"]

stopwords.update(extras)

companies = list(df_review.company.unique())

for company in companies:

    stopwords.add(company)
def wordclouds(df_review,companies):

    for company in companies:

        temp = df_review.loc[df_review["company"]==company]

        text = " ".join(str(review) for review in temp.review_clean)

        # Create and generate a word cloud image:

        wordcloud = WordCloud(stopwords = stopwords, collocations = False).generate(text)

        # Display the generated image:

        plt.imshow(wordcloud, interpolation='bilinear')

        plt.axis("off")

        plt.title(company.upper())

        plt.show()
wordclouds(df_review,companies)