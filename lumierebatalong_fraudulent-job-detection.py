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
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy
from wordcloud import WordCloud, STOPWORDS
from warnings import filterwarnings
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 300)
filterwarnings('ignore')
#create class
class jobposting:
    """
    This class help me to do some task that I need. It contain 7 attributes:
    
    - read: read a csv file
    - multi_categorical_plot
    - feature_value_counts
    - catplot_multi
    - cleaning_preparation
    - extract_most_popular_words
    
    
    """
    
    def __init__(self, data=None, cols=None, target='fraudulent'):
        
        self.data = data # feature
        self.cols = cols # feature columns name
        self.targ = target # target 
        
    #Read csv file
    def read(self, file, index_cols = None):
        return pd.read_csv(file, index_col=index_cols)
    
    def multi_categorical_plot(self, data):
    
        """ plot a categorical feature
        
            data: float64 array  n_observationxn_feature
        
        """
        # Find a feature that type is object
        string = []
        for i in data.columns:
            if data[i].dtypes == "object":
                string.append(i)
    
        fig = plt.figure(figsize=(20,20))
        fig.subplots_adjust(wspace=0.4, hspace = 0.1)
        for i in range(1,len(string)+1):
            ax = fig.add_subplot(2,2,i)
            sns.countplot(y=string[i-1], data=data, hue=self.targ, orient = 'h', ax=ax)
            ax.set_title(f" {string[i-1]} countplot")
            
    def feature_value_counts(self, data):
        """
        counts a unique values
        """
        
        print('{}\n'.format(data.value_counts()))
            
    def catplot_multi(self, data):
        """ plot multi catplot"""
    
    
        cols = data.columns
        
        gp = plt.figure(figsize=(20,20))
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(2,2,i)
            sns.catplot(x =cols[i-1],  data=data)
            ax.set_title('{}'.format(cols[i-1]))
            
    
    def cleaning_preparation(self, data):
        """
            This function cleans and prepares data
        
        """
        
        import string
        
        # convert a text to lower case
        data['text'] = data['text'].str.lower()
        
        # Splitting and Removing Punctuation from the Text
        all_data = data['text'].str.split(' ')
        
        #Joining the Entire Text
        all_data_cleaned = []
        
        for text in all_data:
            text = [x.strip(string.punctuation) for x in text]
            all_data_cleaned.append(text)
            
        text_data = [" ".join(text) for text in all_data_cleaned]
        final_text_data = " ".join(text_data)
        
        return final_text_data
    
    def word_cloud(self, data):
        """
        this function plot word cloud
        """
        
        stopwords = set(STOPWORDS) # initialize a stop words
        stopwords.update(['will', 'be', 'you', 'are', 'looking', 'must', 'for', 'look', 'within' ])
        
        # we apply our data to WordCloud
        wordcloud_data = WordCloud(stopwords=stopwords, background_color="white",\
                                   max_font_size=50, max_words=100).generate(data)
        
        plt.figure(figsize = (20,20))
        plt.imshow(wordcloud_data, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
    
    def extract_most_popular_words(self, data):
        """
        this function extract a most popular words using in this data.
        """
        #import package
        import collections
        
        stopwords = set(STOPWORDS)
        stopwords.update(['will', 'be', 'you', 'are', 'looking', 'must', 'for', 'look', 'within' ])
        
        # we filters a words and count it.
        filtered_words_data = [word for word in data.split() if word not in stopwords]
        counted_words_data = collections.Counter(filtered_words_data)

        word_count_data = {}

        #we take 30 first words most used.
        for letter, count in counted_words_data.most_common(30):
            word_count_data[letter] = count
    
        # show a words.
        for i,j in word_count_data.items():
            print('Word: {0}, count: {1}'.format(i,j))
            
    def processed_corpus(self, data):
        
        """
        This function 
        """
        import string
        from spacy.lang.en.stop_words import STOP_WORDS
        from spacy.lang.en import English
        
        # Create our list of punctuation marks
        punctuations = string.punctuation
        
        # create our list of stopwords
        stop_words = STOP_WORDS
        
        # Load English tokenizer, tagger, parser, NER and word vectors
        parser = English()
        
        corpus = [] #coprus
        for text in data['text']:
            
            mytokens = parser(text)
       
        
            # Lemmatizing each token and converting each token into lowercase
            mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_\
                        for word in mytokens ]

        
            # Removing stop words
            mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
            
            corpus.append(mytokens)
        
            
        return corpus  # return preprocessed text data
    
    
    def getSimilarities(self, corpus, query_text):
        
        #import package
        from gensim import corpora
        from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
        from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

        #dictionary for corpus
        dictionary = corpora.Dictionary(corpus)

       
        #We create the bag of words representation for our corpus
        BoW_corpus = [dictionary.doc2bow(text) for text in corpus]

        # create a model word to vec
        model = Word2Vec(corpus, size=20, min_count=1)
        
        #Computes cosine similarities between word embeddings and retrieves the closest word embeddings
        #by cosine similarity for a given word embedding.
        termsim_index = WordEmbeddingSimilarityIndex(model.wv)

        # construct similarity matrix and Builds a sparse term similarity matrix using
        #a term similarity index.
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary) 

        #Compute soft cosine similarity against a corpus of documents by storing the index matrix in memory.
        docsim_index = SoftCosineSimilarity(BoW_corpus, similarity_matrix, num_best=10)
        
        #we split and do bow of our query real job
        query_bow = dictionary.doc2bow(query_text.split())
        
        #calculate similarity of query to each doc from bow_corpus
        return docsim_index[query_bow]  
file = '/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv'
myjob = jobposting()
job = myjob.read(file, index_cols='job_id')
job.head(3)
job.info()
#check missing value
a = job.isnull().sum()
a[a>0]
job_data = job.fillna(value=' ')# impute a missing values
job_data.tail()
# we create country feature using location
def split(feature):
    l = feature.split(',')
    return l[0]

job_data['country'] = job_data.location.apply(split)
need_objcols_eda = ['employment_type', 'required_experience','required_education', 'function', 'fraudulent']
need_numcols_eda = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
myjob.multi_categorical_plot(job_data[need_objcols_eda])
country = job_data.groupby(by=['country', 'fraudulent'])['location'].count().reset_index()
country = country.sort_values(by ='location',ascending=False)
plt.figure(figsize=(15,5))
sns.barplot(x='country', y='location', hue='fraudulent', data=country[:10])
plt.title('10th countries most posted job')
plt.show()
plt.figure(figsize=(5,20), dpi=100)
sns.countplot(y='industry', data=job_data, hue='fraudulent')
plt.title('Industry job posting')
plt.show()
fakejobDescription = job_data[job_data.fraudulent==1][['title','description']]
fakejobDescription['text'] = fakejobDescription['title'] + ':  ' + fakejobDescription['description']

del fakejobDescription['description']
del fakejobDescription['title']
fakejobDescription.tail()
text_corpus = myjob.cleaning_preparation(fakejobDescription)
print('Number of characters of text corpus is: {}'.format(len(text_corpus)))
%%time
#dispay 200 first characters
nlp = spacy.load("en_core_web_sm")
doc = nlp(text_corpus[:1000])
displacy.render(doc, style="ent", jupyter=True)
%%time
myjob.word_cloud(text_corpus)
%%time
#we give a 30 words most used.
myjob.extract_most_popular_words(text_corpus)
realjobDescription = job_data[job_data.fraudulent==0]
realjobDescription['text'] = realjobDescription['title'] + ' ' + realjobDescription['location'] + ' ' +\
                            realjobDescription['department'] + ' ' + \
                            realjobDescription['company_profile'] + ' ' + realjobDescription['description'] +\
                            realjobDescription['requirements'] + ' ' + realjobDescription['benefits'] + ' '+\
                            realjobDescription['required_education'] + ' ' +\
                            realjobDescription['required_experience'] + ' ' +\
                            realjobDescription['employment_type'] + ' ' + realjobDescription['industry'] +\
                            realjobDescription['function']
del realjobDescription['title']
del realjobDescription['location']
del realjobDescription['department'] 
del realjobDescription['company_profile']
del realjobDescription['description']
del realjobDescription['requirements']
del realjobDescription['benefits']
del realjobDescription['required_education']
del realjobDescription['required_experience']
del realjobDescription['employment_type']
del realjobDescription['industry']
del realjobDescription['function']
realjobDescription.head()
from gensim.utils import simple_preprocess
#preprocessed our documents
corpus = [simple_preprocess(text) for text in realjobDescription.text]
%time
tag = realjobDescription.text.iloc[0]
sims = myjob.getSimilarities(corpus, tag)
%time
#we display
sims
#we check it below
realjob = realjobDescription.reset_index()
realjob[realjob.index.isin([0,4386, 4044, 3071, 2144, 8263, 1995, 14908, 6599, 12491])][['country', 'text']]
#Comparison index job = 0, index job = 1995, index job = 4044 and 
tag1 = realjobDescription.text.iloc[0]
tag2 = realjobDescription.text.iloc[1995]

#dispay 200 first characters
nlp = spacy.load("en_core_web_sm")

doc1 = nlp(tag1)
doc2 = nlp(tag2)

doc1.user_data['title'] = 'Job description for index 0'
doc2.user_data['title'] = 'Job description for index 1995'

displacy.render([doc1, doc2], style="ent", jupyter=True)
#Comparison index job = 0, index job = 1995, index job = 4044 and 
#we take a first real job where job_id = 1.
%time
tag1 = realjobDescription.text.iloc[0]
tag2 = realjobDescription.text.iloc[14908]

#dispay 200 first characters
nlp = spacy.load("en_core_web_sm")

doc1 = nlp(tag1)
doc2 = nlp(tag2)

doc1.user_data['title'] = 'Job description for index 0'
doc2.user_data['title'] = 'Job description for index 14908'

displacy.render([doc1, doc2], style="ent", jupyter=True)
job_data['text'] =job_data['title'] + ' ' + job_data['location'] + ' ' +\
                            job_data['department'] + ' ' + \
                            job_data['company_profile'] + ' ' + job_data['description'] +\
                            job_data['requirements'] + ' ' + job_data['benefits'] + ' '+\
                            job_data['required_education'] + ' ' +\
                            job_data['required_experience'] + ' ' +\
                            job_data['employment_type'] + ' ' + job_data['industry'] +\
                            job_data['function']
del job_data['title']
del job_data['location']
del job_data['department'] 
del job_data['company_profile']
del job_data['description']
del job_data['requirements']
del job_data['benefits']
del job_data['required_education']
del job_data['required_experience']
del job_data['employment_type']
del job_data['industry']
del job_data['function']
job_data.tail(3)

