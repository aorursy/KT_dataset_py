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
# Libraries



import numpy as np

import os

import glob

import pandas as pd

import datetime

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud ,STOPWORDS

from nltk.util import ngrams

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag, wordpunct_tokenize

from gensim.models import word2vec





from sklearn import feature_selection

from sklearn import model_selection

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, SGDClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

from sklearn.model_selection import train_test_split, KFold

from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.pipeline import Pipeline

from sklearn.manifold import TSNE



import gensim

from xgboost import XGBClassifier

import xgboost as xgb

import lightgbm as lgb

stop = set(stopwords.words('english'))

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import json

import ast

import eli5

import shap

from catboost import CatBoostRegressor

from urllib.request import urlopen

from PIL import Image

import time



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('always')



import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix
dept_df = pd.read_csv("../input/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/data/document_departments.csv")

train = dept_df

train['Document_ID'] = dept_df['Document ID']

train.drop('Document ID',axis =1, inplace=True)

train.head()
company_analysis = sorted(glob.glob('../input/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/data/docs/*.json'))

print('num of train sentiment files: {}'.format(len(company_analysis)))

print(company_analysis[0])
company_analysis = sorted(glob.glob('../input/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/data/docs/*.json'))

# company_analysis = sorted(glob.glob('../input/tour_and_travel_datasets/tour_and_travel_dataset/data/docs/*.json'))



salary_information=[]

jd_id=[]

docid=[]

jd_information=[]

industry = []

department = []

job_location = []

job_keywords = []

job_industry = []

company_name = []

job_experience = []

job_title = []

telephone = []

email = []

company_description = []



for filename in company_analysis:

        with open(filename, 'r') as f:

            sentiment_file = json.load(f)

            jdid =  np.asarray(sentiment_file['api_data']['jd_id'])

            salaryinformation =np.asarray(sentiment_file['topbox_information']['salary_information'])

            jdinformation = np.asarray(sentiment_file['jd_information']['description'])

            industryy = np.asarray(sentiment_file['other_details']['Industry:'])

            job_ind = np.asarray(sentiment_file['api_data']['job_industry'])

            comp_name = np.asarray(sentiment_file['api_data']['company_name'])

            job_exp = np.asarray(sentiment_file['api_data']['job_experience'])

            jobtitle = np.asarray(sentiment_file['api_data']['job_title'])

            dept = np.asarray(sentiment_file['other_details']['Department:'])

            job_loc = np.asmatrix(sentiment_file['api_data']['job_location'])

            job_keyword = np.asarray(sentiment_file['api_data']['job_keywords'])

            try:

                tel = np.asarray(sentiment_file['company_info']['Telephone']) 

                mail = np.asarray(sentiment_file['company_info']['Email']) 

                company_des = np.asarray(sentiment_file['company_info']['Company Description']) 

            except KeyError: pass            

            

        jd_id.append(jdid)

        salary_information.append(salaryinformation)

        jd_information.append(jdinformation)

        industry.append(industryy)

        job_industry.append(job_ind)

        company_name.append(comp_name)

        job_experience.append(job_exp)

        job_title.append(jobtitle)

        department.append(dept)

        job_location.append(job_loc)

        job_keywords.append(job_keyword)

        telephone.append(tel)

        email.append(mail)

        company_description.append(company_des)

        

        docid.append(filename.replace('.json','').replace('../input/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/mohitatbb-machine-learning-assessment-2ef59f0cd1ce/data/docs/', ''))

#         docid.append(filename.replace('.json','').replace('../input/tour_and_travel_datasets/tour_and_travel_dataset/data/docs/', ''))



        

company_analysis = pd.concat([ pd.DataFrame(docid, columns =['Document_ID']) ,pd.DataFrame(jd_id, columns =['JdID']),

                              pd.DataFrame(company_description, columns =['Company_Description']),

                              pd.DataFrame(jd_information, columns =['JobInformation']),

                              pd.DataFrame(industry, columns =['Industry']),

                              pd.DataFrame(job_industry, columns =['Job_Industry']),

                              pd.DataFrame(company_name, columns =['Company_Name']),

                              pd.DataFrame(job_experience, columns =['Experience']),

                              pd.DataFrame(job_title, columns =['Job_Title']),

                              pd.DataFrame(department, columns =['Job_Department','Job_Department1']),

#                               pd.DataFrame(job_location, columns =['Job_Location']),

                              pd.DataFrame(job_keywords, columns =['Job_Keywords','Job_Keywords2','Job_Keywords3','Job_Keywords4','Job_Keywords5','Job_Keywords6','Job_Keywords7','Job_Keywords8','Job_Keywords9','Job_Keywords10','Job_Keywords11','Job_Keywords12','Job_Keywords13','Job_Keywords14','Job_Keywords15','Job_Keywords16','Job_Keywords17','Job_Keywords18','Job_Keywords19','Job_Keywords20','Job_Keywords21','Job_Keywords22','Job_Keywords23','Job_Keywords24','Job_Keywords25','Job_Keywords26','Job_Keywords27','Job_Keywords28','Job_Keywords29','Job_Keywords30','Job_Keywords31','Job_Keywords33','Job_Keywords32','Job_Keywords34']),

                              pd.DataFrame(telephone, columns =['Telephone']),

                              pd.DataFrame(email, columns =['Email']),

                              pd.DataFrame(salary_information, columns =['SalaryInformation'])],axis =1)

                                                

company_analysis.sample(3)
company_analysis.loc[pd.isnull(company_analysis['Job_Department1']), 'Job_Department1'] = ''

company_analysis['Job_Department'] = company_analysis['Job_Department']+','+company_analysis['Job_Department1']

company_analysis.drop('Job_Department1', axis= 1, inplace= True)

company_analysis['Job_Department'] = company_analysis['Job_Department'].str.rstrip(',')



columns = ['Job_Keywords2','Job_Keywords3','Job_Keywords4','Job_Keywords5','Job_Keywords6','Job_Keywords7','Job_Keywords8','Job_Keywords9','Job_Keywords10','Job_Keywords11','Job_Keywords12','Job_Keywords13','Job_Keywords14','Job_Keywords15','Job_Keywords16','Job_Keywords17','Job_Keywords18','Job_Keywords19','Job_Keywords20','Job_Keywords21','Job_Keywords22','Job_Keywords23','Job_Keywords24','Job_Keywords25','Job_Keywords26','Job_Keywords27','Job_Keywords28','Job_Keywords29','Job_Keywords30','Job_Keywords31','Job_Keywords32','Job_Keywords33','Job_Keywords34']

for col in columns:

    company_analysis.loc[pd.isnull(company_analysis[col]),col] = ''

    company_analysis['Job_Keywords'] = company_analysis['Job_Keywords']+','+company_analysis[col]

    company_analysis.drop(col ,axis = 1, inplace= True)

    

company_analysis['Job_Keywords'] = company_analysis['Job_Keywords'].str.strip(',')

company_analysis.sample(3)
company_analysis.info()
company_analysis.describe(include='all')
company_analysis['Document_ID'] = company_analysis['Document_ID'].astype(np.int64)

df = company_analysis.merge(train, on="Document_ID", how = 'inner')

df.head()
## Memory reducer



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df



df = reduce_mem_usage(df)
df.fillna(0, inplace=True)

columns = ['Company_Description','JobInformation','Company_Name','Job_Title','Job_Department','Job_Keywords','Department']

for col in columns:

    df.loc[df[col]=='', col] = 'None'



df.fillna(-99999, inplace=True)

df = df.reset_index()



df[df==np.inf]=np.nan

df.fillna(df.mean(), inplace=True)
## col having same values

for col in df.columns:

    if df[col].nunique()==1:

        print(col)
df.Industry.value_counts()

df.drop(['Industry','Job_Industry'],axis=1,inplace=True)
df.Email.value_counts().head(10)
df['has_email'] = 0

df.loc[df['Email'].isnull() == False, 'has_email'] = 1

df.has_email.value_counts()
df.drop(['Email','has_email'],axis=1, inplace=True)
df.Telephone.value_counts().head(10)
df['has_telephone'] = 0

df.loc[df['Telephone'].isnull() == False, 'has_telephone'] = 1

df.has_telephone.value_counts()
df.drop(['Telephone','has_telephone'],axis=1, inplace=True)
df.drop('Document_ID',axis=1, inplace=True)
df.SalaryInformation.value_counts().head(10)
df['has_SalaryInformation'] = 1

df.loc[df['SalaryInformation'] == '', 'has_SalaryInformation'] = 0

df.has_SalaryInformation.value_counts()
df.drop('SalaryInformation',axis=1, inplace=True)
df.Experience.value_counts().head()
experience_split = df['Experience'].str[0:-1].str.split('to', expand=True)

experience_split.head()
#remove space in left and right 

experience_split[1] =  experience_split[1].str.strip()

#remove comma 

experience_split[1] = experience_split[1].str.replace('Yr', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

experience_split[1] = experience_split[1].str.replace(r'Yr', '')

#display 

experience_split[1].head()
experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce')

experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce')



experience=pd.concat([experience_split[0], experience_split[1]], axis=1, sort=False)



experience.rename(columns={0:'min_experience', 1:'max_experience'}, inplace=True)

experience.head()
df =pd.concat([df , experience], axis=1, sort=False)

df.drop('Experience',axis=1,inplace=True)

df.sample(3)



df.loc[df['min_experience'].isna() == True, 'min_experience'] = 0

df.loc[df['max_experience'].isna() == True, 'max_experience'] = 0
df['avg_experience']=(df['min_experience'].values + df['max_experience'].values)/2
df['Job_Title'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

df['Job_Title'].value_counts().head(9).plot(kind = 'bar')
plt.figure(figsize = (12, 12))

text = ' '.join(df['Job_Title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in job titles')

plt.axis("off")

plt.show()
df['Job_Department'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

df['Job_Department'].value_counts().head(5).plot(kind = 'bar')
plt.figure(figsize = (12, 12))

text = ' '.join(df['Job_Department'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in job titles')

plt.axis("off")

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(df['Job_Keywords'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in job titles')

plt.axis("off")

plt.show()
df.JobInformation.value_counts()
df['has_JobInformation'] = 1

df.loc[df['JobInformation'] == "None", 'has_JobInformation'] = 0

df.has_JobInformation.value_counts()
plt.figure(figsize = (12, 12))

text = ' '.join(df['JobInformation'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white',stopwords=set(STOPWORDS), width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in job titles')

plt.axis("off")

plt.show()
jobinfo =' '.join(text for text in df['JobInformation'])



lem=WordNetLemmatizer()

text=[lem.lemmatize(w) for w in word_tokenize(jobinfo)]

vect=TfidfVectorizer(ngram_range=(1,3),max_features=100)

vectorized_data=vect.fit_transform(text)

vect.vocabulary_.keys()
def build_corpus(df,col):

    

    '''function to build corpus from dataframe'''

    lem=WordNetLemmatizer()

    corpus= []

    for x in df[col]:

        

        

        words=word_tokenize(x)

        corpus.append([lem.lemmatize(w) for w in words])

    return corpus
corpus=build_corpus(df,'JobInformation')

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=30, workers=4)
def tsne_plot(model,title='None'):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=80, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(12, 12)) 

    plt.title(title)

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model,'JobInformation')
token=word_tokenize(jobinfo)

counter=Counter(token)

count=[x[0] for x in counter.most_common(40) if len(x[0])>3]

print("Most common words in Requirement")

print(count)
lem=WordNetLemmatizer()

text=[lem.lemmatize(w) for w in word_tokenize(jobinfo)]

vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)

vectorized_data=vect.fit_transform(text)

id2word=dict((v,k) for k,v in vect.vocabulary_.items())
df.to_csv('data_cleaned.csv',index=False)