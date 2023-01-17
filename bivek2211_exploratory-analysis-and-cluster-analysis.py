# Libraries



import numpy as np

import os

import glob

import pandas as pd

import datetime

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag, wordpunct_tokenize



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