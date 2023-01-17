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
# import tensorflow as tf

# p = tf.config.experimental.list_physical_devices('GPU')

# tf.config.experimental.set_visible_devices(p[0], 'GPU')
train = pd.read_csv("/kaggle/input/data-scientist-salary/train.csv")

test = pd.read_csv("/kaggle/input/data-scientist-salary/test.csv")
train.head()
train.shape, test.shape
train.nunique()
train.info()
train.isna().sum()
train = train.dropna(subset = ["key_skills"])

df_train = train[['key_skills', 'job_desig', 'job_description', 'location', 'job_type', 'experience','salary']]

df_test = test[['key_skills', 'job_desig', 'job_description', 'job_type', 'experience', 'location']]
df_train.head()
import re



def clean_skills(skl):

    skills = str(skl).lower()

    skills = re.sub('\...','',skills)

    skills = re.sub(',','',skills)

    skills = re.sub(r'\s+', ' ', skills)

    return skills



df_train['skills_cleaned'] = df_train['key_skills'].apply(clean_skills)

df_test['skills_cleaned'] = df_test['key_skills'].apply(clean_skills)
df_train.head()
train.job_description.fillna('missing',inplace = True)

test['job_description'].fillna('missing', inplace=True)



def clean_job_desc(job):

    job_desc = str(job).lower()

    job_desc = re.sub(r'[^a-z]', ' ', job_desc)

    job_desc = re.sub(r'\s+', ' ', job_desc)

    return job_desc



df_train['job_desc_cleaned'] = df_train['job_description'].apply(clean_job_desc)

df_test['job_desc_cleaned'] = df_test['job_description'].apply(clean_job_desc)
df_train.head()


def clean_location(loc):

    location = loc.lower()

    location = re.sub(r'[^a-z]', ' ', location)

    location = re.sub(r'\s+', ' ', location)

    return location



df_train['loc_cleaned'] = df_train['location'].apply(clean_location)

df_test['loc_cleaned'] = df_test['location'].apply(clean_location)
train['job_type'].fillna('missingjobtype', inplace=True)

train['job_type'].replace('Analytics', 'analytics', inplace=True)

train['job_type'].replace('Analytic', 'analytics', inplace=True)

train['job_type'].replace('ANALYTICS', 'analytics', inplace=True)

train['job_type'].replace('analytic', 'analytics', inplace=True)



test['job_type'].fillna('missingjobtype', inplace=True)

test['job_type'].replace('Analytics', 'analytics', inplace=True)

test['job_type'].replace('Analytic', 'analytics', inplace=True)

test['job_type'].replace('ANALYTICS', 'analytics', inplace=True)

test['job_type'].replace('analytic', 'analytics', inplace=True)



df_train['job_type_cleaned'] = train['job_type'] 

df_test['job_type_cleaned'] = test['job_type']
df_train.head()
df_train.isna().sum()
df_train.head()
def min_exp(val):

    exp = re.sub('-',' ',val)

    exp = exp.split(" ")

    exp = int(exp[0])

    return exp

    

def max_exp(val):

    exp = re.sub('-',' ',val)

    exp = exp.split(' ')

    exp = int(exp[1])

    return exp

    

df_train['min_exp'] = df_train['experience'].apply(lambda x : min_exp(x))

df_train['max_exp'] = df_train['experience'].apply(lambda x : max_exp(x))



df_test['min_exp'] = df_test['experience'].apply(lambda x : min_exp(x))

df_test['max_exp'] = df_test['experience'].apply(lambda x : max_exp(x))

        
df_train.head()
def clean_job_desig(desig):

    job_desig = desig.lower()

    job_desig = re.sub(r'[^a-z]', ' ', job_desig)

    job_desig = re.sub(r'\s+', ' ', job_desig)

    return job_desig



df_train['desig_cleaned'] = df_train['job_desig'].apply(clean_job_desig)

df_test['desig_cleaned'] = df_test['job_desig'].apply(clean_job_desig)
df_train['merged'] = (df_train['desig_cleaned'] + ' ' + df_train['job_desc_cleaned'] + ' ' + df_train['skills_cleaned']

                      + ' ' + df_train['job_type_cleaned'])



df_test['merged'] = (df_test['desig_cleaned'] + ' ' + df_test['job_desc_cleaned'] + ' ' + df_test['skills_cleaned']

                     + ' ' + df_test['job_type_cleaned'])
df_train.head()
data_train  = df_train[['merged', 'loc_cleaned', 'min_exp', 'max_exp']] 

data_test = df_test[['merged', 'loc_cleaned', 'min_exp', 'max_exp']] 
data_train.head()
data_test.head()
data_train = data_train.rename(columns = {'merged':'emp_info'},inplace = False)
data_test = data_test.rename(columns = {'merged':'emp_info'},inplace = False)
def min_sal(sal):

    val = str(sal).split("to")

    return val[0]

def max_sal(sal):

    val = str(sal).split("to")

    return val[1]



target = pd.DataFrame()

target["min_sal"] = df_train["salary"].apply(lambda x: min_sal(x))

target["max_sal"] = df_train["salary"].apply(lambda x: max_sal(x))

target1 = target.min_sal

target2 = target.max_sal
target.head()
import matplotlib.pyplot as plt

import seaborn as sns



def get_ax(rows = 1,cols = 2,size = 7):

    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return fig,ax
fig,ax = get_ax()

sns.distplot(data_train["emp_info"].str.len(),ax = ax[0])

sns.distplot(data_test["emp_info"].str.len(),ax = ax[1])
data_train.nunique()
fig,ax = get_ax()



sns.distplot(data_train.min_exp,ax = ax[0])

sns.distplot(data_train.max_exp,ax = ax[0])





sns.distplot(data_test.min_exp,ax = ax[1])

sns.distplot(data_test.max_exp,ax = ax[1])
sns.distplot(data_train.max_exp-data_train.min_exp)


from wordcloud import WordCloud

def wordcloud(data):

    wordcloud = WordCloud(background_color = 'Black',

                         max_words = 50,

                         max_font_size = 40,

                         scale = 5,

                         random_state = 5).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()

wordcloud(data_train["emp_info"]) 
from wordcloud import WordCloud

def wordcloud(data):

    wordcloud = WordCloud(background_color = 'Black',

                         max_words = 50,

                         max_font_size = 40,

                         scale = 5,

                         random_state = 5).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()

wordcloud(data_test["emp_info"]) 
data_train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['salary'] = le.fit_transform(train['salary'])
from sklearn.model_selection import train_test_split



X_train, X_cv, y_train, y_cv = train_test_split(

    data_train,train['salary'], test_size=0.20, 

    stratify=train['salary'], random_state=75)
print('No. of sample texts X_train: ', len(X_train))

print('No. of sample texts X_cv   : ', len(X_cv))

X_train_merged = X_train['emp_info']

X_train_loc = X_train['loc_cleaned']



X_cv_merged = X_cv['emp_info']

X_cv_loc = X_cv['loc_cleaned']
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tf1 = TfidfVectorizer(min_df=3, token_pattern=r'\w{3,}', ngram_range=(1,3), max_df=0.9)

tf2 = TfidfVectorizer(min_df=2, token_pattern=r'\w{3,}')



X_train_merged = tf1.fit_transform(X_train_merged)

X_train_loc = tf2.fit_transform(X_train_loc)



X_cv_merged = tf1.transform(X_cv_merged)

X_cv_loc = tf2.transform(X_cv_loc)

# X_cv_merged
from scipy import sparse

from sklearn.preprocessing import StandardScaler



sc1 = StandardScaler()

X_train_MinExp = sc1.fit_transform(np.array(X_train['min_exp']).reshape(-1,1))

X_cv_MinExp = sc1.transform(np.array(X_cv['min_exp']).reshape(-1,1))

X_train_MinExp = sparse.csr_matrix(X_train_MinExp)

X_cv_MinExp = sparse.csr_matrix(X_cv_MinExp)



sc2 = StandardScaler()

X_train_MaxExp = sc2.fit_transform(np.array(X_train['max_exp']).reshape(-1,1))

X_cv_MaxExp = sc2.transform(np.array(X_cv['max_exp']).reshape(-1,1))

X_train_MaxExp = sparse.csr_matrix(X_train_MaxExp)

X_cv_MaxExp = sparse.csr_matrix(X_cv_MaxExp)
from scipy.sparse import hstack, csr_matrix



merged_train = hstack((X_train_merged, X_train_loc, X_train_MinExp, X_train_MaxExp))

merged_cv  = hstack((X_cv_merged, X_cv_loc, X_cv_MinExp, X_cv_MaxExp))
merged_train.shape, merged_cv.shape

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb

train_data = lgb.Dataset(merged_train, label=y_train)

test_data = lgb.Dataset(merged_cv, label=y_cv)
param = {'objective': 'multiclass',

         'num_iterations': 80,

         'learning_rate': 0.04,  

         'num_leaves': 23,

         'max_depth': 7, 

         'min_data_in_leaf': 28, 

         'max_bin': 10, 

         'min_data_in_bin': 3,   

         'num_class': 6,

         'metric': 'multi_logloss'

         }
lgbm = lgb.train(params=param,

                 train_set=train_data,

                 num_boost_round=100,

                 valid_sets=[test_data])



y_pred_class = lgbm.predict(merged_cv)
X_train_merged = data_train['emp_info']

X_train_loc = data_train['loc_cleaned']



X_test_merged = data_test['emp_info']

X_test_loc = data_test['loc_cleaned']



y_train = train['salary']
tf1 = TfidfVectorizer(min_df=3, token_pattern=r'\w{3,}', ngram_range=(1,3))

tf2 = TfidfVectorizer(min_df=2, token_pattern=r'\w{3,}')



X_train_merged = tf1.fit_transform(X_train_merged)

X_train_loc = tf2.fit_transform(X_train_loc)



X_test_merged = tf1.transform(X_test_merged)

X_test_loc = tf2.transform(X_test_loc)
from scipy import sparse

from sklearn.preprocessing import StandardScaler



sc1 = StandardScaler()

X_train_MinExp = sc1.fit_transform(np.array(df_train['min_exp']).reshape(-1,1))

X_test_MinExp = sc1.transform(np.array(df_test['min_exp']).reshape(-1,1))

X_train_MinExp = sparse.csr_matrix(X_train_MinExp)

X_test_MinExp = sparse.csr_matrix(X_test_MinExp)



sc2 = StandardScaler()

X_train_MaxExp = sc2.fit_transform(np.array(df_train['max_exp']).reshape(-1,1))

X_test_MaxExp = sc2.transform(np.array(df_test['max_exp']).reshape(-1,1))

X_train_MaxExp = sparse.csr_matrix(X_train_MaxExp)

X_test_MaxExp = sparse.csr_matrix(X_test_MaxExp)
merged_train = hstack((X_train_merged, X_train_loc, X_train_MinExp, X_train_MaxExp))

merged_test  = hstack((X_test_merged, X_test_loc, X_test_MinExp, X_test_MaxExp))
import lightgbm as lgb

train_data = lgb.Dataset(merged_train, label=y_train)



param = {'objective': 'multiclass',

         'num_iterations': 80,

         'learning_rate': 0.04, 

         'num_leaves': 23,

         'max_depth': 7, 

         'min_data_in_leaf': 28, 

         'max_bin': 10, 

         'min_data_in_bin': 3,   

         'num_class': 6,

         'metric': 'multi_logloss'

         }



lgbm = lgb.train(params=param, 

                 train_set=train_data)



predictions = lgbm.predict(merged_test)



y_pred_class = []

for x in predictions:

    y_pred_class.append(np.argmax(x))



y_pred_class = le.inverse_transform(y_pred_class)
df_sub = pd.DataFrame(data=y_pred_class, columns=['salary'])

df_sub
df_sub.to_csv("sub.csv",index = False)
def min_sal(sal):

    val = str(sal).split("to")

    return val[0]

def max_sal(sal):

    val = str(sal).split("to")

    return val[1]



minsal = df_sub["salary"].apply(lambda x: min_sal(x))

max_sal = df_sub["salary"].apply(lambda x: max_sal(x))

X = pd.DataFrame({"min_sal":minsal,

                  "max_sal":max_sal})
fig,ax = plt.subplots(1,1 ,figsize = (14,8))

sns.distplot(minsal)

sns.distplot(max_sal)
test1 = pd.read_csv("/kaggle/input/data-scientist-salary/test.csv")
final = pd.concat([test1,X],axis=1)
final.head()
def min_exp(val):

    exp = re.sub('-',' ',val)

    exp = exp.split(" ")

    exp = int(exp[0])

    return exp

    

def max_exp(val):

    exp = re.sub('-',' ',val)

    exp = exp.split(' ')

    exp = int(exp[1])

    return exp

    

final['min_exp'] = final['experience'].apply(lambda x : min_exp(x))

final['max_exp'] = final['experience'].apply(lambda x : max_exp(x))



# df_test['min_exp'] = df_test['experience'].apply(lambda x : min_exp(x))

# df_test['max_exp'] = df_test['experience'].apply(lambda x : max_exp(x))

        
final.head()
labels = ["min_exp","max_exp","min_sal","max_sal"]

sns.pairplot(final[labels])
# def min_exp(val):

#     exp = re.sub('-',' ',val)

#     exp = exp.split(" ")

#     exp = int(exp[0])

#     return exp

    

# def max_exp(val):

#     exp = re.sub('-',' ',val)

#     exp = exp.split(' ')

#     exp = int(exp[1])

#     return exp

col = final.loc[: , "min_sal":"max_sal"]

final['salary_mean'] = col.mean(axis=1)



cols = final.loc[: , "min_exp":"max_exp"]

final['exp_mean'] = cols.mean(axis=1)

final.head()
sns.scatterplot(x = final["exp_mean"],y = final["salary_mean"])