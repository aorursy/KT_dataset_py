# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


import re
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip /kaggle/input/donorschoose-application-screening/train.zip
!unzip /kaggle/input/donorschoose-application-screening/resources.zip
!unzip /kaggle/input/donorschoose-application-screening/test.zip
!unzip /kaggle/input/donorschoose-application-screening/sample_submission.zip
train_data = pd.read_csv('/kaggle/working/train.csv')
resource_data = pd.read_csv('/kaggle/working/resources.csv')
test_data = pd.read_csv('/kaggle/working/test.csv')
train_data.head(2)
test_data.head(2)
print(train_data.shape)
print(test_data.shape)
train_data['source']='train'
test_data['source']='test'
project_data = pd.concat([train_data,test_data],axis=0)
project_data.tail(2)
print("Number of data points in train data", project_data.shape)
print('-'*50)
print("The attributes of data :", project_data.columns.values)
project_data['project_grade_category'].value_counts()
project_data['project_grade_category'] = project_data['project_grade_category'].str.replace(' ','_')
project_data['project_grade_category'] = project_data['project_grade_category'].str.replace('-','_')
project_data['project_grade_category'] = project_data['project_grade_category'].str.lower()
project_data['project_grade_category'].value_counts()
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' The ','')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' ','')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace('&','_')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(',','_')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.lower()
#teacher_prefix
project_data['teacher_prefix'].value_counts()
# check if we have any nan values are there
print(project_data['teacher_prefix'].isnull().values.any())
print("number of nan values",project_data['teacher_prefix'].isnull().values.sum())
project_data['teacher_prefix'].replace(np.nan,'Mr.',inplace=True)
project_data['teacher_prefix'] = project_data['teacher_prefix'].str.replace('.','')
project_data['teacher_prefix'] = project_data['teacher_prefix'].str.lower()
project_data['teacher_prefix'].value_counts()
project_data['project_subject_subcategories'].value_counts()
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' The ','')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' ','')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace('&','_')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(',','_')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.lower()
project_data['project_subject_subcategories'].value_counts()
project_data['school_state'].value_counts().head()
project_data['school_state'] = project_data['school_state'].str.lower()
project_data['school_state'].value_counts().head()
# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
# Removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
project_data['project_title'].head(5)
print("printing some random reviews")
print(9, project_data['project_title'].values[9])
print(34, project_data['project_title'].values[34])
print(147, project_data['project_title'].values[147])
# Combining all the above stundents 
from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text
preprocessed_titles = preprocess_text(project_data['project_title'].values)
print("printing some random reviews")
print(9, preprocessed_titles[9])
print(34, preprocessed_titles[34])
print(147, preprocessed_titles[147])
# merge two column text dataframe: 
project_data["essay"] = project_data["project_essay_1"].map(str) +\
                        project_data["project_essay_2"].map(str) + \
                        project_data["project_essay_3"].map(str) + \
                        project_data["project_essay_4"].map(str)
print("printing some random essay")
print(9, project_data['essay'].values[9])
print('-'*50)
print(34, project_data['essay'].values[34])
print('-'*50)
print(147, project_data['essay'].values[147])
preprocessed_essays = preprocess_text(project_data['essay'].values)
print("printing some random essay")
print(9, preprocessed_essays[9])
print('-'*50)
print(34, preprocessed_essays[34])
print('-'*50)
print(147, preprocessed_essays[147])
# https://stackoverflow.com/questions/22407798/how-to-reset-a-dataframes-indexes-for-all-groups-in-one-step
price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
price_data.head(2)
# join two dataframes in python: 
project_data = pd.merge(project_data, price_data, on='id', how='left')
project_data['price'].head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(project_data['price'].values.reshape(-1, 1))
project_data['std_price']=scaler.transform(project_data['price'].values.reshape(-1, 1) )
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(project_data['price'].values.reshape(-1, 1))
project_data['nrm_price']=scaler.transform(project_data['price'].values.reshape(-1, 1))
project_data.head()
preprocessed_data = project_data[['project_is_approved','school_state','teacher_prefix','project_grade_category','project_subject_categories','project_subject_subcategories','project_is_approved','teacher_number_of_previously_posted_projects','essay','price','source']]
preprocessed_data.head()
preprocessed_data.rename(columns={'project_subject_categories':'clean_categories'},inplace=True)
preprocessed_data.rename(columns={'project_subject_subcategories':'clean_subcategories'},inplace=True)
preprocessed_data.head()
preprocessed_data.to_csv('data.csv')
import pandas as pd
data = pd.read_csv('/kaggle/working/data.csv')
data.head()
data.drop(columns=['Unnamed: 0'],inplace=True)
y = data[data['source']=='train']['project_is_approved'].values
X_train = data[data['source']=='train'].drop(['project_is_approved','source'],axis=1)
X_test = data[data['source']=='test'].drop(['project_is_approved','source'],axis=1)
X_train.head(2)
# Splitting data into train, test and Cross Validation set
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, stratify=y)
X_train, X_cv, y_train,y_cv = train_test_split(X_train,y,test_size=0.20,stratify=y)
X_train.head(2)
X_cv.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer_essay = CountVectorizer(min_df=10,ngram_range=(1,3), max_features=2000)
vectorizer2_essay = TfidfVectorizer(min_df=10,ngram_range=(1,3), max_features=2000)

vectorizer_essay.fit(X_train['essay'].values) # fit has to happen only on train data
vectorizer2_essay.fit(X_train['essay'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_essay_bow = vectorizer_essay.transform(X_train['essay'].values)
X_cv_essay_bow = vectorizer_essay.transform(X_cv['essay'].values)
X_test_essay_bow = vectorizer_essay.transform(X_test['essay'].values)
# For set 2
X_train_essay_tfidf = vectorizer2_essay.transform(X_train['essay'].values)
X_cv_essay_tfidf = vectorizer2_essay.transform(X_cv['essay'].values)
X_test_essay_tfidf = vectorizer2_essay.transform(X_test['essay'].values)
print("After vectorizations")
print(X_train_essay_bow.shape, y_train.shape)
print(X_cv_essay_bow.shape, y_cv.shape)
data['school_state'].unique()
vectorizer_ss = CountVectorizer()
#vectorizer2_ss = TfidfVectorizer()

vectorizer_ss.fit(X_train['school_state'].values) # fit has to happen only on train data
#vectorizer2_ss.fit(X_train['school_state'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_state_ohe = vectorizer_ss.transform(X_train['school_state'].values)
X_cv_state_ohe = vectorizer_ss.transform(X_cv['school_state'].values)
X_test_state_ohe = vectorizer_ss.transform(X_test['school_state'].values)

print("After vectorizations")
print(X_train_state_ohe.shape, y_train.shape)
print(X_cv_state_ohe.shape, y_cv.shape)
data['teacher_prefix'].value_counts()
X_cv['teacher_prefix'].fillna('Mr.',inplace=True)
vectorizer_tp = CountVectorizer()
#vectorizer2_tp = TfidfVectorizer()

vectorizer_tp.fit(X_train['teacher_prefix'].values) # fit has to happen only on train data
#vectorizer2_tp.fit(X_train['teacher_prefix'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_teacher_ohe = vectorizer_tp.transform(X_train['teacher_prefix'].values)
X_cv_teacher_ohe = vectorizer_tp.transform(X_cv['teacher_prefix'].values)
X_test_teacher_ohe = vectorizer_tp.transform(X_test['teacher_prefix'].values)

print("After vectorizations")
print(X_train_teacher_ohe.shape, y_train.shape)
print(X_cv_teacher_ohe.shape, y_cv.shape)
data['project_grade_category'].value_counts()
vectorizer_pgc = CountVectorizer()
#vectorizer2_pgc = TfidfVectorizer()

vectorizer_pgc.fit(X_train['project_grade_category'].values) # fit has to happen only on train data
#vectorizer2_pgc.fit(X_train['project_grade_category'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_grade_ohe = vectorizer_pgc.transform(X_train['project_grade_category'].values)
X_cv_grade_ohe = vectorizer_pgc.transform(X_cv['project_grade_category'].values)
X_test_grade_ohe = vectorizer_pgc.transform(X_test['project_grade_category'].values)

print("After vectorizations")
print(X_train_grade_ohe.shape, y_train.shape)
print(X_cv_grade_ohe.shape, y_cv.shape)
print("="*100)
len(data['clean_categories'].value_counts())
vectorizer_cc = CountVectorizer()
#vectorizer2_cc = TfidfVectorizer()

vectorizer_cc.fit(X_train['clean_categories'].values) # fit has to happen only on train data
#vectorizer2_cc.fit(X_train['clean_categories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_clean_categories_ohe = vectorizer_cc.transform(X_train['clean_categories'].values)
X_cv_clean_categories_ohe = vectorizer_cc.transform(X_cv['clean_categories'].values)
X_test_clean_categories_ohe = vectorizer_cc.transform(X_test['clean_categories'].values)


print("After vectorizations")
print(X_train_clean_categories_ohe.shape, y_train.shape)
print(X_cv_clean_categories_ohe.shape, y_cv.shape)
print("="*100)
vectorizer_cs = CountVectorizer()
#vectorizer2_cs = TfidfVectorizer()

vectorizer_cs.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data
#vectorizer2_cs.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer/TfidfVectorizer to convert the text to vector
X_train_clean_subcategories_ohe = vectorizer_cs.transform(X_train['clean_subcategories'].values)
X_cv_clean_subcategories_ohe = vectorizer_cs.transform(X_cv['clean_subcategories'].values)
X_test_clean_subcategories_ohe = vectorizer_cs.transform(X_test['clean_subcategories'].values)
print("After vectorizations")
print(X_train_clean_subcategories_ohe.shape, y_train.shape)
print(X_cv_clean_subcategories_ohe.shape, y_cv.shape)
print("="*100)
from sklearn.preprocessing import Normalizer
normalizer_price = Normalizer()
# Reshapes dataset
# array.reshape(-1, 1) to 2D if your data has a single feature 
normalizer_price.fit(X_train['price'].values.reshape(1,-1))

X_train_price_norm = normalizer_price.transform(X_train['price'].values.reshape(-1,1))
X_cv_price_norm = normalizer_price.transform(X_cv['price'].values.reshape(-1,1))
X_test_price_norm = normalizer_price.transform(X_test['price'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
print(X_cv_price_norm.shape, y_cv.shape)
print("="*100)
from sklearn.preprocessing import Normalizer
normalizer_tnppp = Normalizer()
# Reshapes dataset
# array.reshape(-1, 1) to 2D if your data has a single feature 
normalizer_tnppp.fit(X_train['price'].values.reshape(1,-1))

X_train_tnppp_norm = normalizer_tnppp.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_cv_tnppp_norm = normalizer_tnppp.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_test_tnppp_norm = normalizer_tnppp.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_tnppp_norm.shape, y_train.shape)
print(X_cv_tnppp_norm.shape, y_cv.shape)
print("="*100)
from scipy.sparse import hstack
X_tr = hstack((X_train_essay_bow, X_train_state_ohe, X_train_teacher_ohe, X_train_grade_ohe, X_train_price_norm, X_train_clean_categories_ohe,X_train_clean_subcategories_ohe,X_train_tnppp_norm)).tocsr()
X_cr = hstack((X_cv_essay_bow, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_grade_ohe, X_cv_price_norm, X_cv_clean_categories_ohe,X_cv_clean_subcategories_ohe,X_cv_tnppp_norm)).tocsr()
X_te = hstack((X_test_essay_bow, X_test_state_ohe, X_test_teacher_ohe, X_test_grade_ohe, X_test_price_norm, X_test_clean_categories_ohe,X_test_clean_subcategories_ohe,X_test_tnppp_norm)).tocsr()

X_tr2 = hstack((X_train_essay_tfidf, X_train_state_ohe, X_train_teacher_ohe, X_train_grade_ohe, X_train_price_norm, X_train_clean_categories_ohe,X_train_clean_subcategories_ohe,X_train_tnppp_norm)).tocsr()
X_cr2 = hstack((X_cv_essay_tfidf, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_grade_ohe, X_cv_price_norm, X_cv_clean_categories_ohe,X_cv_clean_subcategories_ohe,X_cv_tnppp_norm)).tocsr()
X_te2 = hstack((X_test_essay_tfidf, X_test_state_ohe, X_test_teacher_ohe, X_test_grade_ohe, X_test_price_norm, X_test_clean_categories_ohe,X_test_clean_subcategories_ohe,X_test_tnppp_norm)).tocsr()

print("Final Data matrix1 for BOW")
print(X_tr.shape, y_train.shape)
print(X_cr.shape, y_cv.shape)
print("="*100)

print("Final Data matrix2 for Tfidf")
print(X_tr2.shape, y_train.shape)
print(X_cr2.shape, y_cv.shape)
print("="*100)
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import GridSearchCV
parameters = {'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
mnb_gs = MultinomialNB()
clf_gs = GridSearchCV(mnb_gs,parameters,cv=4,scoring='roc_auc',return_train_score=True)
clf_gs.fit(X_tr,y_train)
results_gs = pd.DataFrame(clf_gs.cv_results_)
#results_gs
results_gs = results_gs.sort_values(['param_alpha'])
train_auc= results_gs['mean_train_score']
train_auc_std= results_gs['std_train_score']
cv_auc = results_gs['mean_test_score'] 
cv_auc_std= results_gs['std_test_score']
laplaceS_alpha =  results_gs['param_alpha']
laplaceS_alpha = list(map(lambda x: np.log10(x),laplaceS_alpha))
plt.plot(laplaceS_alpha, train_auc, label='Train AUC')
plt.plot(laplaceS_alpha, cv_auc, label='CV AUC')
plt.scatter(laplaceS_alpha, train_auc, label='Train AUC points')
plt.scatter(laplaceS_alpha, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("Laplace Soomthing(alpha), with log: hyperparameter")
plt.ylabel("AUC")
plt.title("Hyper parameter Vs AUC plot(BOW implementation)")
plt.grid()
plt.show()
X_cv
ans = mnb.predict_proba(X_te)[:,1]
print(len(ans))
ans
sample = pd.read_csv('/kaggle/working/sample_submission.csv')
sample.head()
sample['project_is_approved']=ans
sample.head()
sample.to_csv('ans.csv')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
gbdt_clf_tfidf = GradientBoostingClassifier(learning_rate=0.1,min_samples_split=500)
gbdt_clf_tfidf.fit(X_tr,y_train)

y_train_pred =  gbdt_clf_tfidf.predict_proba(X_tr)[:,1] #mnb.predict(X_tr,)
y_test_pred = gbdt_clf_tfidf.predict_proba(X_cr)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_cv, y_test_pred)

plt.plot(train_fpr, train_tpr, label="Train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="Test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC_AUC CURVE(Tfidf implementation)")
plt.grid()
plt.show()

ans_gbdt = gbdt_clf_tfidf.predict_proba(X_te)[:,1]
ans_gbdt
sample['project_is_approved']=ans_gbdt
sample.to_csv('ans_gbdt.csv')