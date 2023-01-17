import numpy as np 

import pandas as pd

import gc
init_df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')

init_df.head()
init_df.nunique()
drop_columns = ['job_id', 'title', 'location', 'department', 'industry', 'function']



proc_df = init_df.drop(drop_columns, axis=1)



del init_df

gc.collect()
proc_df.isnull().sum()
cat_columns = ['employment_type', 'required_experience', 'required_education']



for col in cat_columns:

    proc_df[col].fillna("Unknown", inplace=True)

text_columns = ['company_profile', 'description', 'requirements', 'benefits']



proc_df = proc_df.dropna(subset=text_columns, how='all')



for col in text_columns:

    proc_df[col].fillna(' ', inplace=True)

    
unique_salary = proc_df['salary_range'].unique()

print(unique_salary[0:5])
new = proc_df['salary_range'].str.split("-", n = 1, expand = True) 



proc_df['salary_range_min']= new[0]

proc_df['salary_range_max']= new[1]



proc_df['salary_range_min'].fillna('-1', inplace=True)

proc_df['salary_range_max'].fillna('-1', inplace=True)



def remove_string(x):

    if not x.isnumeric(): 

        val = '-1'

    else:

        val = x

    return val



proc_df['salary_range_min'] = proc_df['salary_range_min'].apply(lambda x: remove_string(x))

proc_df['salary_range_max'] = proc_df['salary_range_max'].apply(lambda x: remove_string(x))



proc_df.drop('salary_range', axis=1, inplace = True) 
cat_eda_columns = ['telecommuting', 'has_company_logo', 'has_questions', 'employment_type', 'required_experience', 'required_education']



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec 



grid = gridspec.GridSpec(5, 2, wspace=0.5, hspace=0.5) 

plt.figure(figsize=(15,25)) 



for n, col in enumerate(proc_df[cat_eda_columns]): 

    ax = plt.subplot(grid[n]) 

    sns.countplot(x=col, data=proc_df, hue='fraudulent', palette='Set2', order=proc_df[col].value_counts().iloc[:5].index) 

    ax.set_ylabel('Count', fontsize=12)

    ax.set_title(f'{col} Distribution by Target', fontsize=15) 

    xlabels = ax.get_xticklabels() 

    ax.set_xticklabels(xlabels,  fontsize=10)

    plt.legend(fontsize=8)

    plt.xticks(rotation=30) 

    total = len(proc_df)

    sizes=[] 

    for p in ax.patches: 

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=10) 

    

plt.show()

text_cols = ['company_profile', 'description', 'requirements', 'benefits']



for col in text_cols:

    fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(10, 2.5), dpi=100)

    num=proc_df[proc_df["fraudulent"]==1][col].str.split().map(lambda x: len(x))

    ax1.hist(num,bins = 20,color='orangered')

    ax1.set_title('Fake Post')

    num=proc_df[proc_df["fraudulent"]==0][col].str.split().map(lambda x: len(x))

    ax2.hist(num, bins = 20)

    ax2.set_title('Real Post')

    fig.suptitle(f'Words in {col}')

    plt.show()
text_cols = ['company_profile', 'description', 'requirements', 'benefits']



proc_df['aggr_post'] = proc_df[text_cols].apply(lambda x: ' '.join(x), axis=1)

proc_df.drop(text_cols, axis=1, inplace=True)



proc_df.head()



print(proc_df.loc[0, 'aggr_post'])
import langid



def detect_lang(x):

    code,_ = langid.classify(x)

    

    return code



proc_df = proc_df[proc_df['aggr_post'].apply(lambda x: detect_lang(x) == 'en')]



proc_df.head()
import re

import string



def clean_text(text):

    text = text.lower()                                              # make the text lowercase

    text = re.sub('\[.*?\]', '', text)                               # remove text in brackets

    text = re.sub('http?://\S+|www\.\S+', '', text)                  # remove links

    text = re.sub('https?://\S+|www\.\S+', '', text)                 # remove links

    text = re.sub('<.*?>+', '', text)                                # remove HTML stuff

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # get rid of punctuation

    text = re.sub('\n', '', text)                                    # remove line breaks

    #text = re.sub('\w*\d\w*', '', text)                             # remove anything with numbers, if you want

    #text = re.sub(r'[^\x00-\x7F]+',' ', text)                       # remove unicode

    return text



proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x: clean_text(x))



proc_df.head()
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

stop_words = stopwords.words('english')



def remove_stopwords(text):

    words = [w for w in text if w not in stop_words]

    return words



def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text



proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x: tokenizer.tokenize(x))

proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x : remove_stopwords(x))

proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x : combine_text(x))



proc_df.head()
random_state = 42



real_df = proc_df[proc_df['fraudulent']==0].copy()

fake_df = proc_df[proc_df['fraudulent']==1].copy()



real_sampled_df = real_df.sample(n=3000, random_state=random_state)



final_df = pd.concat([real_sampled_df, fake_df], axis=0)
del proc_df

del real_df

del real_sampled_df

del fake_df



gc.collect()
from sklearn.model_selection import train_test_split



seed_state = 315

random_state = 42



real_df = final_df[final_df['fraudulent']==0]

fake_df = final_df[final_df['fraudulent']==1]



y_real = real_df['fraudulent'].copy()

x_real = real_df.drop(['fraudulent'], axis=1)



y_fake = fake_df['fraudulent'].copy()

x_fake = fake_df.drop(['fraudulent'], axis=1)



x_real_tv, x_real_test, y_real_tv, y_real_test = train_test_split(x_real, y_real, test_size=0.3, random_state=seed_state)

x_real_train, x_real_val, y_real_train, y_real_val = train_test_split(x_real_tv, y_real_tv, test_size=0.2, random_state=seed_state)



x_fake_tv, x_fake_test, y_fake_tv, y_fake_test = train_test_split(x_fake, y_fake, test_size=0.3, random_state=seed_state)

x_fake_train, x_fake_val, y_fake_train, y_fake_val = train_test_split(x_fake_tv, y_fake_tv, test_size=0.2, random_state=seed_state)



x_train = pd.concat([x_real_train, x_fake_train])

y_train = pd.concat([y_real_train, y_fake_train])



x_val = pd.concat([x_real_val, x_fake_val])

y_val = pd.concat([y_real_val, y_fake_val])



x_test = pd.concat([x_real_test, x_fake_test])

y_test = pd.concat([y_real_test, y_fake_test])
x_train_post = x_train['aggr_post'].copy()

x_val_post = x_val['aggr_post'].copy()

x_test_post = x_test['aggr_post'].copy()



x_train_cat = x_train.drop(['aggr_post'], axis=1)

x_val_cat = x_val.drop(['aggr_post'], axis=1)

x_test_cat = x_test.drop(['aggr_post'], axis=1)
import seaborn as sns

from sklearn.metrics import confusion_matrix, mean_absolute_error, make_scorer 



# Showing Confusion Matrix

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(14,14)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer 



count_vectorizer = CountVectorizer()

x_train_post_vec = count_vectorizer.fit_transform(x_train_post)

x_val_post_vec = count_vectorizer.transform(x_val_post)

x_test_post_vec = count_vectorizer.transform(x_test_post) 



lr_post = LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, verbose=0, n_jobs=-1)

lr_post.fit(x_train_post_vec, y_train)
weights = lr_post.coef_

abs_weights = np.abs(weights)
lr_post_val_preds = lr_post.predict(x_val_post_vec)



f1_score(y_val, lr_post_val_preds, average = 'macro')

plot_cm(y_val, lr_post_val_preds, 'Confusion Matrix: LR Validation Set Predictions ')
lr_post_test_preds = lr_post.predict(x_test_post_vec)



f1_score(y_test, lr_post_test_preds, average = 'macro')

plot_cm(y_test, lr_post_test_preds, 'Confusion Matrix: LR Test Set Predictions ')
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



le_employment = LabelEncoder()

le_experience = LabelEncoder()

le_education  = LabelEncoder()



x_train_cat['employment_type'] = le_employment.fit_transform(x_train_cat['employment_type'])

x_val_cat['employment_type'] = le_employment.transform(x_val_cat['employment_type'])

x_test_cat['employment_type'] = le_employment.transform(x_test_cat['employment_type'])



x_train_cat['required_experience'] = le_experience.fit_transform(x_train_cat['required_experience'])

x_val_cat['required_experience'] = le_experience.transform(x_val_cat['required_experience'])

x_test_cat['required_experience'] = le_experience.transform(x_test_cat['required_experience'])



x_train_cat['required_education'] = le_education.fit_transform(x_train_cat['required_education'])

x_val_cat['required_education'] = le_education.transform(x_val_cat['required_education'])

x_test_cat['required_education'] = le_education.transform(x_test_cat['required_education'])



rf_cat = RandomForestClassifier(n_estimators=2000,bootstrap=True)

rf_cat.fit(x_train_cat, y_train)
rf_cat_val_pred = rf_cat.predict(x_val_cat)



f1_score(y_val, rf_cat_val_pred.round(), average = 'macro')

plot_cm(y_val, rf_cat_val_pred.round(), 'Confusion Matrix: RF Validation Set Predictions ')
rf_cat_test_pred = rf_cat.predict(x_test_cat)



f1_score(y_test, rf_cat_test_pred.round(), average = 'macro')

plot_cm(y_test, rf_cat_test_pred.round(), 'Confusion Matrix: RF Test Set Predictions ')
aggregate_val = pd.DataFrame()

aggregate_val['post_preds'] = lr_post_val_preds

aggregate_val['cat_preds'] = rf_cat_val_pred

aggregate_val.head()
aggregate_test = pd.DataFrame()

aggregate_test['post_preds'] = lr_post_test_preds

aggregate_test['cat_preds'] = rf_cat_test_pred

aggregate_test.head()
lr_final = LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, verbose=0, n_jobs=-1)

lr_final.fit(aggregate_val, y_val)



lr_final_preds = lr_final.predict(aggregate_test)



f1_score(y_test, lr_final_preds, average = 'macro')

plot_cm(y_test, lr_final_preds, 'Confusion Matrix: Aggregate Model Final Predictions ')