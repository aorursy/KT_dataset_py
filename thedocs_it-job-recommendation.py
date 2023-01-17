# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import nltk

from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder

from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score



    

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



data = pd.read_csv('/kaggle/input/jobposts/data job posts.csv')

print(data.columns)

# selecting only IT Jobs

df = data[data['IT']]

# selecting 

cols = ['RequiredQual', 'Eligibility', 'Title', 'JobDescription', 'JobRequirment']

df=df[cols]

df.head(5)
classes = df['Title'].value_counts()[:21]

keys = classes.keys().to_list()



df = df[df['Title'].isin(keys)]

df['Title'].value_counts()
def chane_titles(x):

    x = x.strip()

    if x == 'Senior Java Developer':

        return 'Java Developer'

    elif x == 'Senior Software Engineer':

        return 'Software Engineer'

    elif x == 'Senior QA Engineer':

        return 'Software QA Engineer'

    elif x == 'Senior Software Developer':

        return 'Senior Web Developer'

    elif x =='Senior PHP Developer':

        return 'PHP Developer'

    elif x == 'Senior .NET Developer':

        return '.NET Developer'

    elif x == 'Senior Web Developer':

        return 'Web Developer'

    elif x == 'Database Administrator':

        return 'Database Admin/Dev'

    elif x == 'Database Developer':

        return 'Database Admin/Dev'



    else:

        return x

        

    

df['Title'] = df['Title'].apply(chane_titles)

df['Title'].value_counts()
from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):

    def __init__(self):

        # lemmatize text - convert to base form 

        self.wnl = WordNetLemmatizer()

        # creating stopwords list, to ignore lemmatizing stopwords 

        self.stopwords = stopwords.words('english')

    def __call__(self, doc):

        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]



# removing new line characters, and certain hypen patterns                  

df['RequiredQual']=df['RequiredQual'].apply(lambda x: x.replace('\n', ' ').replace('\r', '').replace('- ', ''). replace(' - ', ' to '))
from sklearn.feature_extraction.text import TfidfVectorizer

# train features and labels 

y = df['Title']

X = df['RequiredQual']

# tdif feature rep 

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')

vectorizer.fit(X)

# transoforming text to tdif features

tfidf_matrix = vectorizer.transform(X)

# sparse matrix to dense matrix for training

X_tdif = tfidf_matrix.toarray()

# encoding text labels in categories 

enc = LabelEncoder() 

enc.fit(y.values)

y_enc=enc.transform(y.values)



X_train_words, X_test_words, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=10)



X_train = vectorizer.transform(X_train_words)

X_train = X_train.toarray()



X_test = vectorizer.transform(X_test_words)

X_test = X_test.toarray()

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

gnb = GaussianNB()

train_preds = gnb.fit(X_train, y_train).predict(X_train)

test_preds = gnb.predict(X_test)



print('Train acc: {0}'.format(accuracy_score(y_train, train_preds)))

print('Test acc: {0}'.format(accuracy_score(y_test, test_preds)))

from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(max_iter=15,verbose=1, C=0.75)



train_preds = logistic.fit(X_train, y_train).predict(X_train)

test_preds = logistic.predict(X_test)



print('Train acc: {0}'.format(accuracy_score(y_train, train_preds)))

print('Test acc: {0}'.format(accuracy_score(y_test, test_preds)))

preds_data = {'Current Position Requirments': [], 'Current Position': [], 'Alternative 1': [], 'Alternative 2': []}

y_preds_proba = logistic.predict_proba(X_test)



counter = 0 

for idx, (pred_row, true_job_position) in enumerate(zip(y_preds_proba, y_test)):

    class_preds = np.argsort(pred_row)

    # delete true class

    for i in [-1, -2]:

        if class_preds[i] == true_job_position:

            class_preds=np.delete(class_preds,i)

    # getting other 2 highest job predictions         

    top_classes = class_preds[-2:]

    # obtaining class name string from int label 

    class_names = enc.inverse_transform(top_classes)

    true_job_position_name = enc.inverse_transform([true_job_position])

    # saving to dict

    preds_data['Current Position Requirments'].append(X_test_words.iloc[idx])

    preds_data['Current Position'].append(true_job_position_name[0])

    preds_data['Alternative 1'].append(class_names[1])

    preds_data['Alternative 2'].append(class_names[0])



    

    counter +=1
preds_df = pd.DataFrame.from_dict(preds_data)

preds_df.to_csv('Recommendations.csv', index=False)

preds_df
