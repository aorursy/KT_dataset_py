# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 500)



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,StandardScaler

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import Word



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier



import wordsegment as ws

import re



from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input/predict-the-data-scientists-salary-in-india-hackathon"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train_df = pd.read_csv("../input/predict-the-data-scientists-salary-in-india-hackathon/Final_Train_Dataset.csv")

test_df = pd.read_csv('../input/predict-the-data-scientists-salary-in-india-hackathon/Final_Test_Dataset.csv')

train_df.drop(['Unnamed: 0'], axis=1,inplace=True)

combine = [train_df, test_df]

#train_df.head(10)
def segmentHashes(text):

  text_tag = " ".join(re.findall(r"[a-zA-Z]+", text))

  #print (text_tag) 

  return " ".join(ws.segment(text_tag)).lower()
labelencoder = LabelEncoder()



ws.load()

for dataset in combine:

    dataset['jd'] = dataset['job_description'].astype(str) + "," + dataset['job_desig'].astype(str) + "," + dataset['job_type'].astype(str) + "," + dataset['key_skills'].astype(str)

    dataset.drop(['job_desig','key_skills','job_description','job_type'], axis=1,inplace=True)

    dataset['experience']=dataset['experience'].str.extract('-(\d+)').astype(int)

    

    #To lower

    dataset['jd'] = dataset['jd'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    

    dataset['jd'] = dataset['jd'].apply(lambda x: " ".join(re.findall(r"[a-zA-Z]+", x)))

    #Removing Punctuation

    dataset['jd'] = dataset['jd'].str.replace('[^\w\s]','')

    #dataset['jd']= dataset['jd'].apply(segmentHashes).astype(str)

    #Common word removal

    #freq = pd.Series(' '.join(dataset['jd']).split()).value_counts()[:15]

    #freq = list(freq.index)

    #dataset['jd'] = dataset['jd'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    #Rare words removal

    #freq = pd.Series(' '.join(dataset['jd']).split()).value_counts()[-10:]

    #freq = list(freq.index)

    #dataset['jd'] = dataset['jd'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    dataset['jd'][:5].apply(lambda x: str(TextBlob(x).correct()))

    dataset['jd']=  dataset['jd'].apply(lambda loc: TextBlob(loc).words).astype(str)

    #Lemmatize

    

    dataset['jd'] = dataset['jd'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    

    #To lower

    dataset['location'] = dataset['location'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    #Removing Punctuation

    dataset['location'] = dataset['location'].str.replace('[^\w\s]','')

    dataset['location']= dataset['location'].apply(segmentHashes).astype(str)

    #Common word removal

    #freq = pd.Series(' '.join(dataset['location']).split()).value_counts()[:10]

    #freq = list(freq.index)

    #dataset['location'] = dataset['location'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    #Rare words removal

    #freq = pd.Series(' '.join(dataset['location']).split()).value_counts()[-10:]

    #freq = list(freq.index)

    #dataset['location'] = dataset['location'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    dataset['location'] = dataset['location'].apply(lambda x: " ".join(re.findall(r"[a-zA-Z]+", x)))

                                      

    dataset['location'][:7].apply(lambda x: str(TextBlob(x).correct()))

    dataset['location']=  dataset['location'].apply(lambda loc: TextBlob(loc).words).astype(str)

    #Lemmatize

    dataset['location'] = dataset['location'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    

    



train_df['salary']=labelencoder.fit_transform(train_df['salary'])
test_df.head(20)

X=train_df.loc[:, train_df.columns != 'salary']

y=train_df.loc[:, train_df.columns == 'salary']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=1)
inner_scaler = StandardScaler(with_mean=False)

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, column):

        self.column = column



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        return X[self.column]



class ArrayCaster(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self



    def transform(self, data):

        return np.transpose(np.matrix(data))

    





clsGBM = Pipeline([

    ('features', FeatureUnion([

            ('experience', Pipeline([

                ('selector',ItemSelector(column='experience')),

                ('caster', ArrayCaster()),

                #('inner_scale', StandardScaler(with_mean=True)), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

           ('company', Pipeline([

                ('selector',ItemSelector(column='company_name_encoded')),

                ('caster', ArrayCaster()),

                ('onhot', preprocessing.OneHotEncoder(handle_unknown='ignore')), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

            ('location', Pipeline([

                ('selector',ItemSelector(column='location')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 9),min_df=2,stop_words='english',sublinear_tf=True)),               

            ])),

             ('jd', Pipeline([

                ('selector',ItemSelector(column='jd')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1,8),stop_words='english',min_df=11,sublinear_tf=True)),

            ]))

         ],

    transformer_weights={

            'experience': 1,

            'location': 0.4,

            'jd': 0.4,

            'company': 0.6,

        },

    )

      

    ),

    ('GBM', GradientBoostingClassifier())

   

]

)





clsLR = Pipeline([

    ('features', FeatureUnion([

            ('experience', Pipeline([

                ('selector',ItemSelector(column='experience')),

                ('caster', ArrayCaster()),

                #('inner_scale', StandardScaler(with_mean=True)), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

            ('company', Pipeline([

                ('selector',ItemSelector(column='company_name_encoded')),

                ('caster', ArrayCaster()),

                ('onhot', preprocessing.OneHotEncoder(handle_unknown='ignore')),

                #('inner_scale', StandardScaler(with_mean=True)), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

            ('location', Pipeline([

                ('selector',ItemSelector(column='location')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 9),min_df=2,stop_words='english',sublinear_tf=True)),

            ])),

             ('jd', Pipeline([

                ('selector',ItemSelector(column='jd')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1,8),stop_words='english',min_df=11,sublinear_tf=True)),

            ]))

         ],

        transformer_weights={

            'experience': 1,

            'location': 0.4,

            'jd': 0.4,

            'company': 0.6,

        },

    )

      

    ),

    ('LR', LogisticRegression())



])



    

clsSVM = Pipeline([

    ('features', FeatureUnion([

            ('experience', Pipeline([

                ('selector',ItemSelector(column='experience')),

                ('caster', ArrayCaster()),

                #('inner_scale', StandardScaler(with_mean=True)), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

            ('company', Pipeline([

                ('selector',ItemSelector(column='company_name_encoded')),

                ('caster', ArrayCaster()),

                ('onhot', preprocessing.OneHotEncoder(handle_unknown='ignore')),

                #('inner_scale', StandardScaler(with_mean=True)), 

                #('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=10, stop_words='english',sublinear_tf=True)),

            ])),

            ('location', Pipeline([

                ('selector',ItemSelector(column='location')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1, 9),min_df=2,stop_words='english',sublinear_tf=True)),

            ])),

             ('jd', Pipeline([

                ('selector',ItemSelector(column='jd')),

                #('vect', CountVectorizer(analyzer="word", stop_words='english')), 

                #('tfidf', TfidfTransformer(use_idf = True)),

                ('tfidf',TfidfVectorizer(analyzer='word',ngram_range=(1,8),stop_words='english',min_df=11,sublinear_tf=True)),

            ]))

         ],

        transformer_weights={

            'experience': 1,

            'location': 0.4,

            'jd': 0.4,

            'company': 0.6,

        },

    )

      

    ),

    ('SVM', SVC(kernel='linear'))

])

from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('gbm', clsGBM), ('lr', clsLR), ('svm', clsSVM)], voting='hard')

eclf = eclf.fit(X_train,y_train)

print(eclf.score(X_test,y_test))
Y_pred = eclf.predict(test_df)

submission = pd.DataFrame({

        "Salary": labelencoder.inverse_transform(Y_pred)

    })

submission.head()

submission.to_excel('submission.xlsx')