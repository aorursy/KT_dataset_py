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
#df column names: ['id', 'keyword', 'location', 'text', 'target']

train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

train_data.head(10)
#Dropping keyword and location columns since we are mainly interested 

#in solely classifying with tweets. 

train_data = train_data.drop(columns=['keyword','location'])

test_data = test_data.drop(columns=['keyword','location'])
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier,LogisticRegression

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.base import TransformerMixin 

from sklearn.metrics import f1_score,accuracy_score

from sklearn.utils import shuffle



#matrix error fix

class DenseMatrix(TransformerMixin):

    def fit(self, X, y=None, **params):

        return self

    def transform(self, X, y=None, **params):

        return X.todense()

    

#(text + model) processing pipeline

def pipeline(models):

    #text-pre/processing pipeline

    text_pipeline = Pipeline([

        ('shuffle',shuffle(random_state=44)),

        ('c_vect',CountVectorizer(

            analyzer = 'word',

            stop_words = 'english',

            lowercase = True,

            strip_accents = 'ascii'

        )),

        ('t_vect',TfidfTransformer()),

        ('d_matrix',DenseMatrix())

    ])

    #add each specified model to pipeline

    for k in range(len(models)):

        model = models[k]

        inner_pipeline = Pipeline([

            ('textpipeline',text_pipeline),

            ('model',model)

        ])

        models[k] = inner_pipeline 

    return models



#fit each model in pipeline 

def fit(x,y,pipeline):

    for model in pipeline:

        model.fit(x,y)

    return pipeline



#test each fitted model in pipeline using specified metrics (f1/accuracy)

#could write a kfold cv test function, however, quite slow 

#decided to use a hold-out sample instead

def test(x,y,pipeline):

    f1=[]

    acc=[]

    model_name=[]

    for model in pipeline: 

        p = model.predict(x)

        f1.append(f1_score(y,p))

        acc.append(accuracy_score(y,p))

        model_name.append(model[1].__class__.__name__)

    report = {'model' : model_name,'accuracy' : acc,'f1_score' : f1}

    return pd.DataFrame(data=report) 
#70/30 train-test split

x_train,x_test,y_train,y_test = train_test_split(train_data['text'],train_data['target'],test_size=0.3,random_state=13) 



models = [MultinomialNB(),GaussianNB(),BernoulliNB(),SGDClassifier(),RandomForestClassifier(),LogisticRegression()]

model_pipeline = pipeline(models)

trained_models = fit(x_train,y_train,model_pipeline)



#test fitted models on hold-out sample 

test(x_test,y_test,trained_models)
#we will go ahead and submit using the SGDClassifier, not too bad considering no parameters were tinkered with.

sgd_model = trained_models[3]

p = sgd_model.predict(test_data['text'])

submission['target'] = p
submission.to_csv('submission.csv',index=False)