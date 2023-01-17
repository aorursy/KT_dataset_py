# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.pipeline import Pipeline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#readind files

path_data_test='/kaggle/input/nlp-getting-started/test.csv'

test_data=pd.read_csv(path_data_test)

path_data_train='/kaggle/input/nlp-getting-started/train.csv'

train_data=pd.read_csv(path_data_train)
#Data cleaning from stopwords and repeating meaningless words that wont help in the prediction and training

#Searched for that cleaning part that use regex library which is built in library (regular expresions to clean)



def clean(text):

    text = re.sub(r"\n","",text)

    text = text.lower()

    text = re.sub(r"\d","",text)        #Remove digits

    text = re.sub(r'[^\x00-\x7f]',r' ',text) # Remove non-ascii

    text = re.sub(r'[^\w\s]','',text) #Remove punctuation

    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http

    return text





#Apply cleaning func using lampda and adding new column called cleaned 



train_data['cleaned'] = train_data['text'].apply(lambda x : clean(x))

test_data['cleaned']= test_data['text'].apply(lambda x : clean(x))

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

from sklearn import svm

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import KNeighborsClassifier



#Related to features

tweets_pipeline = Pipeline([('CVec', CountVectorizer(stop_words='english')),

                     ('Tfidf', TfidfTransformer())])





#Get train ready

X=train_data['cleaned'].to_numpy()

Y=train_data['target'].to_numpy()

X_train_tranformed = tweets_pipeline.fit_transform(X)





#Get test ready

X_test=test_data['cleaned']

X_test_tranformed = tweets_pipeline.transform(X_test)







#Train the model

train_x,val_x,train_y,val_y=train_test_split(X,Y,random_state=0)

model=KNeighborsClassifier(n_neighbors=7)

model.fit(X_train_tranformed,Y)



######################################################################################333

#Test and output part



preds=model.predict(X_test_tranformed)



output = pd.DataFrame({'id': test_data.id, 'target': preds})

output.to_csv('sample_submission.csv', index=False)

#mae=mean_absolute_error(val_y,preds)

#print(mae)
