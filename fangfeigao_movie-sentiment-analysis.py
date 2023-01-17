## Import basic packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
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
## Read data

train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep="\t") 

test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep="\t") 
train.head()
train.info()
## Show the number of class distributed

plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(x=train.Sentiment,data=train)
train.Phrase[:10]
import string

string.punctuation
train.Phrase=train.Phrase.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)).lower())
train.Phrase[:10]
train.Phrase=train.Phrase.str.split(' ')
train.Phrase[:10]
from nltk.corpus import stopwords

stopwords_e=stopwords.words('english')
stopwords_e=stopwords.words('english')
train.Phrase=[w for w in train.Phrase if w not in stopwords_e]

train.Phrase.head()
import nltk

##nltk.download()
from nltk.stem import WordNetLemmatizer

lemmar=WordNetLemmatizer()
train.Phrase=train.Phrase.apply(lambda x: [lemmar.lemmatize(w) for w in x])
## Method1:

from nltk.stem import PorterStemmer

porter=PorterStemmer()
train.Phrase=train.Phrase.apply(lambda x: [porter.stem(w) for w in x])
## Method2:

from nltk.stem import SnowballStemmer

snow=SnowballStemmer('english')
train.Phrase=train.Phrase.apply(lambda x: [snow.stem(w) for w in x])
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer

vector=TfidfVectorizer(stop_words='english')
train.Phrase=train.Phrase.apply(lambda x: ' '.join(x))
vector1=vector.fit(train.Phrase)
train_feature=vector1.transform(train.Phrase)
train_feature.toarray()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

lr=LogisticRegression(multi_class='ovr')
train.head()
train.info()
lr=lr.fit(train_feature,train.Sentiment)
## Coefficient

lr.coef_
## Get the model performance on train dataset since we don't have test response data

train_predict=lr.predict(train_feature)
## the number of data in each class

train.Sentiment.value_counts().sort_index()
## number of data in predict result

np.unique(train_predict,return_counts=True)
## Plot predict result

plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(train_predict)
print(classification_report(train_predict, train.Sentiment))
from sklearn import svm
svm1=svm.SVC(decision_function_shape='ovo')
svm1.fit(train_feature, train.Sentiment)
svm_train_pred=svm1.predict(train_feature)
## Number of predict class

np.unique(svm_train_pred,return_counts=True)
plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(svm_train_pred)
print(classification_report(svm_train_pred, train.Sentiment))
from sklearn.tree import DecisionTreeClassifier
ds=DecisionTreeClassifier()

ds.fit(train_feature, train.Sentiment)
print(ds.feature_importances_)
ds_train_pred=ds.predict(train_feature)
train.Sentiment.value_counts().sort_index()
## Number of predict class

np.unique(ds_train_pred,return_counts=True)
plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(ds_train_pred)
print(classification_report(ds_train_pred, train.Sentiment))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

rf.fit(train_feature, train.Sentiment)
print(rf.feature_importances_)
rf_train_pred=rf.predict(train_feature)
plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(rf_train_pred)
print(classification_report(rf_train_pred, train.Sentiment))
def data_preprocess(text):

    text_nonpunc=[w.lower() for w in text if w not in string.punctuation]

    text_nonpunc=''.join(text_nonpunc)

    text_rmstop=[x for x in text_nonpunc.split(' ') if x not in stopwords_e]

    text_stem=[snow.stem(w) for w in text_rmstop]

    text1=' '.join(text_stem)

    return (text1)
from sklearn.pipeline import Pipeline
# Can't use TfidVecterizer() because line: 

# https://stackoverflow.com/questions/50192763/python-sklearn-pipiline-fit-attributeerror-lower-not-found

# TfidTransformer should combine with countVectorizer()

lrpipeline=Pipeline([('preprocess',CountVectorizer(analyzer=data_preprocess)),

                  ('Tfidf',TfidfTransformer()),

                  ('classify',LogisticRegression())])
lrpipeline.fit(train.Phrase,train.Sentiment)
## have to saved the vocabulary

result=lrpipeline.predict(test['Phrase'])
np.unique(result)
plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of sentiment class')

sns.countplot(result)
## Import every packages

from scipy import stats

import string

from nltk.corpus import stopwords

stopwords_e=stopwords.words('english')

from nltk.stem import SnowballStemmer

snow=SnowballStemmer('english')

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import  RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer

vector=TfidfVectorizer(stop_words='english')
## Preprocess function

def data_preprocess(text):

    text_nonpunc=[w.lower() for w in text if w not in string.punctuation]

    text_nonpunc=''.join(text_nonpunc)

    text_rmstop=[x for x in text_nonpunc.split(' ') if x not in stopwords_e]

    text_stem=[snow.stem(w) for w in text_rmstop]

    text1=' '.join(text_stem)

    return (text1)
## OOP Class 

## Notice: Class name and the first def should have a blank line

class EstimatorSelection:

    

    def __init__(self, models):

        self.models=models

        self.keys=models.keys()

        self.results={}

        self.modelfit={}

        self.modelpredict={}

    def fit(self, x, y):

        x1=x.apply(lambda i: data_preprocess(i))

        x_feature1=vector.fit_transform(x1)

        for key in self.keys:

            model=self.models[key]

            self.modelfit[key]=model.fit(x_feature1,y)

            y_pred=model.predict(x_feature1)

            self.results[key]=classification_report(y, y_pred,output_dict=True)

    def predict(self,test_x):

        test_x1=test_x.apply(lambda i: data_preprocess(i))

        test_feature1=vector.transform(test_x1)

        test_frames=[]

        for key in self.keys:

            modelfit=self.modelfit[key]

            test_y=modelfit.predict(test_feature1)

            test_frame=pd.DataFrame(test_y,columns=[key])

            test_frames.append(test_frame)

        predict_frame=pd.concat(test_frames,axis=1)            

        return(predict_frame)     

    def summary(self):

        Frames=[]

        for key in self.keys:

            result=self.results[key]

            Frame=pd.DataFrame(result['macro avg'], index=[key])

            Frames.append(Frame)

        result_sum=pd.concat(Frames)

        return result_sum.iloc[:,:3]
## Models want to predict on test data

models = { 

    'LogisticClassifier': LogisticRegression(multi_class='ovr'),

    'RandomforestClassifier':RandomForestClassifier(),

    'DecisionTreeClassifier':DecisionTreeClassifier()

}
model_compare=EstimatorSelection(models)
model_compare.fit(train.Phrase, train.Sentiment)
summary=model_compare.summary()

summary
predict_result=model_compare.predict(test.Phrase)

predict_result
predict_result1=predict_result.reset_index().rename(columns={'index':'case'})

predict_result2=pd.melt(predict_result1,id_vars='case', value_vars=['LogisticClassifier', 'RandomforestClassifier', 'DecisionTreeClassifier'])
predict_result2=pd.melt(predict_result1,id_vars='case', value_vars=['LogisticClassifier', 'RandomforestClassifier', 'DecisionTreeClassifier'])

predict_result2
predict_result3=predict_result2.groupby(['variable','value']).size().reset_index().rename(columns={0:'count'})

predict_result3
plt.figure(figsize=(10,5))

ax=plt.axes()

ax.set_title('Number of class for each methods')

sns.barplot(x='value', y='count', hue='variable', data=predict_result3)
Final_results=[]

for i in range(predict_result1.shape[0]):

    Final_result=stats.mode(predict_result1.iloc[i,]).mode.item()

    Final_results.append(Final_result)
predict_result1['Final_result']=Final_results

predict_result1
test['Sentiment']=Final_results

test
#make the predictions with trained model and submit the predictions.

sub_file = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv',sep=',')

sub_file.Sentiment=Final_results

sub_file.to_csv('Submission.csv',index=False)