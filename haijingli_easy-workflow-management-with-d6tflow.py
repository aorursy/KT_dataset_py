!pip install d6tflow
import d6tflow

import luigi



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import defaultdict



import regex as re

import string

import spacy

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate
class TaskGetData(d6tflow.tasks.TaskPqPandas): #save to parquet, load as pandas

    persist=['df_train','df_test'] # declare what you will save

    

    def run(self):

        df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

        df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

        self.save({'df_train':df_train, 'df_test':df_test})
d6tflow.run(TaskGetData())
df_train = TaskGetData().output()['df_train'].load()

df_test = TaskGetData().output()['df_test'].load()
df_train.head()
df_test.head()
df_train.info()
print('% missing values')

print(df_train.apply(lambda x: x.isna().sum()/len(x)))
y=df_train.target.value_counts()

sns.barplot([f'non-disaster({y[0]})',f'disaster:({y[1]})'],y,hue = y.index)

plt.gca().set_ylabel('count')
df_train.groupby(['keyword','target']).size().unstack(1).fillna(0).sort_values([1]).plot(kind = 'barh',figsize=(20,80))
@d6tflow.requires(TaskGetData) # define upstream dependency

class TaskGetNGrams(d6tflow.tasks.TaskPqPandas):

    

    persist = ['df_disaster_ngrams','df_nondisaster_ngrams'] # must declare what to save if a single task has multiple outputs

    n_gram = luigi.IntParameter(default=1) 



    def generate_ngrams(self,text, n_gram):

        stop_words = set(stopwords.words('english'))

        token = [token for token in text.lower().split(' ') if token != '' if token not in stop_words]

        ngrams = zip(*[token[i:] for i in range(n_gram)])

        return [' '.join(ngram) for ngram in ngrams]

    

    def run(self):

        df_train = self.input()['df_train'].load() #load inputs from upstream dependency's outputs

        df_test = self.input()['df_test'].load()

        

        DISASTER_TWEETS = df_train['target']==1

        disaster_ngrams = defaultdict(int)

        nondisaster_ngrams = defaultdict(int)

        

        for tweet in df_train[DISASTER_TWEETS]['text']:

            for word in self.generate_ngrams(tweet,self.n_gram):

                disaster_ngrams[word] += 1

        

        for tweet in df_train[~DISASTER_TWEETS]['text']:

            for word in self.generate_ngrams(tweet,self.n_gram):

                nondisaster_ngrams[word] += 1

        

        df_disaster_ngrams = pd.DataFrame(sorted(disaster_ngrams.items(), key=lambda x: x[1])[::-1],columns = ['term','count'])

        df_nondisaster_ngrams = pd.DataFrame(sorted(nondisaster_ngrams.items(), key=lambda x: x[1])[::-1],columns = ['term','count'])

        self.save({'df_disaster_ngrams':df_disaster_ngrams, 'df_nondisaster_ngrams':df_nondisaster_ngrams}) #save outputs,should match self.persist

   
@d6tflow.requires(TaskGetNGrams) #define upstream dependency

class TaskVisualize(d6tflow.tasks.TaskCache): #save to memory

 

    def run(self):

        df_disaster_ngrams = self.input()['df_disaster_ngrams'].load()

        df_nondisaster_ngrams = self.input()['df_nondisaster_ngrams'].load()

        fig, axs = plt.subplots(1, 2, figsize=(40,30),squeeze=False)

        a = sns.barplot(df_disaster_ngrams['count'][:50],df_disaster_ngrams['term'][:50], ax = axs[0][0],color = 'b').set_title(f'Top 50 most common {self.n_gram}-gram in disaster tweets',fontsize=30)

        b = sns.barplot(df_nondisaster_ngrams['count'][:50],df_nondisaster_ngrams['term'][:50], ax = axs[0][1],color = 'y').set_title(f'Top 50 most common {self.n_gram} in non-disaster tweets',fontsize=30)

        plt.show()

        

    
d6tflow.preview(TaskVisualize())
d6tflow.run(TaskVisualize())
d6tflow.preview(TaskVisualize(n_gram = 2)) # change parameter
d6tflow.run(TaskVisualize(n_gram = 2))
stemmer = SnowballStemmer(language='english')

nlp = spacy.load("en_core_web_sm")
@d6tflow.requires(TaskGetData) #define upstream dependency

class TaskPreprocess(d6tflow.tasks.TaskPqPandas):

    persist=['df_train','df_test']

    do_preprocess = luigi.BoolParameter(default = False)  # We will see if preprocessing really improves model performance

    preprocess_method = luigi.Parameter(default = 'typical') # We will then add preprocessing especially for word embeddings

    

    def typical_preprocess(self,text):

        # Make text lowercase, remove links, stem tokens,remove stop words, remove word of length <=1

        

        text = text.lower()

        text = re.sub(r'http://\S+|https://\S+', '', text)

        text = re.sub(r'www.\S+\.com','',text)

        text = text.replace('...', '')

        text = text.replace('..','')

        text = text.replace("'s","")

        text = [stemmer.stem(token.text) for token in nlp(text) if not (token.is_stop | token.is_punct | len(token.text)<=1)]

        text = ' '.join(text)

        return text

    

    # We can then add preprocessing for word embedding with :

#     def word_embedding_preprocess(self,text):

#         pass

    

    def run(self):

        df_train = self.input()['df_train'].load()

        df_test = self.input()['df_test'].load()

        if self.do_preprocess:

            if self.preprocess_method == 'typical':

                df_train['text']=df_train['text'].apply(lambda x: self.typical_preprocess(x))

                df_test['text']=df_test['text'].apply(lambda x: self.typical_preprocess(x))

            elif self.preprocess_mehod == 'embedding':

                pass

        self.save({'df_train':df_train, 'df_test':df_test})  

    
d6tflow.invalidate_downstream(TaskGetNGrams(), TaskVisualize())

d6tflow.invalidate_downstream(TaskGetNGrams(n_gram = 2), TaskVisualize(n_gram = 2))
d6tflow.preview(TaskVisualize())
d6tflow.preview(TaskVisualize(n_gram = 2))
# You don't need to rewrite all the codes, just change the upstream dependency.

# I copy and paste the codes to ensure kaggle kernel run smoothly



@d6tflow.requires(TaskPreprocess) # change upstream dependency 

class TaskGetNGrams(d6tflow.tasks.TaskPqPandas):

    

    persist = ['df_disaster_ngrams','df_nondisaster_ngrams']

    n_gram = luigi.IntParameter(default=1)



    def generate_ngrams(self,text, n_gram):

        stop_words = set(stopwords.words('english'))

        token = [token for token in text.lower().split(' ') if token != '' if token not in stop_words]

        ngrams = zip(*[token[i:] for i in range(n_gram)])

        return [' '.join(ngram) for ngram in ngrams]

    

    def run(self):

        df_train = self.input()['df_train'].load()

        df_test = self.input()['df_test'].load()

        

        DISASTER_TWEETS = df_train['target']==1

        disaster_ngrams = defaultdict(int)

        nondisaster_ngrams = defaultdict(int)

        

        for tweet in df_train[DISASTER_TWEETS]['text']:

            for word in self.generate_ngrams(tweet,self.n_gram):

                disaster_ngrams[word] += 1

        

        for tweet in df_train[~DISASTER_TWEETS]['text']:

            for word in self.generate_ngrams(tweet,self.n_gram):

                nondisaster_ngrams[word] += 1

        

        df_disaster_ngrams = pd.DataFrame(sorted(disaster_ngrams.items(), key=lambda x: x[1])[::-1],columns = ['term','count'])

        df_nondisaster_ngrams = pd.DataFrame(sorted(nondisaster_ngrams.items(), key=lambda x: x[1])[::-1],columns = ['term','count'])

        self.save({'df_disaster_ngrams':df_disaster_ngrams, 'df_nondisaster_ngrams':df_nondisaster_ngrams})

        

@d6tflow.requires(TaskGetNGrams)

class TaskVisualize(d6tflow.tasks.TaskCache):

    from matplotlib import pyplot as plt

    import seaborn as sns

 

    def run(self):

        df_disaster_ngrams = self.input()['df_disaster_ngrams'].load()

        df_nondisaster_ngrams = self.input()['df_nondisaster_ngrams'].load()

        fig, axs = plt.subplots(1, 2, figsize=(40,30),squeeze=False)

        a = sns.barplot(df_disaster_ngrams['count'][:50],df_disaster_ngrams['term'][:50], ax = axs[0][0],color = 'b').set_title(f'Top 50 most common {self.n_gram}-gram in disaster tweets',fontsize=30)

        b = sns.barplot(df_nondisaster_ngrams['count'][:50],df_nondisaster_ngrams['term'][:50], ax = axs[0][1],color = 'y').set_title(f'Top 50 most common {self.n_gram} in non-disaster tweets',fontsize=30)

        plt.show()
d6tflow.preview(TaskPreprocess())
d6tflow.preview(TaskVisualize())
d6tflow.preview(TaskVisualize(do_preprocess = True))
d6tflow.run(TaskVisualize(do_preprocess = True))
@d6tflow.requires(TaskPreprocess)

class TaskGetFeatures(d6tflow.tasks.TaskPickle):

    persist = ['X_train','Y_train','X_test']

    feature = luigi.Parameter(default = 'tf-idf')

    n_gram_range = luigi.Parameter(default = (1,1)) 

    

    def run(self):

        df_train = self.input()['df_train'].load()

        df_test = self.input()['df_test'].load()

        X_train, X_test = df_train['text'],df_test['text']

        Y_train = df_train['target']

        

        if self.feature == 'tf-idf':

            tfidf = TfidfVectorizer(ngram_range = self.n_gram_range)

            X_train = tfidf.fit_transform(X_train)

            X_test = tfidf.transform(X_test)

        

        self.save({'X_train': X_train, 'Y_train': Y_train, 'X_test':X_test})

  
d6tflow.preview(TaskGetFeatures(do_preprocess = True))
d6tflow.run(TaskGetFeatures(do_preprocess = True))
d6tflow.preview(TaskGetFeatures(do_preprocess = True,n_gram_range = (2,2))) #only use bigrams
d6tflow.run(TaskGetFeatures(do_preprocess = True,n_gram_range = (2,2))) # only use bigrams
@d6tflow.requires(TaskGetFeatures)

class TaskTrain(d6tflow.tasks.TaskPickle):

    model = luigi.Parameter(default = 'logistic')

    

    def run(self):

        X_train = self.input()['X_train'].load()

        Y_train = self.input()['Y_train'].load()

        

        if self.model == 'logistic':

            model = LogisticRegression()

        

        elif self.model == 'svm':

            model = SVC()

        

        elif self.model == 'KNN':

            model = KNeighborsClassifier(2)

        else:

            raise ValueError('invalid model selection')

        

        model.fit(X_train,Y_train)

        self.save(model) 

logistic_model = TaskTrain(do_preprocess=True, model='logistic' , n_gram_range = (1,1))

d6tflow.preview(logistic_model)
X_train = TaskGetFeatures(do_preprocess = True).output()['X_train'].load() # Load features from default feature engineering process

Y_train = TaskGetFeatures(do_preprocess = True).output()['Y_train'].load()

X_test = TaskGetFeatures(do_preprocess = True).output()['X_test'].load()
#unigram

#logistic model

logistic_model = TaskTrain(do_preprocess=True, model='logistic')

d6tflow.run(logistic_model)

logistic_model = logistic_model.output().load()



#svm model

svm_model = TaskTrain(do_preprocess = True, model='svm')

d6tflow.run(svm_model)

svm_model = svm_model.output().load()



#KNN model

KNN_model = TaskTrain(do_preprocess = True, model='KNN')

d6tflow.run(KNN_model)

KNN_model = KNN_model.output().load()



print('Insample accuracy ')

print('logistic_model: ', sklearn.metrics.accuracy_score(Y_train,logistic_model.predict(X_train))) 

print('svm_model: ', sklearn.metrics.accuracy_score(Y_train,svm_model.predict(X_train)))

print('KNN_model: ', sklearn.metrics.accuracy_score(Y_train,KNN_model.predict(X_train)))
models = {'logistics' : logistic_model,

         'svm' : svm_model,

         'KNN' : KNN_model

         }



print ('model  ','mean_accuracy   ', 'mean_f1   ')

for key, model in models.items():

    scores = cross_validate(model, X_train, Y_train, scoring=('accuracy', 'f1'), cv=5)

    print(f'{key}', scores['test_accuracy'].mean(),scores['test_f1'].mean())
X_train = TaskGetFeatures(do_preprocess = True,n_gram_range = (2,2)).output()['X_train'].load() # Load features

Y_train = TaskGetFeatures(do_preprocess = True,n_gram_range = (2,2)).output()['Y_train'].load()

X_test = TaskGetFeatures(do_preprocess = True,n_gram_range = (2,2)).output()['X_test'].load()
#bigram

#logistic model

logistic_model2 = TaskTrain(do_preprocess=True, model='logistic',n_gram_range = (2,2))

d6tflow.run(logistic_model2)

logistic_model2 = logistic_model2.output().load()



#svm model

svm_model2 = TaskTrain(do_preprocess = True, model='svm',n_gram_range = (2,2))

d6tflow.run(svm_model2)

svm_model2 = svm_model2.output().load()



#KNN model

KNN_model2 = TaskTrain(do_preprocess = True, model='KNN',n_gram_range = (2,2))

d6tflow.run(KNN_model2)

KNN_model2 = KNN_model2.output().load()



print('Insample accuracy ')

print('logistic_model2: ', sklearn.metrics.accuracy_score(Y_train,logistic_model2.predict(X_train))) 

print('svm_model2: ', sklearn.metrics.accuracy_score(Y_train,svm_model2.predict(X_train)))

print('KNN_model2: ', sklearn.metrics.accuracy_score(Y_train,KNN_model2.predict(X_train)))
models = {'logistics2' : logistic_model2,

         'svm2' : svm_model2,

         'KNN2' : KNN_model2

         }



print ('model  ','mean_accuracy   ', 'mean_f1   ')

for key, model in models.items():

    scores = cross_validate(model, X_train, Y_train, scoring=('accuracy', 'f1'), cv=5)

    print(f'{key}', scores['test_accuracy'].mean(),scores['test_f1'].mean())