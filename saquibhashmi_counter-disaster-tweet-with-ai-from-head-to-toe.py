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
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import collections

from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import f1_score, log_loss

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import nltk
path = '/kaggle/input/nlp-getting-started'
train_data = pd.read_csv(os.path.join(path, 'train.csv'))

test_data = pd.read_csv(os.path.join(path,'test.csv'))

submission_data = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
train_data.head()
test_data.head()
submission_data.head()
train_data.isnull().sum()
train_data.shape
test_data.isnull().sum()
test_data.shape
del train_data['location']

del test_data['location']
train_data['target'].value_counts()
sns.distplot(train_data['target'].value_counts().values, color='y')
train_data.head()
train_data.shape
train_text_lens = [len(text) for text in train_data['text'].values]

test_text_lens = [len(text) for text in test_data['text'].values]

train_keyword_lens = [len(text) for text in train_data[~train_data.keyword.isnull()]['keyword'].values]

test_keyword_lens = [len(text) for text in test_data[~test_data.keyword.isnull()]['keyword'].values]



train_text_no_word_lens = [len(text.split(' ')) for text in train_data['text'].values]

test_text_no_word_lens = [len(text.split(' ')) for text in test_data['text'].values]

train_keyword_no_word_lens = [len(text.split(' ')) for text in train_data[~train_data.keyword.isnull()]['keyword'].values]

test_keyword_no_word_lens = [len(text.split(' ')) for text in test_data[~test_data.keyword.isnull()]['keyword'].values]
sns.distplot(train_text_lens)

plt.show()

sns.boxplot(train_text_lens)
sns.distplot(train_keyword_lens)

plt.show()

sns.boxplot(train_keyword_lens)
sns.distplot(train_text_no_word_lens)

plt.show()

sns.boxplot(train_text_no_word_lens)
sns.distplot(train_keyword_no_word_lens)

plt.show()

sns.boxplot(train_keyword_no_word_lens)
sns.distplot(test_text_lens,color='y')

plt.show()

sns.boxplot(test_text_lens, color='y')
sns.distplot(test_keyword_lens)

plt.show()

sns.boxplot(test_keyword_lens)
sns.distplot(test_text_no_word_lens)

plt.show()

sns.boxplot(test_text_no_word_lens)
sns.distplot(test_keyword_no_word_lens)

plt.show()

sns.boxplot(test_keyword_no_word_lens)
contains_keyword = [len(set(a.split(' ')).intersection(b.split(' '))) for a, b in zip(train_data[~train_data.keyword.isnull()].keyword, train_data[~train_data.keyword.isnull()].text)]
collections.Counter(contains_keyword)
target = train_data[~train_data.keyword.isnull()].target.values
collections.Counter(target)
collections.Counter(target & contains_keyword)
corr = np.corrcoef(target, contains_keyword)
sns.heatmap(corr)
del train_data['keyword']

del test_data['keyword']
train_text_no_word_unique_lens = [len(set(text.split(' '))) for text in train_data['text'].values]

test_text_no_word_unique_lens = [len(set(text.split(' '))) for text in test_data['text'].values]
sns.distplot(train_text_no_word_unique_lens)

plt.show()

sns.boxplot(test_text_no_word_unique_lens)
all_text_train = ' '.join(train_data['text'].values)

all_text_test = ' '.join(test_data['text'].values)
stopwords = set(STOPWORDS)
wordcloud_train = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(all_text_train)



wordcloud_test = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(all_text_test)
plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud_train)

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('Train Word Cloud')

plt.show()

plt.imshow(wordcloud_test)

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('Test Word Cloud')

plt.show()
stop_words = nltk.corpus.stopwords.words('english')

lemmitizer = nltk.stem.WordNetLemmatizer()
def initial_preprocessing(text):

    word_list = []

    for w in text.split(' '):

        wl = lemmitizer.lemmatize(w)

        if w not in stop_words:

            word_list.append(wl)

    text = ' '.join(word_list)

    

    return text
train_data.text = train_data.text.apply(initial_preprocessing)

test_data.text = test_data.text.apply(initial_preprocessing)
xtrain, xvalid, ytrain, yvalid = train_test_split(train_data.text.values, train_data.target,

                                                 stratify=train_data.target, 

                                                 random_state=42, test_size=0.2, shuffle=True)
xcvalid, xtest, ycvalid, ytest = train_test_split(xvalid, yvalid,

                                                 stratify=yvalid, 

                                                 random_state=42, test_size=0.2, shuffle=True)
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), stop_words = 'english')



tfv.fit(list(xtrain)+list(xvalid))

xtrain_tfv = tfv.transform(xtrain)

xcvalid_tfv = tfv.transform(xcvalid)

xtest_tfv = tfv.transform(xtest)
ctv.fit(list(xtrain)+list(xvalid))

xtrain_ctv = ctv.transform(xtrain)

xcvalid_ctv = ctv.transform(xcvalid)

xtest_ctv = ctv.transform(xtest)
xtrain_tfv.shape, xtrain_ctv.shape
svd = TruncatedSVD(n_components=2500)

svd.fit(xtrain_tfv)
svd.explained_variance_ratio_.sum()
xtrain_tfv_svd = svd.transform(xtrain_tfv)

xcvalid_tfv_svd = svd.transform(xcvalid_tfv)

xtest_tfv_svd = svd.transform(xtest_tfv)
stdscl = StandardScaler()

stdscl_nmean = StandardScaler(with_mean=False)

mmscl = MaxAbsScaler()
features = {}

scalers = {}

features['tfv'] = [xtrain_tfv, xcvalid_tfv, xtest_tfv]

features['ctv'] = [xtrain_ctv, xcvalid_ctv, xtest_ctv]

features['tfv_svd'] = [xtrain_tfv_svd, xcvalid_tfv_svd, xtest_tfv_svd]



# scalers['std'] = stdscl

scalers['std_nmean'] = stdscl_nmean

scalers['mm'] = mmscl
models = {}



# Logistic Regressions

models['lr'] = {"clf" : LogisticRegression(C=1.0), "scale": False}



# Logistic Regression with regularization

models['lr_reg'] = {"clf" : LogisticRegression(C=0.7), "scale": True}



# SVM with RBF

models['svm_rbf'] = {"clf" : SVC(C=0.7, kernel='rbf'), "scale": True}



# SVM with Polynomial

models['svm_poly'] = {"clf" : SVC(C=0.7, kernel='poly'), "scale": True}



# SVM with Sigmoid

models['svm_sigmoid'] = {"clf" : SVC(C=0.7, kernel='sigmoid'), "scale": True}



# SVM with Sigmoid

models['multi_nomial'] = {"clf" : MultinomialNB(), "scale": False, "non_negative": True}



# RandomForest with gini

models['rforest_gini'] = {"clf" : RandomForestClassifier(criterion='gini', n_estimators=200), "scale": False}



# RandomForest with entropy

models['rforest_entropy'] = {"clf" : RandomForestClassifier(criterion='entropy', n_estimators=200), "scale": False}



# Gradient boosting

models['gradient_boost'] = {"clf" : GradientBoostingClassifier(n_estimators=200), "scale": False}
class Trainer:

    def __init__(self, models, features, **params):

        self.models = models

        self.features = features

        self.scalers = params.get('scalers')

        self.pipelines = {}

        self.training_results = {}

    

    def createPipelines(self):

        for model_name, model in self.models.items():

            model_component = (model_name, model['clf'])

            if model.get('non_negative', False) == False:

                for scaler_name, scaler in self.scalers.items():

                    scaler_component = (scaler_name, scaler)

                    self.pipelines[model_name+' '+scaler_name] = [scaler_component, model_component]

            if model['scale'] == False:

                self.pipelines[model_name] = [(model_name, model['clf'])]

        

        return None

                

    def train(self):

        self.createPipelines()

        for pipeline_name, pipeline in self.pipelines.items():

            print("Started training for Pipeline: ", pipeline_name)

            for feature_name, feature in self.features.items():

                if 'nomial' in pipeline_name and 'svd' in feature_name:

                    continue

                print("With %s features"%(feature_name))

                pipeline_clf = Pipeline(pipeline)

                pipeline_clf.fit(feature[0], ytrain)

                prediction_cvalid = pipeline_clf.predict(feature[1])

                prediction_test = pipeline_clf.predict(feature[2])

                f1score_cvalid = f1_score(prediction_cvalid, ycvalid)

                f1score_test = f1_score(prediction_test, ytest)

                print('F1score on cvalid and test set are: %f, %f'%(f1score_cvalid,f1score_test))

                self.training_results[pipeline_name+'/'+feature_name] = {'cvalid':f1score_cvalid,

                                                                         'test':f1score_test

                                                                        }

        

        return None

        
trainer_obj = Trainer(models = models, features=features, scalers=scalers)
trainer_obj.train()
training_results = trainer_obj.training_results
result_df = pd.DataFrame(training_results).T
result_df['Models'] = result_df.index
result_df
result_df = result_df.set_index('Models')
stacked_df = result_df.stack().reset_index()
stacked_df.columns = ['Models', 'Set', 'F1Score']
stacked_df
sns.barplot(y='Models', x='F1Score', hue='Set', data=stacked_df)
clf = GradientBoostingClassifier(n_estimators=200)

clf.fit(xtrain_tfv, ytrain)
prediction_cvalid = clf.predict(xcvalid_tfv)

prediction_test = clf.predict(xtest_tfv)

f1score_cvalid = f1_score(prediction_cvalid, ycvalid)

f1score_test = f1_score(prediction_test, ytest)

print('F1score on cvalid and test set are: %f, %f'%(f1score_cvalid,f1score_test))
test_tfv = tfv.transform(test_data.text)
predictions = clf.predict(test_tfv)
submission_data['id'] = test_data['id']

submission_data['target'] = predictions
submission_data.head()
submission_data.to_csv('output_submissions.csv', index=False)