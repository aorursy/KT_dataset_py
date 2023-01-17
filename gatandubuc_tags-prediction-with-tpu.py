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
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.rcParams["figure.figsize"] = (20, 15)

import seaborn as sns

from bs4 import BeautifulSoup
stack_data = pd.read_csv(r'/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')

stack_data
print(stack_data.info())
stack_data_f = stack_data.dropna(subset=['Y'])

stack_data_f.CreationDate = stack_data_f.CreationDate.astype('datetime64[ns]')

stack_data_f['year'] = stack_data_f.CreationDate.dt.year

stack_data_f['month'] = stack_data_f.CreationDate.dt.month

stack_data_f['day'] = stack_data_f.CreationDate.dt.day

stack_data_f.info()
stack_data_f['date_month'] = pd.to_datetime({'month':stack_data_f.CreationDate.dt.month,

                                             'year':stack_data_f.CreationDate.dt.year,

                                             'day':[1 for i in stack_data_f.CreationDate]})



stack_data_gb_d = stack_data_f.groupby(by=stack_data_f.CreationDate.dt.date

                                    ).agg({'CreationDate':lambda x:(~x.isna()).sum(),

                                           'date_month': lambda x: x.iloc[0]})



stack_data_gb_m = stack_data_gb_d.groupby(by=['date_month']).agg([np.mean, np.max, np.min, np.sum])





stack_data_gb_y = stack_data_f.groupby(by=stack_data_f.CreationDate.dt.year

                                    ).agg({'CreationDate':lambda x:(~x.isna()).sum(),

                                           'date_month': lambda x: x.iloc[0]})





fig = plt.figure(figsize = (20, 15))

plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1,

                       right = 0.9, top = 0.9, wspace = 0, hspace = 0.3)

width = 0.35



# Plot day

ax_d = fig.add_subplot(311)



majorLocator = MultipleLocator(10)

majorFormatter = FormatStrFormatter('%d')

minorLocator = MultipleLocator(2)



l1, = ax_d.plot(stack_data_gb_m.index, stack_data_gb_m.iloc[:,1],

            label='Questions per day', alpha=0.5, c='b')

l2, = ax_d.plot(stack_data_gb_m.index, stack_data_gb_m.iloc[:,2],

            label='Questions per day', alpha=0.5, c='b')

l3, = ax_d.plot(stack_data_gb_m.index, stack_data_gb_m.iloc[:,0], c='orange')



ax_d.set_ylabel('Number of Questions per day')

ax_d.set_title('Number of questions asked per month')

ax_d.yaxis.set_major_locator(majorLocator)

ax_d.yaxis.set_major_formatter(majorFormatter)

ax_d.yaxis.set_minor_locator(minorLocator)



plt.fill_between(stack_data_gb_m.index, 

                     stack_data_gb_m.iloc[:,1],

                     stack_data_gb_m.iloc[:,2],

                     alpha=0.2)

    

plt.grid(axis='both', color='0.95')

plt.legend([l1,l3],['min and max number of questions per day', 'mean'])



# Barplot month

ax_m = fig.add_subplot(3,1,2)



sns.barplot(x=stack_data_gb_m.index.date, y=stack_data_gb_m.iloc[:,3], palette="Blues_d", ax=ax_m)

plt.setp(ax_m.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")

ax_m.set_ylabel('Number of Questions')



# Barplot year

ax_y = fig.add_subplot(3,1,3)



sns.barplot(x=stack_data_gb_y.index, y=stack_data_gb_y['CreationDate'], data=stack_data_gb_y, palette="Blues_d")



ax_y.set_xlabel('time')

ax_y.set_ylabel('Number of Questions')



#From matplotlib exemple

def autolabel(rects, ax, width=0.35, xpos='center'):

    """

    Attach a text label above each bar in *rects*, displaying its height.



    *xpos* indicates which side to place the text w.r.t. the center of

    the bar. It can be one of the following {'center', 'right', 'left'}.

    """



    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}

    i=0

    for height in rects:

        ax.annotate('{}'.format(height),

                    xy=(i , height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom', size=8)

        i+=1





autolabel(stack_data_gb_m.iloc[:,3], ax=ax_m)

autolabel(stack_data_gb_y['CreationDate'], ax=ax_y)

print('graphs based on:')

stack_data_gb_d
# Could be better defined

lang_list = ['( |<)C( |>)','( |<)C[+]','( |<)C[#]','objective-c','Java( |>)','SQL','Javascript','Python','Ruby','PHP','HTML','( |<)R( |>)','MATLAB']



Tags = stack_data_f.Tags.str.split('><',expand=True)

Tags = Tags.apply(lambda x: x.str.replace('<|>',''))



list_tags=pd.Series()

for col in Tags:

    list_tags = pd.concat([list_tags, Tags.loc[:, col]])

    

list_tags = list_tags.dropna().reset_index(drop=True) # List of all the tags

print("Total number of tags over 60k topics:", list_tags.shape[0])



# Languages study

stack_data_l = stack_data_f.copy()

for lang in lang_list:

    stack_data_l[lang] = stack_data.Tags.str.contains(lang, regex=True, case=False)

    

stack_data_nb_l = stack_data_l.loc[:,lang_list].sum().sort_values(ascending=False) # Values for each languages

print('Total number of references to a language:', stack_data_nb_l.sum())



# Graph

lgs = stack_data_nb_l.values

ind = stack_data_nb_l.index

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(ind, lgs, width)



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Scores')

ax.set_title('Number of tags by languages between 2016 and 2020 over 60k topics')

ax.set_xticks(ind)

ax.legend()





#From matplotlib exemple

def autolabel(rects, xpos='center'):

    """

    Attach a text label above each bar in *rects*, displaying its height.



    *xpos* indicates which side to place the text w.r.t. the center of

    the bar. It can be one of the following {'center', 'right', 'left'}.

    """



    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}



    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom')





autolabel(rects1)



# Proportion of languages per year

# Same code than the previous point



def pie_l(periodicity, date, lang_list, ax, data=stack_data_l.copy()):

    """Create a pie on languages proportions according to the periodicity chosen and the date"""

    

    fracs = data.loc[data[periodicity] == date,lang_list].sum().sort_values(ascending=False)

    fracs = fracs.apply(lambda x: x*100/fracs.sum())

    labels = fracs.index

    ax.pie(fracs, labels=labels, autopct='%1.1f%%', textprops={'fontsize':10},

                  shadow=True, explode=tuple(0.2 if i==0 

                                              else 0.1 if i==1

                                              else 0.05 if i==2 

                                              else 0 for i,v in enumerate(fracs)))

    ax.set_title('Proportions of each languages in '+str(date))

    return fracs



# Make figure and axes

fig, axs = plt.subplots(3, 2, figsize=(30,30))

x1 = pie_l('year', 2016, lang_list, ax=axs[0,0])

x2 = pie_l('year', 2017, lang_list, ax=axs[0,1])

x3 = pie_l('year', 2018, lang_list, ax=axs[1,0])

x4 = pie_l('year', 2019, lang_list, ax=axs[1,1])

x5 = pie_l('year', 2020, lang_list, ax=axs[2,0])



paper_rc = {'lines.linewidth': 1, 'lines.markersize': 8}                  

sns.set_context("paper", rc = paper_rc)

sns.lineplot(data=pd.DataFrame([x1,x2,x3,x4,x5], index=pd.date_range('2016', periods=5, freq='Y')),

             markers=['s', 'o', 'v', '<', '>','s', 'o', 'v', '<', '>','o', 'v', '<' ], dashes=False, ax=axs[2,1])

axs[2,1].set_title('Languages proportions over time (Same meaning than the kpi)')
fig, ax = plt.subplots(2,3, figsize=(30,20))



# By Classes

HQ = stack_data_l.loc[stack_data_l.Y=='HQ', lang_list].sum()

HQ.loc['other'] = 20000 - HQ.sum()

HQ.sort_values(ascending=False, inplace=True)



LQ = stack_data_l.loc[stack_data_l.Y=='LQ_EDIT', lang_list].sum()

LQ.loc['other'] = 19999 - LQ.sum()

LQ.sort_values(ascending=False, inplace=True)



LQC = stack_data_l.loc[stack_data_l.Y=='LQ_CLOSE', lang_list].sum()

LQC.loc['other'] = 19998 - LQC.sum()

LQC.sort_values(ascending=False, inplace=True)



# By languages

py = stack_data_l.loc[stack_data_l.Python==True, 'Y'].value_counts()

js = stack_data_l.loc[stack_data_l['Java( |>)']==True, 'Y'].value_counts()

j = stack_data_l.loc[stack_data_l.Javascript==True, 'Y'].value_counts()



pie_HQ = ax[0,0].pie(HQ, labels=HQ.index, autopct='%1.1f%%', textprops={'fontsize':10},

                  shadow=True, explode=tuple(0.2 if i==0 

                                              else 0.1 if i==1

                                              else 0.05 if i==2 

                                              else 0 for i,v in enumerate(HQ)))



pie_LQ = ax[0,1].pie(LQ, labels=LQ.index, autopct='%1.1f%%', textprops={'fontsize':10},

                  shadow=True, explode=tuple(0.2 if i==0 

                                              else 0.1 if i==1

                                              else 0.05 if i==2 

                                              else 0 for i,v in enumerate(LQ)))



pie_LQC = ax[0,2].pie(LQC, labels=LQC.index, autopct='%1.1f%%', textprops={'fontsize':10},

                  shadow=True, explode=tuple(0.2 if i==0 

                                              else 0.1 if i==1

                                              else 0.05 if i==2 

                                              else 0 for i,v in enumerate(LQC)))



pie_py = ax[1,0].pie(py, labels=py.index, autopct='%1.1f%%', textprops={'fontsize':10},

                          shadow=True)



pie_js = ax[1,1].pie(js, labels=js.index, autopct='%1.1f%%', textprops={'fontsize':10},

                          shadow=True)



pie_j = ax[1,2].pie(j, labels=j.index, autopct='%1.1f%%', textprops={'fontsize':10},

                          shadow=True)



ax[0,0].set_title('HQ topics')

ax[0,1].set_title('LQ_EDIT topics')

ax[0,2].set_title('LQ_CLOSE topics')

ax[1,0].set_title('Python')

ax[1,1].set_title('JavaScript')

ax[1,2].set_title('Java')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multioutput import MultiOutputClassifier

import xgboost

from sklearn.model_selection import ParameterGrid

import sklearn

import eli5

from eli5.lime import TextExplainer
import tensorflow as tf

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing import text

from sklearn.metrics import classification_report

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import re
stack_data_f['text'] = stack_data_f.Title+': '+stack_data_f.Body



def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text

stack_data_f.text = stack_data_f.text.apply(clean_text)

# Best model
#Split data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(stack_data_f.text.iloc[:50000]

                                                    , stack_data_f.Y.iloc[:50000], test_size=0.3, random_state=0 )





#Try different classifiers

classifiers = [

    LogisticRegression(C=1),

    MultinomialNB(),

    DecisionTreeClassifier(),

    RandomForestClassifier()]



Classifiers_results = pd.Series(name='results')



for cls in classifiers:

    text_clf = Pipeline([

        ('vect', TfidfVectorizer(ngram_range=(1,1))),

        ('clf', cls)])



    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)

    print(str(cls) +': ' + str(text_clf.score(X_test, y_test)))
def GridSearch(cls, parameters, X, y):

    """Try different parameters. Don't use CV because of huge train dataset"""

    

    results = pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    

    for ind,par in enumerate(list(ParameterGrid(parameters))):

        text_clf = Pipeline([

                ('vect', TfidfVectorizer()),

                ('clf', classifier(**par))])

        text_clf.fit(X_train, y_train)

        predicted = text_clf.predict(X_test)

        results.loc[str(par),'results'] = text_clf.score(X_test, y_test)

        results.loc[str(par),'parameters'] = ind

        ind_best = results.sort_values(by=['results'], ascending=False).iloc[0,1]



    return list(ParameterGrid(parameters))[int(ind_best)]



classifier = LogisticRegression

parameters = {

    'solver':['saga'],

    'C': [1, 1.5, 2],

    'penalty': ['l1', 'l2']

 }



results = GridSearch(classifier, parameters, stack_data_f.text.iloc[:5000], stack_data_f.Y.iloc[:5000])

results
classifier = LogisticRegression(**results)



text_clf = Pipeline([

                ('vect', TfidfVectorizer()),

                ('clf', classifier)])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

text_clf.score(X_test, y_test)
te = TextExplainer(random_state=0)

te.fit(stack_data_f.text.iloc[:50000][0], text_clf.predict_proba)

te.show_prediction(target_names= stack_data_f.Y.unique().tolist())
Tags = stack_data_f.Tags.str.split('><',expand=True)

Tags = Tags.apply(lambda x: x.str.replace('<|>',''))



f_tags = Tags[0]

s_tags = Tags[1].fillna(f_tags)

t_tags = Tags[2].fillna(f_tags)



f_stack_data_f = stack_data_f.text

s_stack_data_f = stack_data_f.text

t_stack_data_f = stack_data_f.text



"""

# If you want to do multiouput classifier, join the three previous columns

f_s_tags = pd.DataFrame({'0':f_tags,'1':s_tags})

binarizer = MultiLabelBinarizer()

f_s_tags = binarizer.fit_transform(f_s_tags.values)"""
#Split data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(f_stack_data_f.iloc[:10000]

                                                    , f_tags[:10000], test_size=0.3, random_state=0 ) # f_s_tags[:10000,:]





#Try different classifiers

classifiers = [

    DecisionTreeClassifier(random_state=0),

    RandomForestClassifier(random_state=0)]



for cls in classifiers:

    text_clf = Pipeline([

        ('vect', TfidfVectorizer()),

        ('clf', cls)])



    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)

    print(str(cls) +': ' + str(text_clf.score(X_test, y_test)))
def GridSearch(cls, parameters, X, y):

    """Try different parameters. Don't use CV because of huge train dataset"""

    

    results = pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    

    for ind,par in enumerate(list(ParameterGrid(parameters))):

        text_clf = Pipeline([

                ('vect', TfidfVectorizer()),

                ('clf', classifier(**par))])

        text_clf.fit(X_train, y_train)

        predicted = text_clf.predict(X_test)

        results.loc[str(par),'results'] = text_clf.score(X_test, y_test)

        results.loc[str(par),'parameters'] = ind

        ind_best = results.sort_values(by=['results'], ascending=False).iloc[0,1]

        

    return list(ParameterGrid(parameters))[int(ind_best)]

    

#Try different parameters

classifier = RandomForestClassifier



parameters = {

    'random_state': [0],

    'max_features': [1000, 2000, 3000],

    'n_estimators': [150, 200, 300],

 }



results = GridSearch(classifier, parameters, f_stack_data_f.iloc[:5000], f_tags[:5000])

results
#Split data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(f_stack_data_f.iloc[:10000]

                                                        , f_tags[:10000], test_size=0.3, random_state=0)



classifier = RandomForestClassifier(**results)



text_clf = Pipeline([

            ('vect', TfidfVectorizer()),

            ('clf', classifier)])



text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print('score :' + str(text_clf.score(X_test, y_test)))
text_clf.predict([f_stack_data_f.iloc[32500]])
f_tags[32500]
MAX_FEATURES = 20000

EPOCHS = 20

BATCH_SIZE = 20
stack_data_f['text'] = stack_data_f.Title+': '+stack_data_f.Body



def clean_text(text):

    text = text.lower()

    text = BeautifulSoup(text,'html.parser').text

    text = text.replace('\n', '').replace('\r\n', '').replace('\r', '').replace("\'", '')

    return text

stack_data_f.text = stack_data_f.text.apply(clean_text)
f_s_tags = pd.DataFrame({'0':f_tags,'1':s_tags})

test = f_s_tags.copy()

encoder = LabelEncoder()

encoder.fit(pd.concat([test.iloc[:,0],test.iloc[:,1]],ignore_index=True)) # Transform columns of tag into integers

test.iloc[:,0] = encoder.transform(test.iloc[:,0])

test.iloc[:,1] = encoder.transform(test.iloc[:,1])



df = sklearn.utils.shuffle(pd.DataFrame({'text':stack_data_f.text, 'Tags1':test.iloc[:,0], 'Tags2':test.iloc[:,1]}), random_state=0) # Shuffle 

Y1 = df.Tags1

Y2 = df.Tags2

text_stack = df.text

Y1 =  np_utils.to_categorical(Y1) # Transform integers into binary output. exemple: Let's [1,2,3] be our output, this vector become [[1,0,0],[0,1,0],[0,0,1]]

Y2 =  np_utils.to_categorical(Y2)



X_train = text_stack.values[:50000]

X_test = text_stack.values[50000:55000]

y_train1 = Y1[:50000]

y_test1 = Y1[50000:55000]

y_train2 = Y2[:50000]

y_test2 = Y2[50000:55000]
tokens = text.Tokenizer(num_words=MAX_FEATURES, lower=True)

tokens.fit_on_texts(list(X_train))

X_train_seq = tokens.texts_to_sequences(X_train)

X_test_seq = tokens.texts_to_sequences(X_test)



length = [len(i) for i  in pd.Series(X_train_seq)]

plt.hist(length)

print(np.quantile(length, 0.90))

# 90% of the questions count more than 230 words 

MAX_LEN = 250



X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='pre')

X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='pre')
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    inputs = tf.keras.Input(shape=(None,), dtype="int32")

    x = layers.Embedding(MAX_FEATURES, 256)(inputs)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)

    x = layers.GlobalMaxPooling1D()(x)

    outputs = layers.Dense(4970, activation='softmax')(x)

    outputs2 = layers.Dense(4971, activation='softmax')(x)

    model = tf.keras.Model(inputs, [outputs,outputs2])

    model.summary()

    

    es_cb = EarlyStopping(monitor='val_loss', min_delta=0,  patience=10, verbose=0, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Momentum permit to our model to cross 'mountains and plateau'

    SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) 

    model.compile(loss='categorical_crossentropy', optimizer=SGD ,metrics=[tf.keras.metrics.CategoricalAccuracy()])
#Learning Rate is one of the most important hyperparameter so the following piece of code is a way to find a good LR

import keras

class ExponentialLearningRate(keras.callbacks.Callback):

    

    def __init__(self, K, factor):

        self.factor = factor

        self.rates = []

        self.losses = []

        self.K = K

        

    def on_batch_end(self, batch, logs):

        

        self.rates.append(self.K.get_value(self.model.optimizer.lr))

        self.losses.append(logs["loss"])

        self.K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

        

        

def bestLearningRate():

        

        print("\n\n********************** Best learning rate calculation ******************\n\n")

        K = keras.backend

        model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=[tf.keras.metrics.CategoricalAccuracy()])

        expon_lr = ExponentialLearningRate(K,factor=1.0002)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 15, callbacks=[expon_lr])

        print("*************************************************************************\n\n")

        

        print("********************** Loss as function of learning rate plot displayed ********************\n\n")

        plt.plot(expon_lr.rates, expon_lr.losses)

        plt.gca().set_xscale('log')

        plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))

        plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])

        plt.xlabel("Learning rate")

        plt.ylabel("Loss")

        

bestLearningRate()
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    inputs = tf.keras.Input(shape=(None,), dtype="int32")

    x = layers.Embedding(MAX_FEATURES, 256)(inputs)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)

    x = layers.GlobalMaxPooling1D()(x)

    outputs = layers.Dense(4970, activation='softmax')(x)

    outputs2 = layers.Dense(4971, activation='softmax')(x)

    model = tf.keras.Model(inputs, [outputs,outputs2])

    model.summary()

    

    es_cb = EarlyStopping(monitor='val_loss', min_delta=0,  patience=10, verbose=0, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Momentum permit to our model to cross 'mountains and plateau'

    SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) 

    model.compile(loss='categorical_crossentropy', optimizer=SGD ,metrics=[tf.keras.metrics.CategoricalAccuracy()])



#training 

history = model.fit(X_train, [y_train1, y_train2], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, [y_test1,y_test2]),callbacks = [es_cb, reduce_lr], verbose=1)
def tags_pred(test_question):

    

    print(test_question)

    seq = tokens.texts_to_sequences([test_question])

    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='pre')

    pred = model.predict(padded)



    labels=list(encoder.classes_)

    pred1 = pred[0][0]

    pred2 = pred[1][0]

    for i in range(3): # We get the three most probable tags

        

        print('Tags 1 : ' +str(labels[np.argmax(pred1)]) + ' ' + str(pred1[np.argmax(pred1)]))

        pred1 = np.delete(pred1, np.argmax(pred1), axis=0)

    for i in range(3):



        print('Tags 2 : ' +str(labels[np.argmax(pred2)]) + ' ' + str(pred2[np.argmax(pred2)]))

        pred2 = np.delete(pred2, np.argmax(pred2), axis=0)

    
labels=list(encoder.classes_)

print('Tag 1 : ' + str(np.argmax(Y1[56311]))+ ' ' + str(labels[np.argmax(Y1[56311])]))

print('Tag 2 : ' + str(np.argmax(Y2[56311]))+ ' ' + str(labels[np.argmax(Y2[56311])]))

tags_pred(text_stack.iloc[56311])