#Importing all the Libraries needed

#Data Preprocessing

from ipywidgets import interact

import unidecode

import pandas as pd

import random

import json

from collections import Counter

from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

import re

#https://pypi.org/project/tqdm/ information on tqdm

from tqdm import tqdm

tqdm.pandas()



#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)

import plotly.offline as offline

import plotly.graph_objs as go

import plotly.express as px



# Data Modeling

from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import FastText, Word2Vec

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, cross_validate, train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn import model_selection 

from nltk.stem import WordNetLemmatizer

import warnings

warnings.filterwarnings('ignore')
def random_colours(number_of_colors):

    '''

    Simple function for random colours generation.

    Input:

        number_of_colors - integer value indicating the number of colours which are going to be generated.

    Output:

        Color in the following format: ['#E86DA4'] .

    '''

    colors = []

    for i in range(number_of_colors):

        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

    return colors
train = pd.read_json("/kaggle/input/whats-cooking-kernels-only/train.json")

test = pd.read_json("/kaggle/input/whats-cooking-kernels-only/test.json")
train.info()
test.info()
train.head()
train['cuisine'].unique()
train['cuisine'].value_counts().plot.bar(color=random_colours(20),figsize=(16,6))
# Taking Out all the ingredients in the dataset and storing in a list

raw_ingredients = [ing for ingredients in train['ingredients'] for ing in ingredients]
print('Maximum Number of Ingredients in a Dish: ',train['ingredients'].str.len().max())

print('Minimum Number of Ingredients in a Dish: ',train['ingredients'].str.len().min())
#no of Ingredients

train['num_ing'] = train['ingredients'].str.len()
plt.figure(figsize=(16,6))

sns.distplot(train['num_ing'],kde =False ,bins=60)
plt.figure(figsize=(16,6))

sns.countplot(x='num_ing',data=train)
longrecip = train[train['num_ing'] > 30]

print(len(longrecip))
longrecip['cuisine'].value_counts()
print(longrecip[longrecip['num_ing'] == 65]['ingredients'].values)

print('Cuisine :-',longrecip[longrecip['num_ing'] == 65]['cuisine'].values)
shortrecip = train[train['num_ing']<=2]

print(len(shortrecip))
shortrecip['cuisine'].value_counts()
train[train['num_ing'] <= 1]
[ingredient for ingredient in raw_ingredients if len(ingredient) <= 2]
plt.figure(figsize=(20,8))

sns.boxplot(x='cuisine',y='num_ing',data=train)
' '.join(sorted([char for char in set(' '.join(raw_ingredients)) if re.findall('[^A-Za-z]', char)]))
list(set([ingredient for ingredient in raw_ingredients if re.findall('[A-Z]+', ingredient)]))[:5]
list(set([ingredient for ingredient in raw_ingredients if '’' in ingredient]))
list(set([ingredient for ingredient in raw_ingredients if re.findall('-', ingredient)]))[:5]
temp_ing = list(set([ingredient for ingredient in raw_ingredients if re.findall('[0-9]', ingredient)]))

temp_ing[:6]
len(temp_ing)
units = ['inch', 'oz', 'lb', 'ounc', '%'] # ounc is a misspelling of ounce?



@interact(unit=units)

def f(unit):

    ingredients_df = pd.DataFrame([ingredient for ingredient in raw_ingredients if unit in ingredient], columns=['ingredient'])

    return ingredients_df.groupby(['ingredient']).size().reset_index(name='count').sort_values(['count'], ascending=False)
keywords = ['american', 'greek', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'thai',

    'vietnamese', 'cajun', 'creole', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian',

]

d ={}

for k in keywords:

    temp = [ingredient for ingredient in raw_ingredients if k in ingredient]

    d[k] = temp
d['american']
top = Counter([item for sublist in train['ingredients'] for item in sublist])

print(top.most_common(20))
temp = pd.DataFrame(top.most_common(20))

temp.columns = ['ingredients','total_count']

plt.figure(figsize=(7,9))

sns.barplot(x='total_count',y='ingredients',data=temp)
labels = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican',

       'spanish', 'italian', 'mexican', 'chinese', 'british', 'thai',

       'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese',

       'irish', 'korean', 'moroccan', 'russian']

templist=[]

for cus in labels:

    lisofing=[]

    for lis in train[train['cuisine'] == cus]['ingredients']:

        for ing in lis:

            lisofing.append(ing)

    templist.append([cus,len(list(set(lisofing)))])

Unique_ing = pd.DataFrame(templist,columns=['cuisine','unique_ing']).sort_values(by='unique_ing',ascending=False)
plt.figure(figsize=(20,8))

sns.barplot(x='cuisine',y='unique_ing',data=Unique_ing)
def cuisine_unique(cuisine,numingr,raw_ingredients):

    '''

    Input:

        cuisine - cuisine category (ex. 'brazilian');

        numingr - how many specific ingredients do you want to see in the final result; 

        allingredients - list  for item in train_data[train_data.cuisine == cuisine]['ingredients']:

    Output: 

        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..



    '''

    allother = []

    for item in train[train.cuisine != cuisine]['ingredients']:

        for ingr in item:

            allother .append(ingr)

    allother  = list(set(allother ))

    

    specificnonly = [x for x in raw_ingredients if x not in allother]

    

    mycounter = Counter()

     

    for item in train[train.cuisine == cuisine]['ingredients']:

        for ingr in item:

            mycounter[ingr] += 1

    keep = list(specificnonly)

    

    for word in list(mycounter):

        if word not in keep:

            del mycounter[word]

    

    cuisinespec = pd.DataFrame(mycounter.most_common(numingr), columns = ['ingredient','count'])

    

    return cuisinespec
cuisinespec= cuisine_unique('mexican', 10, raw_ingredients)

print("The top 10 unique ingredients in Mexican cuisine are:")

cuisinespec
#Removing Outliers Values that were irrevelant to model

train = train[train['num_ing'] > 1]
train = train[train['num_ing']<60]
train.shape
lemmatizer = WordNetLemmatizer()

def preprocess(ingredients):

    ingredients_text = ' '.join(ingredients)

    ingredients_text = ingredients_text.lower() #Lower - Casing

    ingredients_text = ingredients_text.replace('-', ' ') # Removing Hyphen

    words = []

    for word in ingredients_text.split():

        word = re.sub("[0-9]"," ",word) #removing numbers,punctuations and special characters

        word = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', word) # Removing Units

        if len(word) <= 2: continue # Removing words with less than two characters

        word = unidecode.unidecode(word) #Removing accents

        word = lemmatizer.lemmatize(word) #Lemmatize

        if len(word) > 0: words.append(word)

    return ' '.join(words)
#Checking if our function works

for ingredient, expected in [

    ('Eggs', 'egg'),

    ('all-purpose flour', 'all purpose flour'),

    ('purée', 'puree'),

    ('1% low-fat milk', 'low fat milk'),

    ('half & half', 'half half'),

    ('safetida (powder)', 'safetida (powder)')

]:

    actual = preprocess([ingredient])

    assert actual == expected, f'"{expected}" is excpected but got "{actual}"'
train['x'] = train['ingredients'].progress_apply(preprocess)

test['x'] = test['ingredients'].progress_apply(preprocess)

train.head()
def apply_word2vec(sentences):

    vectorizer = Word2Vec(

        sentences,

        size=500,

        window=20,

        min_count=3,

        sg=1,

        iter=20

    )



    def to_vector(sentence):

        words = [word for word in sentence if word in vectorizer.wv.vocab]

        if words:

            return np.mean(vectorizer[words], axis=0)

        else:

            return np.zeros(500)



    return np.array([to_vector(sentence) for sentence in sentences])



def apply_fasttext(sentences):

    vectorizer = FastText(

        size=500,

        window=20,

        min_count=3,

        sg=1,

        iter=20

        )

    vectorizer.build_vocab(sentences)

    vectorizer.train(sentences, total_examples=vectorizer.corpus_count, epochs=vectorizer.iter)



    def to_vector(sentence):

        words = [word for word in sentence if word in vectorizer.wv.vocab]

        if words:

            return np.mean(vectorizer.wv[words], axis=0)

        else:

            return np.zeros(500)



    return np.array([to_vector(sentence) for sentence in sentences])



def train_model(x, y, n_splits=3):

    model = LogisticRegression(C=10, solver='sag', multi_class='multinomial', max_iter=300, n_jobs=-1)

    i = 0

    accuracies = []

    kfold = KFold(n_splits)

    for train_index, test_index in kfold.split(x):

        classifier = LogisticRegression(C=10, solver='sag', multi_class='multinomial', max_iter=300, n_jobs=-1)

        classifier.fit(x[train_index], y[train_index])

        predictions = classifier.predict(x[test_index])

        accuracies.append(accuracy_score(predictions, y[test_index]))

        i += 1

    average_accuracy = sum(accuracies) / len(accuracies)

    return average_accuracy



def run_experiment(preprocessor):

    train = json.load(open('/kaggle/input/whats-cooking-kernels-only/train.json'))



    target = [doc['cuisine'] for doc in train]

    lb = LabelEncoder()

    y = lb.fit_transform(target)



    x = preprocessor.fit_transform(train)



    return train_model(x, y)

import time

results = []

for (name, preprocessor) in [

    ('TfidfVectorizer()', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(),

    )),

    ('TfidfVectorizer(binary=True)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(binary=True),

    )),

    ('TfidfVectorizer(min_df=3)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(min_df=3),

    )),

    ('TfidfVectorizer(min_df=5)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(min_df=5),

    )),

    ('TfidfVectorizer(max_df=0.95)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(max_df=0.95),

    )),

     ('TfidfVectorizer(max_df=0.9)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(max_df=0.9),

    )),

    ('TfidfVectorizer(sublinear_tf=True)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(sublinear_tf=True),

    )),

    ('TfidfVectorizer(strip_accents=unicode)', make_pipeline(

        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),

        TfidfVectorizer(strip_accents='unicode'),

    )),

]:

    start = time.time()

    accuracy = run_experiment(preprocessor)

    execution_time = time.time() - start

    results.append({

        'name': name,

        'accuracy': accuracy,

        'execution time': f'{round(execution_time, 2)}s'

    })

pd.DataFrame(results, columns=['name', 'accuracy', 'execution time']).sort_values(by='accuracy', ascending=False)
vectorizer = TfidfVectorizer(sublinear_tf=True)
X_train = vectorizer.fit_transform(train['x'].values)

X_train.sort_indices()

X_test = vectorizer.transform(test['x'].values)
label_encoder = LabelEncoder()

Y_train = label_encoder.fit_transform(train['cuisine'].values)
classifier = SVC(C=100, # penalty parameter

	 			 kernel='rbf', # kernel type, rbf working fine here

	 			 degree=3, # default value

	 			 gamma=1, # kernel coefficient

	 			 coef0=1, # change to 1 from default value of 0.0

	 			 shrinking=True, # using shrinking heuristics

	 			 tol=0.001, # stopping criterion tolerance 

	      		 probability=False, # no need to enable probability estimates

	      		 cache_size=200, # 200 MB cache size

	      		 class_weight=None, # all classes are treated equally 

	      		 verbose=False, # print the logs 

	      		 max_iter=-1, # no limit, let it run

          		 decision_function_shape=None, # will use one vs rest explicitly 

          		 random_state=None)
model = OneVsRestClassifier(classifier, n_jobs=4)

model.fit(X_train, Y_train)
print ("Predict on test data ... ")

Y_test = model.predict(X_test)

Y_pred = label_encoder.inverse_transform(Y_test)
Y_pred[:20]
test_id = test['id']

sub = pd.DataFrame({'id': test_id, 'cuisine': Y_pred}, columns=['id', 'cuisine'])

sub.to_csv('submission.csv', index=False)
sub.head()