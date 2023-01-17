# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
import scipy

import sklearn
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

specs
data = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
ids = specs.event_id.unique()[~np.isin(specs.event_id.unique(), data.event_id.unique())]

print(ids)

specs.loc[specs.event_id.isin(ids), 'info'].values
np.isin(ids, test.event_id.unique())
import json
args = np.vectorize(json.loads)(specs.args.values)
def get_names(args_one_event_id, name_unique):

    for info in args_one_event_id:

#         print(info, info['name'])

        name_unique[info['name']] = info['info']
name_unique = dict()



np.vectorize(lambda x: get_names(x, name_unique=name_unique))(args)

len(name_unique)
name_unique
names = set(['round', 'level','media_type', 'duration', 'dwell_time', 'misses', 'prompt',

         'mode', 'round_number', 'exit_type', 'tutorial_step','time_played',

         'round_prompt', 'target_water_level'])

to_search = pd.Series(['target', 'prompt', 'help', 'media', 'tutorial', 'wrong', 'correct', 'pass'])



print(len(names))
import nltk

from nltk.stem import *
name_unique = pd.Series(name_unique)

name_unique = name_unique.apply(nltk.word_tokenize)

name_unique
lemma =  WordNetLemmatizer()

name_unique = name_unique.apply(lambda x: [lemma.lemmatize(xi, pos='v') for xi in x])
name_unique
res = name_unique.apply( lambda x: to_search.isin(x).any())

names.update(res[res].index)

len(names)
name_unique = dict()



np.vectorize(lambda x: get_names(x, name_unique=name_unique))(args)

for name in names:

    print(name, ':', name_unique[name])
to_drop = ['target_containers', 'target_bucket', 'round_target', 'target_distances'] # just info about target

for name in to_drop:

    names.remove(name)

    

len(names)
def get_event_data(temp):

    args = json.loads(temp)

    return {k:args[k] for k in names if k in args.keys()}
%%time

event_data = pd.DataFrame(data.event_data.apply(get_event_data).tolist())

# event_data.columns = 'event_data_'+event_data.columns 
event_data.nunique()
categorical = event_data.columns[event_data.nunique() < 10]

categorical
for f in categorical:

    display(event_data[f].value_counts(), name_unique[f])
names.remove('exit_type')

len(categorical)
real = event_data.columns[event_data.apply(lambda x: x.nunique()) >= 10]

len(real), real
event_data['level'].plot.hist()
event_data['misses'].plot.hist()
names, len(names)
event_data[names].isna().sum()/ event_data.shape[0]
data_event_id = data.groupby('event_id')[['event_code', 'title', 'type', 'world']].agg([lambda x: x.unique(), lambda x: x.nunique()])

data_event_id
for f in ['event_code', 'title', 'type', 'world']:

    if data_event_id[data_event_id[f]['<lambda_1>'] != 1].shape[0] == 0:

        assert np.isin(data_event_id[f]['<lambda_0>'].unique(), data[f].unique()).all()

    else: display(data_event_id[data_event_id[f]['<lambda_1>'] != 1])

    
specs[specs.event_id == '27253bdc']['info'].values
specs_ini = specs.copy()

# specs = specs[specs.event_id != '27253bdc']
specs = specs_ini.copy()
for index in data_event_id.columns.levels[0]:

    data_event_id.drop(columns=(index, '<lambda_1>'), inplace=True)

    

data_event_id.columns = data_event_id.columns.droplevel(1)

specs = specs.merge(data_event_id.reset_index(), on='event_id', how='outer') 
specs.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_df=0.9, min_df=5)

description = tfidf.fit_transform(specs['info'])

description = pd.DataFrame(description.toarray())

description
num_to_word = {k:v for v,k in tfidf.vocabulary_.items()}

files_num = list(range(description.shape[0]))

num_to_event_id= {k:v for k,v in enumerate(specs.event_id)}
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns

import matplotlib.pyplot as plt
# idx = np.argsort(new.toarray(), axis=1)[:, -10:]

similarity = cosine_similarity(description, description)# dense_output=False)

similarity = np.tril(similarity, k=-1)
threshold = 0.999

n_similar_ids = (similarity >= threshold).sum(axis=0) 

# calculate total numbers of id which have duplicates

(n_similar_ids > 0).sum(), (n_similar_ids > 0).sum()/similarity.shape[0]*100
to_drop = set()

for i in range(similarity.shape[1]):

    add = np.argwhere(similarity[:, i] >= threshold).ravel().tolist()

    print(i, add)    

    to_drop.update(add)

new = description.drop(index=to_drop)

new
new.shape
# check

specs['info'].iloc[[219, 223, 224, 239, 240, 319, 320, 323, 344, 360]].values
# now we want to see distribution of similarity coeffitient

similarity = cosine_similarity(new, new)

similarity = np.tril(similarity, k=-1)



n_similar_ids = (similarity >= 0.99).sum(axis=0) 

assert (n_similar_ids > 0).sum() == 0



idx = np.argwhere(similarity.ravel() != 0)

sns.distplot(similarity.ravel()[idx])