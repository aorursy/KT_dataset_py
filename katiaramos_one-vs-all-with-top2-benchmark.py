# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")

plt.rcParams['figure.figsize'] = 16, 12

import pandas as pd

pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_colwidth', 300)

pd.options.display.float_format = '{:,.6f}'.format



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from functools import reduce



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, accuracy_score







import pickle
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_target_train.csv')

print('df_target_train:', df_target_train.shape)



df_sample_submit = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_sample_submit.csv')

print('df_sample_submit:', df_sample_submit.shape)



df_tracks = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_tracks.csv')

print('df_tracks:', df_tracks.shape)



df_genres = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_genres.csv')

print('df_genres:', df_genres.shape)



df_features = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_features.csv')

print('df_features:', df_features.shape)
from tqdm.notebook import tqdm

from collections import defaultdict



# extract tracks for each genre

genre2tracks = defaultdict(list)

for _, row in tqdm(df_target_train.iterrows(), total=df_target_train.shape[0]):

    for g_id in row['track:genres'].split(' '):

        genre2tracks[int(g_id)].append(row['track_id'])    
df_tmp = pd.DataFrame(

    [(k, len(v)) for (k, v) in genre2tracks.items()], 

    columns=['genre', 'n_tracks']

).sort_values(

    ['n_tracks'], 

    ascending=False

)



sns.barplot(

    data=df_tmp, 

    x='genre',

    y='n_tracks',

    order=df_tmp['genre']

)

plt.title('Disribution of tracks per genre')

plt.show()
df_features.set_index(['track_id'], inplace=True)

df_target_train.set_index(['track_id'], inplace=True)
from collections import defaultdict

r = defaultdict(int)

for _, row in df_target_train.iterrows():

    for x in row['track:genres'].split(' '):

        r[int(x)] += 1

        

labels = list(sorted(r.keys()))
labels
# for each class we build a linear model for binary classification

# we skip all classes if it has less then 1000 positive samples



val_ratio = 0.2

c_grid = [0.01, 0.1, 1]

models = {}



for g_id, positive_samples in tqdm(genre2tracks.items()):

    if len(positive_samples) < 1000:

        continue

    # construct train/val using negative sampling

    negative_samples = list(set(reduce(lambda a, b: a + b, [v for (k, v) in genre2tracks.items() if k != g_id])).difference(positive_samples))

    negative_samples = np.random.choice(

        negative_samples,

        size=len(positive_samples),

        replace=len(negative_samples) < len(positive_samples)

    )

    

    train_positive_samples = np.random.choice(

        positive_samples,

        size=int((1 - val_ratio)*len(positive_samples)),

        replace=False

    )

    val_positive_samples = list(set(positive_samples).difference(train_positive_samples))

    

    train_negative_samples = np.random.choice(

        negative_samples,

        size=int((1 - val_ratio)*len(negative_samples)),

        replace=False

    )

    val_negative_samples = list(set(negative_samples).difference(train_negative_samples))

    

    

    # train a models and pick one

    models[g_id] = {

        'acc': -1

    }

    for c in c_grid:

        model = LogisticRegression(

            penalty='l2',         

            C=c, 

            fit_intercept=True, 

            intercept_scaling=1,

            max_iter=100,

            n_jobs=-1

        )

        model = model.fit(

            df_features.loc[np.hstack([train_positive_samples, train_negative_samples])],

            [1.0]*len(train_positive_samples) + [0.0]*len(train_negative_samples)

        )

        p_val = model.predict_proba(df_features.loc[np.hstack([val_positive_samples, val_negative_samples])])[:, 1]

        y_val = [1.0]*len(val_positive_samples) + [0.0]*len(val_negative_samples)

        

        # choose threshold

        best_t = -1

        best_acc = -1

        for t in np.linspace(0.01, 0.99, 99):

            acc = accuracy_score(y_val, (p_val >= t).astype(np.float))

            if acc > best_acc:

                best_acc = acc

                best_t = t

        

        if models[g_id]['acc'] < best_acc:

            models[g_id]['acc'] = best_acc

            models[g_id]['t'] = best_t

            models[g_id]['model'] = model

            models[g_id]['c'] = c
len(models)
with open('./models.pkl', 'wb') as f:

    pickle.dump(models, f)
# correct test preditions to equalize median number of genres per track



def get_test(k=1.0):

    g_prediction = {}

    for g_id, d in tqdm(models.items()):

        p = d['model'].predict_proba(df_features.loc[df_sample_submit['track_id'].values])[:, 1]

        g_prediction[g_id] = df_sample_submit['track_id'].values[p > k*d['t']]



    track2genres = defaultdict(list)

    for g_id, tracks in g_prediction.items():

        for t_id in tracks:

            track2genres[t_id].append(g_id)

            

    return track2genres



track2genres = get_test(k=1.0)
# median number of genres per track in test

np.median([len(v) for v in track2genres.values()])
# median number of genres per track in train

z = df_target_train['track:genres'].apply(lambda s: len([int(x) for x in s.split(' ')])).median()



z
for k in np.linspace(1, 2, 11):

    track2genres = get_test(k=k)

    print(k, np.median([len(v) for v in track2genres.values()]))
track2genres = get_test(k=1.55)
df_sample_submit['track:genres'] = df_sample_submit.apply(lambda r: ' '.join([str(x) for x in track2genres[r['track_id']]]), axis=1)
# add top-2

df_sample_submit['track:genres'] = df_sample_submit['track:genres'].apply(lambda r: r + ' 15 38' if len(r) > 0 else '15 38')
df_sample_submit
df_sample_submit.to_csv('./submit.csv', index=False)
!head ./submit.csv