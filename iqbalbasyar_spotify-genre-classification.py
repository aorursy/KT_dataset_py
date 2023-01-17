import pandas as pd

data = pd.read_csv("../input/ultimate-spotify-tracks-db/SpotifyFeatures.csv")

data.head()
data.info()
data.describe()
data.head()
unused_col = ['artist_name', 'track_name', 'track_id']

df = data.drop(columns=unused_col).reset_index(drop=True)

df.head()
df.select_dtypes(exclude='number').head()
df['time_signature'].unique().tolist()
df['mode'].unique().tolist()
df['key'].unique().tolist()
mode_dict = {'Major' : 1, 'Minor' : 0}

key_dict = {'C' : 1, 'C#' : 2, 'D' : 3, 'D#' : 4, 'E' : 5, 'F' : 6, 

        'F#' : 7, 'G' : 9, 'G#' : 10, 'A' : 11, 'A#' : 12, 'B' : 12}



df['time_signature'] = df['time_signature'].apply(lambda x : int(x[0]))

df['mode'].replace(mode_dict, inplace=True)

df['key'] = df['key'].replace(key_dict).astype(int)



df.head()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



def corr_heatmap(data):

    corr_matrix = data.corr()

    mask = np.zeros_like(corr_matrix, dtype=np.bool)

    mask[np.triu_indices_from(mask)]= True

    f, ax = plt.subplots(figsize=(11, 15)) 

    heatmap = sns.heatmap(corr_matrix, 

                          mask = mask,

                          square = True,

                          linewidths = .5,

                          cmap = 'coolwarm',

                          cbar_kws = {'shrink': .4, 

                                    'ticks' : [-1, -.5, 0, 0.5, 1]},

                          vmin = -1, 

                          vmax = 1,

                          annot = True,

                          annot_kws = {'size': 12})#add the column names as labels

    ax.set_yticklabels(corr_matrix.columns, rotation = 0)

    ax.set_xticklabels(corr_matrix.columns)

    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
corr_heatmap(data)
df.isna().sum().sum()
duplicated_all = data[data.duplicated(subset = 'track_id', keep=False)]

duplicated = data[data.duplicated(subset = 'track_id', keep='first')]
data[data['track_id'] == duplicated['track_id'].iloc[0]]
print(f'''Unique Duplicates: {duplicated.shape[0]}

Total Duplicates: {duplicated_all.shape[0]}

Total Data: {data.shape[0]}

Duplicates %: {round(duplicated_all.shape[0]/data.shape[0]*100, 2)}''')
data['genre'].value_counts()/len(data)
sns.countplot(y="genre", data=data, color='green')
from sklearn.model_selection import train_test_split

import time
X = df.drop(columns=['genre'])

y = df['genre']

random_state = 11

test_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
y_train.value_counts().sort_index()
y_valid.value_counts().sort_index()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=30, random_state=42)

rfc.fit(X_train, y_train)
y_rfc = rfc.predict(X_valid)
print(classification_report(y_valid, y_rfc))
from sklearn.feature_selection import RFE

selector = RFE(model, n_features_to_select=1)

selector.fit(X_train, y_train)
print(f"Model's Feature Importance")

for i in range(len(selector.ranking_)):

    print(f"#{i+1}: {X.columns[selector.ranking_[i]-1]} ")
def rebalance(data):

    from tqdm import tqdm

    # remove `a capella class`

    data = data[data['genre'] != 'A Capella']

    # set maximum occurence of data

    max_val = 5400 

    # create new dataframe 

    _data = pd.DataFrame(columns=data.columns)

    

    # iteratively add sample of songs based on genre 

    for genre in tqdm(data['genre'].unique()):

        _data = _data.append(data[data['genre'] == genre].sample(n=max_val, random_state=1), ignore_index=True, sort=False)  

    return _data
balanced_df = rebalance(data).drop(columns=unused_col).reset_index(drop=True)

balanced_df['duration_ms'] = balanced_df['duration_ms'].astype(int)

balanced_df['time_signature'] = balanced_df['time_signature'].apply(lambda x : int(x[0]))

balanced_df['mode'].replace(mode_dict, inplace=True)

balanced_df['key'] = balanced_df['key'].replace(key_dict).astype(int)

balanced_df['popularity'] = balanced_df['popularity'].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(balanced_df.drop(columns='genre'),balanced_df['genre'], test_size=0.2, random_state=1)
lrm_rebalance = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', max_iter=200).fit(X_train, y_train)
print(classification_report(y_valid, lrm_rebalance.predict(X_valid)))
corr_heatmap(balanced_df)
corr_heatmap(balanced_df.drop(columns=['energy', 'loudness']))
X_train, X_valid, y_train, y_valid = train_test_split(balanced_df.drop(columns=['genre', 'energy', 'loudness']),balanced_df['genre'], test_size=0.2, random_state=1)

lrm_cor = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', max_iter=200).fit(X_train, y_train)
print(classification_report(y_valid, lrm_cor.predict(X_valid)))
X_train, X_valid, y_train, y_valid = train_test_split(balanced_df.drop(columns=['genre', 'loudness', 'danceability', 'acousticness', 'valence', 'energy', 'duration_ms']),balanced_df['genre'], test_size=0.2, random_state=1)

lrm_uf = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', max_iter=200).fit(X_train, y_train)
print(classification_report(y_valid, lrm_uf.predict(X_valid)))
rfc = RandomForestClassifier(n_estimators=30, random_state=42)

print(classification_report(y_valid, rfc.fit(X_train, y_train).predict(X_valid)))
preds = lrm_uf.predict_proba(X_valid)

preds[0]
preds[0].argmax()
lrm_uf.classes_[preds[0].argmax()]
print(f"""Predicted Class\t: {lrm_cor.classes_[preds[0].argmax()]}

Ground Truth\t: {y_valid.values[0]}

Correctness\t: {lrm_cor.classes_[preds[0].argmax()] == y_valid.values[0]}

""")
preds[0].argsort()[::-1][:5]
lrm_uf.classes_[preds[0].argsort()[::-1][:5]]
print(f"""Predicted Class\t: {lrm_cor.classes_[preds[0].argsort()[::-1][:5]]}

Ground Truth\t: {y_valid.values[0]}

Correctness\t: {y_valid.values[0] in lrm_cor.classes_[preds[0].argsort()[::-1][:5]] }

""")
preds_proba = lrm_uf.predict_proba(X_valid)
def top5score(preds, ground_truth, model):

    if not len(preds) == len(ground_truth):

        raise exception('Shape Mismatch')

    

    mdfd_pred = []

    for i in range(len(preds)):

        preds_classes = model.classes_[preds[i].argsort()[::-1][:5]]

        if ground_truth[i] in preds_classes :

            mdfd_pred.append(ground_truth[i])

        else:

            mdfd_pred.append(preds_classes[0])

    return mdfd_pred
preds_proba[0]
mdfd_pred = top5score(preds_proba, y_valid.values, lrm_uf)
print(classification_report(y_valid,mdfd_pred ))