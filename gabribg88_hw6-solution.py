import numpy as np 
import pandas as pd 
pd.set_option('display.max_rows', 1000)
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
RANDOM_STATE = 42
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_target_train.csv')
print('target_train:', df_target_train.shape)
df_sample_submit = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_sample_submit.csv')
print('sample_submit:', df_sample_submit.shape)
df_tracks = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_tracks.csv')
print('tracks:', df_tracks.shape)
df_genres = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_genres.csv')
print('genres:', df_genres.shape)
df_features = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_features.csv')
print('features:', df_features.shape)
train_idx = df_target_train.track_id.values
test_idx = df_sample_submit.track_id.values
comb = pd.concat([df_target_train, df_sample_submit], ignore_index=True)
comb = pd.merge(comb, df_tracks, on='track_id', how='inner')
comb.set_index('track_id', inplace=True, drop=True)
#df_features.set_index('track_id', drop=True, inplace=True)
# pca = decomposition.PCA().fit(df_features.values)
# plt.figure(figsize=(10,7))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2, marker='.')
# plt.xlabel('Number of components')
# plt.ylabel('Total explained variance')
# plt.xlim(0, 63)
# plt.yticks(np.arange(0, 1.1, 0.1))
# #plt.axvline(21, c='b')
# plt.axhline(0.9, c='r')
# plt.show();
df_features.set_index('track_id', drop=True, inplace=True)
df_features_idx = df_features.index
pca = decomposition.PCA(n_components=0.99, random_state=RANDOM_STATE)
X_features_pca = pca.fit_transform(df_features)
df_features_pca = pd.DataFrame(X_features_pca, index=df_features_idx, columns=[f'x{i}' for i in range(X_features_pca.shape[1])])
comb = pd.merge(comb, df_features_pca, on='track_id', how='inner')
features = comb.columns.tolist() 
date_cols = ['album:date_created', 'album:date_released', 'artist:active_year_begin', 'artist:active_year_end', 'track:date_created', 'track:date_recorded']
numericals = [c for c in features if comb[c].dtype in [int, float]]
categoricals = [c for c in features if comb[c].dtype==object and c not in date_cols and c!='track:genres']
target = 'track:genres'
for date_col in date_cols:
    comb[date_col] = pd.to_datetime(comb[date_col])
    comb[date_col + '_year'] = comb[date_col].dt.year.fillna(0).astype(str) 
    comb[date_col + '_month'] = comb[date_col].dt.month.fillna(0).astype(str) 
    comb[date_col + '_week'] = comb[date_col].dt.week.fillna(0).astype(str) 
    comb[date_col + '_day'] = comb[date_col].dt.day.fillna(0).astype(str) 
    #train[date_col + '_weekofyear'] = train[date_col].dt.weekofyear # same as week
    comb[date_col + '_dayofyear'] = comb[date_col].dt.dayofyear.fillna(0).astype(str) 
    comb[date_col + '_weekday'] = comb[date_col].dt.weekday.fillna(0).astype(str) 
    comb[date_col + '_quarter'] = comb[date_col].dt.quarter.fillna(0).astype(str) 
    comb[date_col + '_is_month_start'] = comb[date_col].dt.is_month_start.fillna(0).astype(str) 
    comb[date_col + '_is_month_end'] = comb[date_col].dt.is_month_end.fillna(0).astype(str) 
    comb[date_col + '_is_quarter_start'] = comb[date_col].dt.is_quarter_start.fillna(0).astype(str) 
    comb[date_col + '_is_quarter_end'] = comb[date_col].dt.is_quarter_end.fillna(0).astype(str) 
    comb[date_col + '_is_year_start'] = comb[date_col].dt.is_year_start.fillna(0).astype(str) 
    comb[date_col + '_is_year_end'] = comb[date_col].dt.is_year_end.fillna(0).astype(str) 
    comb.drop(date_col, axis=1, inplace=True)    

categoricals += comb.columns[-13*len(date_cols):].tolist()
features = numericals + categoricals
labels = comb.loc[train_idx, target].apply(lambda x: x.split(' ')).apply(lambda x: [int(i) for i in x]).explode().to_frame().groupby('track:genres')\
             .size().to_frame().rename({0:'sizes'}, axis=1)
labels
from collections import defaultdict
r = defaultdict(int)
for _, row in df_target_train.iterrows():
    for x in row['track:genres'].split(' '):
        r[int(x)] += 1
        
labels_ = list(sorted(r.keys()))
sizes_ = list(sorted(r.values()))
assert(labels.index.values==labels_).all()
assert(labels.sort_values('sizes').sizes.values==sizes_).all()
track_labels = comb.loc[train_idx, target].apply(lambda x: x.split(' ')).apply(lambda x: [int(i) for i in x]).to_frame().explode('track:genres')\
                   .reset_index().set_index('track:genres')
track_labels
comb[categoricals] = comb[categoricals].replace(np.nan, None)
train = comb.loc[train_idx, features].copy()
test = comb.loc[test_idx, features].copy()
labels.sort_values('sizes', ascending=False, inplace=True)
labels
models = {}

for label, row in labels.iterrows():
    size = row.sizes
    if (size > 1000) : # FIRST MODEL IS OBTAINED WITH LABELS WITH MORE THAN 1000 APPAREANCES
    #if (size <= 1000) & (size>250): # SECOND MODEL IS OBTAINED WITH LABELS WITH LESS THAN 1000 APPAREANCES AND MORE THEN 250 APPAREANCES
        print('Predico label: ' + str(label) + ' di size: '+ str(size))

        positive_idx = track_labels.loc[label].track_id.values
        train_pos = train.loc[positive_idx].copy()
        negative_idx = train.index.difference(train_pos.index)
        train_neg = train.loc[negative_idx].copy()

        train_pos['target'] = 1
        train_neg['target'] = 0

        train_temp = pd.concat([train_neg,train_pos]).sort_index()
        assert(train.sort_index().equals(train_temp[features]))

        negative_size = train_temp.groupby('target').size().loc[0]
        positive_size = train_temp.groupby('target').size().loc[1]

        #negative_weight = round(1 - negative_size/(negative_size+positive_size),2)
        #positive_weight = round(1 - positive_size/(negative_size+positive_size),2)
        negative_weight = 1
        positive_weight = round(negative_size/positive_size,2)

        cat_dict = {c:train_temp.columns.get_loc(c) for c in train_temp[categoricals].columns}

        X_train, X_validation, y_train, y_validation = train_test_split(train_temp[features], train_temp['target'], train_size=0.7, stratify=train_temp['target'], 
                                                                        random_state=RANDOM_STATE)

        models[label] = {'acc': -1}

        model = CatBoostClassifier(iterations=200, random_seed=RANDOM_STATE, eval_metric='F1', class_weights=[negative_weight, positive_weight])
        model.fit(X_train, y_train, cat_features=list(cat_dict.values()), eval_set=(X_validation, y_validation), verbose=False, plot=False)
        #print('F1 score validation; ', f1_score(y_validation, model.predict(data=X_validation)))
        #print('F1 score train; ', f1_score(y_train, model.predict(data=X_train)))
        preds_validation = model.predict_proba(data=X_validation)[:,1]

        best_t = -1
        best_f1 = -1
        for t in np.linspace(0.01, 0.99, 99):
            f1 = f1_score(y_validation, (preds_validation >= t).astype(np.float))
            #print(t,f1)
            if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
        
        print('Best F1 score: ', round(best_f1,2))
        
        if models[label]['acc'] < best_f1:
                models[label]['acc'] = best_f1
                models[label]['t'] = best_t
                models[label]['model'] = model
dsfdsfdrs
# import pickle

# with open('./models2.pkl', 'wb') as f:
#     pickle.dump(models, f)
with open('/kaggle/input/models-hw6/models.pkl', 'rb') as f:
    models1 = pickle.load(f)
with open('/kaggle/input/models-hw6/models2.pkl', 'rb') as f:
    models2 = pickle.load(f)
models = {**models1, **models2}
models
from tqdm import notebook

def get_test(k=1.0):
    g_prediction = {}
    for g_id, d in notebook.tqdm(models.items()):
        #p = d['model'].predict_proba(df_features.loc[df_sample_submit['track_id'].values])[:, 1]
        p = d['model'].predict_proba(data=test[features])[:, 1]
        g_prediction[g_id] = df_sample_submit['track_id'].values[p > k*d['t']]

    track2genres = defaultdict(list)
    for g_id, tracks in g_prediction.items():
        for t_id in tracks:
            track2genres[t_id].append(g_id)
            
    return track2genres

track2genres = get_test(k=0.6)
np.median([len(v) for v in track2genres.values()])
# for k in np.linspace(1, 2, 11):
#     track2genres = get_test(k=k)
#     print(k, np.median([len(v) for v in track2genres.values()]))
df_sample_submit['track:genres'] = df_sample_submit.apply(lambda r: ' '.join([str(x) for x in track2genres[r['track_id']]]), axis=1)
df_sample_submit.groupby(['track:genres']).size()
missing_idx = df_sample_submit[df_sample_submit['track:genres']==''].index
df_sample_submit.loc[missing_idx, 'track:genres'] = '15 38'
df_sample_submit.groupby(['track:genres']).size().sort_values(ascending=False)
df_sample_submit.to_csv('./submit_06.csv', index=False)
!head ./submit_06.csv
