import json, datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
%matplotlib inline
with open('../input/history-of-hearthstone/data.json') as file:
    data = json.load(file)
with open('../input/history-of-hearthstone/refs.json') as file:
    refs = json.load(file)
# to dataframe
decks = pd.DataFrame(data)
# transform date strings to datetime objects
decks['date'] = pd.to_datetime(decks['date'])
# reformat the card column
card_col = ['card_{}'.format(str(i)) for i in range(30)]
cards = pd.DataFrame([c for c in decks['cards']], columns=card_col)
cards = cards.apply(np.sort, axis=1)
decks = pd.concat([decks, cards], axis=1)
decks = decks.drop('cards', axis=1)
# remove tabs and newlines from user names
decks['user'] = decks['user'].apply(str.strip)
# delete unnecessary variables
del(data)
del(cards)
raw_length = len(decks)
print ('Number of decks :', raw_length)
print ('First deck :', min(decks['date']))
print ('Last deck :', max(decks['date']))
release_date = datetime(2014, 3, 11)
decks = decks[decks['date'] > release_date]
# now let's check the dates
print ('First deck :', min(decks['date']))
print ('Last deck :', max(decks['date']))
prc_left = round(len(decks) / raw_length * 100)
print ('Decks removed :', raw_length - len(decks))
print ('Original dataset left :', prc_left, '%')
assert len(decks['deck_id'].unique()) == len(decks)
# decks = decks[decks['deck_format'] == 'W']
decks.loc[(decks['deck_archetype'] == 'Edit'), 'deck_archetype'] = 'Unknown'
# duplicates check
assert len(decks['deck_id'].unique()) == len(decks)
decks['deck_type'].value_counts()
# get the none and ranked decks
none_decks = decks[decks['deck_type'] == 'None']
ranked_decks = decks[decks['deck_type'].isin(['Ranked Deck', 'Tournament'])]
# looks for none ids with cards already in ranked
none_ids = pd.merge(none_decks, ranked_decks, on=card_col, how='inner')['deck_id_x']
# add the none_ids decks to ranked
none_could_be_ranked = none_decks[none_decks['deck_id'].isin(none_ids)]
ranked_decks = pd.concat([none_could_be_ranked, ranked_decks])
# the same for theorycraft decks
theory_decks = decks[decks['deck_type'] == 'Theorycraft']
theory_decks_ids = pd.merge(theory_decks, ranked_decks, on=card_col, how='inner')['deck_id_x']
theory_could_be_ranked = theory_decks[theory_decks['deck_id'].isin(theory_decks_ids)]
decks = pd.concat([theory_could_be_ranked, ranked_decks])
# duplicates check
assert len(decks['deck_id'].unique()) == len(decks)
prc_left = round(decks.shape[0] / raw_length * 100)
print ('Decks removed :', raw_length - decks.shape[0])
print ('Original dataset left :', prc_left, '%')
release_dates = {
    'Explorers' : datetime(2015, 11, 12),
    'Old Gods' : datetime(2016, 4, 26),
    'Classic Nerfs' : datetime(2016, 3, 14),
    'Yogg Nerf' : datetime(2016, 10, 3),
    'Karazhan' : datetime(2016, 8, 11),
    'Gadgetzan' : datetime(2016, 12, 1),
    'Naxx Launch' : datetime(2014, 7, 22),
    'Live Patch 5506' : datetime(2014, 5, 28),
    'Undertaker Nerf' : datetime(2015, 1, 29),
    'Blackrock Launch' : datetime(2015, 4, 2),
    'GvG Launch' : datetime(2014, 12, 8),
    'TGT Launch' : datetime(2015, 8, 24),
    'Warsong Nerf' : datetime(2016, 1, 16),
    'Live Patch 4973' : datetime(2014, 3, 14),
    'Aggro Downfall' : datetime(2017, 2, 28),
    # 'Beta Patch 4944' : datetime(2014, 3, 11),
    # 'GvG Prelaunch' : datetime(2014, 12, 5)
}
date_decks = decks.set_index(pd.DatetimeIndex(decks['date'])).sort_index()
weekly_submissions = date_decks.resample('W')['date'].count()
fig = plt.figure()
ax = weekly_submissions.plot(figsize=(25, 10), fontsize=15)

for key, date in release_dates.items():
    ax.axvline(date, color='green', alpha=.35)
    ax.text(date, 12000, key, rotation=90, fontsize=15)
class_count = decks['deck_class'].value_counts()
class_count_df = class_count.to_frame().reset_index()
colors = {
    'Druid' : 'sandybrown',
    'Hunter' : 'green',
    'Mage' : 'royalblue',
    'Paladin' : 'gold',
    'Priest' : 'lightgrey',
    'Rogue' : 'darkgrey',
    'Shaman' : 'darkblue',
    'Warlock' : 'purple',
    'Warrior' : 'firebrick',
}

# sort colors to make plotting easier
colors = OrderedDict(sorted(colors.items()))
class_count_df['color'] = class_count_df['index'].replace(colors)
class_count_df.plot.pie(
    y='deck_class', 
    labels=class_count_df['index'], 
    colors=class_count_df['color'],
    autopct='%.2f', 
    fontsize=15,
    figsize=(10, 10),
    legend=False,
)
weekly_classes = date_decks.groupby('deck_class').resample('W').size().T
weekly_classes_rf = weekly_classes.divide(weekly_classes.sum(axis=1), axis=0)

ax = weekly_classes_rf.plot(
    kind='area', 
    figsize=(20, 12), 
    color=colors.values(), 
    alpha=.35,
    legend=False,
    fontsize=15,
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.2, key, rotation=90, fontsize=15)
rating_count = decks['rating'].value_counts(normalize=True).sort_index()
rating_count.head(n=10)
users = decks.groupby('user')['rating'].count().sort_values(ascending=False)
users.head(n=30).plot(kind='bar', figsize=(20, 10), fontsize=15, color=sns.color_palette())
# total of differents archetypes
decks['deck_archetype'].value_counts().size
known = decks[decks['deck_archetype'] != 'Unknown']
counts = known.groupby(['deck_class', 'deck_archetype']).size().reset_index()
for i, group in counts.groupby('deck_class'):
    fig = plt.figure()
    group.sort_values(0, ascending=False).plot(
        kind='bar', 
        x='deck_archetype', 
        title=str(i),
        color=colors[str(i)],
        legend=False,
        figsize=(15, 6),
        fontsize=15,
    )
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
def predict_archetype(class_name, decks, card_col, n_trees=500, max_feats=5, log=True):

    # known / unknown split
    dclass = decks[decks['deck_class'] == class_name]
    known = dclass[dclass['deck_archetype'] != 'Unknown']
    unknown = dclass[dclass['deck_archetype'] == 'Unknown']

    # data / target split
    X = known[card_col]
    y = known['deck_archetype']

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    
    # random forest
    clf = RandomForestClassifier(
        n_estimators=n_trees, 
        max_features=max_feats, 
        class_weight=None
    )
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # metrics
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, pred), 
        columns=clf.classes_, 
        index=clf.classes_
    )
    
    if log:
        print (classification_report(y_test, pred))
        # print (cohen_kappa_score(y_test, pred))
        # print (conf_matrix)
    
    return clf
# save the classifiers each class
clfs = {}
for c in decks['deck_class'].unique():
    clfs[c] = predict_archetype(c, decks, card_col)
paladin_clf = clfs['Paladin']
paladin_set = decks[decks['deck_class'] == 'Paladin']
paladin_ukn = paladin_set[paladin_set['deck_archetype'] == 'Unknown']
paladin_ukn.is_copy = False
pred = paladin_clf.predict(paladin_ukn[card_col])
paladin_ukn['deck_archetype'] = pred
paladin_ukn = paladin_ukn.set_index(pd.DatetimeIndex(paladin_ukn['date'])).sort_index()
weekly_paladin = paladin_ukn.groupby('deck_archetype').resample('W').size().reset_index()
weekly_paladin_piv = weekly_paladin.pivot(index='date', columns='deck_archetype', values=0).fillna(0)
weekly_paladin_rf = weekly_paladin_piv.divide(weekly_paladin_piv.sum(axis=1), axis=0)

ax = weekly_paladin_rf.plot(
    kind='area', 
    figsize=(20, 5), 
    legend=True,
    fontsize=15,
    color=sns.color_palette('Set3', 10)
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.5, key, rotation=90, fontsize=15)
from imblearn.over_sampling import SMOTE
# building a card id - set mapping from refs file
refs_dict = {c.get('dbfId') : c.get('set') for c in refs}
def multi_smote(X, y, kind='svm'):
    
    # regroup observations by classes
    full = pd.concat([X, y], axis=1)
    by_arch = full.groupby('deck_archetype')

    samples = []
    
    for name, group in by_arch:
        
        # create a 2-classes dataset
        all_but_one = full[full['deck_archetype'] != name]
        all_but_one.is_copy = False
        all_but_one['deck_archetype'] = 'Other'
        
        toSMOTE = pd.concat([group, all_but_one])
        _X = toSMOTE[X.columns]
        _y = toSMOTE['deck_archetype']
        
        # resample with 2 classes
        sm = SMOTE(kind=kind)
        X_re, y_re = sm.fit_sample(_X, _y)
        re = np.column_stack([X_re, y_re])
        
        # remove reference to other
        re = re[~(re == 'Other').any(axis=1)]
    
        samples.append(re)
        
    resampled = np.concatenate(samples)
    
    return resampled[:, :len(X.columns)], resampled[:, -1]
def predict_archetype(class_name, decks, card_col, refs, n_trees=500, max_feats=5, log=True):

    # known / unknown split
    dclass = decks[decks['deck_class'] == class_name]
    known = dclass[dclass['deck_archetype'] != 'Unknown']
    unknown = dclass[dclass['deck_archetype'] == 'Unknown']

    # data / target split
    X = known[card_col]
    y = known['deck_archetype']
    
    # adding expansions counts
    set_df = known[card_col].apply(pd.Series.replace, to_replace=refs_dict)
    counts = set_df.apply(pd.value_counts, axis=1).fillna(0)
    X = pd.concat([X, counts], axis=1)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5)
    
    # over-sampling the training set
    X_train, y_train = multi_smote(X_train, y_train)
    
    # random forest
    clf = RandomForestClassifier(
        n_estimators=n_trees, 
        max_features=max_feats, 
        class_weight=None
    )
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # metrics
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, pred), 
        columns=clf.classes_, 
        index=clf.classes_
    )
    
    if log:
        print (classification_report(y_test, pred))
        # print (conf_matrix)
        # print (cohen_kappa_score(y_test, pred))
        # print (clf.feature_importances_)
    
    return clf
# save the classifiers each class
clfs = {}
for c in decks['deck_class'].unique():
    clfs[c] = predict_archetype(c, decks, card_col, refs_dict)
from gensim import corpora, models
# building a card id - set mapping from refs file
names_dict = {c.get('dbfId') : c.get('name') for c in refs}

# we'll test paladin, as an example
subset = decks[decks['deck_class'] == 'Paladin']
names_df = subset[card_col].apply(pd.Series.replace, to_replace=names_dict)
lists = names_df.values.tolist()
dictionary = corpora.Dictionary(lists)
corpus = [dictionary.doc2bow(l) for l in lists]

# we'll consider the 10 most important topics, as a starter
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)

topics = ldamodel.print_topics(num_topics=10, num_words=30)

def get_topic(row):
    '''This function returns the most likely archetype'''
    topics = ldamodel.get_document_topics(row)
    best = max(topics, key=lambda x: x[1])
    return best[0]

# get the archetypes based on card BOW
pred = [get_topic(d) for d in corpus]
subset.is_copy = False
subset['deck_archetype'] = pred
paladin_ukn = subset.set_index(pd.DatetimeIndex(subset['date'])).sort_index()
weekly_paladin = paladin_ukn.groupby('deck_archetype').resample('W').size().reset_index()
weekly_paladin_piv = weekly_paladin.pivot(index='date', columns='deck_archetype', values=0).fillna(0)
weekly_paladin_rf = weekly_paladin_piv.divide(weekly_paladin_piv.sum(axis=1), axis=0)

ax = weekly_paladin_rf.plot(
    kind='area', 
    figsize=(20, 8), 
    legend=True,
    fontsize=15,
    color=sns.color_palette('Set3', 10)
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.3, key, rotation=90, fontsize=15)
topics[5]
topics[0]
topics[9]
topics[8]
topics[3]
topics[4]
topics[2]