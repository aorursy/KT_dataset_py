import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

input_dir = '/kaggle/input/mushroom-classification/mushrooms.csv'

data = pd.read_csv(input_dir)

data.shape
print("Any nulls?", data.isnull().any().any()) # no nulls
data.info() # all good, all categorical data
data.head()
labels_data = pd.DataFrame({col: [len(set(data[col])), list(set(data[col]))] for col in data.columns}).T
labels_data.columns = ['# unique labels', 'Labels']
labels_data
if 'veil-type' in data.columns:
    data.drop('veil-type', axis=1, inplace=True)
data.shape
sns.set_style('darkgrid')

fig, axs = plt.subplots(7,3, figsize=(20,15))

plt.subplots_adjust(top=2, bottom=None)


for i, col in enumerate(data.columns[1:]):
    sns.countplot(
        x=col, hue='class',
        data=data,
        ax=axs[i//3, i%3]
    )
    axs[i//3, i%3].set_title(col)
    axs[i//3, i%3].set_xlabel(None)
    axs[i//3, i%3].set_ylabel(None)
    axs[i//3, i%3].legend(loc='upper right')
plt.show()
spc_count = data.query('`spore-print-color`=="h"')['class'].value_counts()
print(spc_count)
print(spc_count['p']/spc_count.sum())
gc_count = data.query('`gill-color`=="b"')['class'].value_counts()
print(gc_count)
print(gc_count['p']/gc_count.sum())
other_toxic_count = data.query('`spore-print-color`!="h" & `gill-color`!="b"')['class'].value_counts()
print(other_toxic_count)
print(other_toxic_count['p']/other_toxic_count.sum())
data_excl_1 = data.query('`spore-print-color`!="h" & `gill-color`!="b"')

sns.set_style('darkgrid')

fig, axs = plt.subplots(7,3, figsize=(20,15))

plt.subplots_adjust(top=2, bottom=None)


for i, col in enumerate(data.columns[1:]):
    sns.countplot(
        x=col, hue='class',
        data=data_excl_1,
        ax=axs[i//3, i%3]
    )
    axs[i//3, i%3].set_title(col)
    axs[i//3, i%3].set_xlabel(None)
    axs[i//3, i%3].set_ylabel(None)
    axs[i//3, i%3].legend(loc='upper right')
plt.show()
data_excl_2 = data_excl_1.query('odor!="c" & odor!="p"')
data_excl_2_counts = data_excl_2['class'].value_counts()

print(data_excl_2_counts)
print(data_excl_2_counts['p']/data_excl_2_counts.sum())
sns.set_style('darkgrid')

fig, axs = plt.subplots(7,3, figsize=(20,15))

plt.subplots_adjust(top=2, bottom=None)


for i, col in enumerate(data.columns[1:]):
    sns.countplot(
        x=col, hue='class',
        data=data_excl_2,
        ax=axs[i//3, i%3]
    )
    axs[i//3, i%3].set_title(col)
    axs[i//3, i%3].set_xlabel(None)
    axs[i//3, i%3].set_ylabel(None)
    axs[i//3, i%3].legend(loc='upper right')
plt.show()
data_excl_3 = data_excl_2.query('`stalk-shape`!="e"')
data_excl_3_counts = data_excl_3['class'].value_counts()
print(data_excl_3_counts)
data_excl_4 = data_excl_2.query('`spore-print-color`!="w" & `spore-print-color`!="r"')
data_excl_4_counts = data_excl_4['class'].value_counts()
print(data_excl_4_counts)
data_excl_4_counts - data_excl_3_counts
data['class'].value_counts()['e'] - data_excl_4_counts['e']
labels_data = pd.DataFrame({col: [len(set(data[col])), list(set(data[col])), '?' in set(data[col])] for col in data.columns}).T
labels_data.columns = ['# unique labels', 'Labels', 'Unknown (\'?\') label']
labels_data
from sklearn.preprocessing import OneHotEncoder

# For convenience, pre-shuffle the data before splitting it into features and targets
data = data.sample(frac=1) 

# Target labels (edible - 1 ; poisonous - 0)
data_y = data['class'].apply(lambda x: 1 if x=='e' else 0)

# Predictive features. An empty DataFrame for now, but soon we will populate it with properly encoded data
data_X = pd.DataFrame()

# Encoder for encoding features with more than 2 labels
encoder = OneHotEncoder()

for col_name in data.columns[1:]: # For each column except the first one, which is 'class'
    # The list of all unique labels in the column
    col_unique = list(set(data[col_name].values))
    # If there are only two unique labels
    if len(col_unique)==2:
        # Encode as binary
        col_encoded = data[col_name].apply(lambda x: 0 if x==col_unique[0] else 1)
        # For better interpretability, contain the meaning of 0s and 1s with respect to the original descriptive labels in the name of the encoded column
        col_encoded_name = f'{col_name}-bin{col_unique[0]}-{col_unique[1]}'
        # Add this encoded column to the data_X DataFrame
        data_X[col_encoded_name] = col_encoded
    # If there are more than two unique labels
    else:
        # Encode as one-hot with previously initialized OneHotEncoder
        # Immediately convert into a numpy array
        col_encoded = encoder.fit_transform(data[col_name].values.reshape(-1,1)).toarray()
        # Labels for each column of the one-hot array, into which this column has just been encoded
        col_encoded_labels = encoder.categories_[0]
        # For each of this column's labels
        for i, label in enumerate(col_encoded_labels):
            if label=='?': # Skip, if this label is '?'
                continue
            # The name of the column encoding this label will contain this label's name (again, for interpretability)
            col_encoded_name = f'{col_name}-{label}'
            # Add this column to the data_X DataFrame 
            data_X[col_encoded_name] = col_encoded[:,i]
            
data_X.shape, data_y.shape
expected_n_columns = 0
for col in data.columns[1:]:
    L = len(set(data[col])) 
    if L==2:
        expected_n_columns += 1
    else:
        if '?' in set(data[col]):
            L-=1
        expected_n_columns += L
expected_n_columns
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import cross_val_score

dtc = DTC()

dtc_cvs_acc = cross_val_score(dtc, data_X, data_y, scoring='accuracy', cv=3, n_jobs=-1)

print("\tDecision Tree Classifier:")
print(dtc_cvs_acc.round(5))
print(f"Mean: {dtc_cvs_acc.mean().round(5)}\tStd: {dtc_cvs_acc.std().round(5)}")
from sklearn.ensemble import RandomForestClassifier as RFC

rfc = RFC()

rfc_cvs_acc = cross_val_score(rfc, data_X, data_y, scoring='accuracy', cv=3, n_jobs=-1)
print("\Random Forest Classifier:")
print(rfc_cvs_acc.round(7))
print(f"Mean: {rfc_cvs_acc.mean().round(5)}\tStd: {rfc_cvs_acc.std().round(5)}")
from sklearn.model_selection import KFold

splits = KFold(n_splits=10, shuffle=True)

rfc_scores = [] # A list to write scores into 
forests = [] # A list to contain the trained models (to be explained in a while)

for test_idx, train_idx in tqdm(splits.split(data_X)):
    train_X, train_y = data_X.iloc[train_idx], data_y.iloc[train_idx]
    test_X, test_y = data_X.iloc[test_idx], data_y.iloc[test_idx]
    
    rfc = RFC()
    rfc.fit(train_X, train_y)
    
    rfc_score = rfc.score(test_X, test_y)
    rfc_scores.append(rfc_score)
    forests.append(rfc)
    
print("\tRandom Forest Classifier (reverse cross-validation):")
print(f"Mean: {np.mean(rfc_scores)}\tMin: {np.min(rfc_scores)}")
rfc_fi_df = pd.DataFrame()

rfc = RFC()
rfc.fit(data_X, data_y)
rfc_fi_df['RFC 1'] = rfc.feature_importances_
rfc = RFC()
rfc.fit(data_X, data_y)
rfc_fi_df['RFC 2'] = rfc.feature_importances_

rfc_fi_df.index = data_X.columns

rfc_fi_df
fig, axs = plt.subplots(1,2, figsize=(15,9))

rfc_fi_df['RFC 1'].hist(ax=axs[0])
axs[0].set_ylim(top=10) # Most of the features are of very low importance, which is represented below as a "skyscrapper" at the left, so we will "zoom in" a little bit
rfc_fi_df['RFC 2'].hist(ax=axs[1])
axs[1].set_ylim(top=10)

plt.show()
# Average feature importances
av_fi = np.zeros((rfc.feature_importances_.shape))

for rfc in forests:
    av_fi += rfc.feature_importances_
    
av_fi /= len(forests)

# Plot the data

fig, ax = plt.subplots(figsize=(15,9))

ax.hist(av_fi)
ax.set_ylim(top=10)
plt.show()
av_fi_df = pd.Series({feature: importance for feature, importance in zip(data_X.columns, av_fi)}).to_frame('importance')
av_fi_df
features_01 = av_fi_df.query('importance >= 0.01').index.values
features_02 = av_fi_df.query('importance >= 0.02').index.values
features_04 = av_fi_df.query('importance >= 0.04').index.values
features_01.shape, features_02.shape, features_04.shape
rfc_01_scores, rfc_02_scores, rfc_04_scores = [], [], []

splits = KFold(n_splits=10, shuffle=True)

for test_idx, train_idx in tqdm(splits.split(data_X)):
    train_X, train_y = data_X.iloc[train_idx], data_y.iloc[train_idx]
    test_X, test_y = data_X.iloc[test_idx], data_y.iloc[test_idx]
    
    rfc = RFC()
    rfc.fit(train_X[features_01], train_y)
    rfc_01_score = rfc.score(test_X[features_01], test_y)
    rfc_01_scores.append(rfc_01_score)
    
    rfc = RFC()
    rfc.fit(train_X[features_02], train_y)
    rfc_02_score = rfc.score(test_X[features_02], test_y)
    rfc_02_scores.append(rfc_02_score)
    
    rfc = RFC()
    rfc.fit(train_X[features_04], train_y)
    rfc_04_score = rfc.score(test_X[features_04], test_y)
    rfc_04_scores.append(rfc_04_score)
print(f"0.01:\n\tMean:{np.mean(rfc_01_scores).round(5)}\tMin:{np.min(rfc_01_scores).round(5)}\tStd:{np.std(rfc_01_scores).round(5)}")
print(f"0.02:\n\tMean:{np.mean(rfc_02_scores).round(5)}\tMin:{np.min(rfc_02_scores).round(5)}\tStd:{np.std(rfc_02_scores).round(5)}")
print(f"0.04:\n\tMean:{np.mean(rfc_04_scores).round(5)}\tMin:{np.min(rfc_04_scores).round(5)}\tStd:{np.std(rfc_04_scores).round(5)}")
print(f"All:\n\tMean:{np.mean(rfc_scores).round(5)}\tMin:{np.min(rfc_scores).round(5)}\tStd:{np.std(rfc_scores).round(5)}")
from sklearn.model_selection import train_test_split as tts

rfc_01_cv = cross_val_score(RFC(), data_X[features_01], data_y, cv=5, scoring='accuracy', n_jobs=-1,)
rfc_02_cv = cross_val_score(RFC(), data_X[features_02], data_y, cv=5, scoring='accuracy', n_jobs=-1,)
rfc_04_cv = cross_val_score(RFC(), data_X[features_04], data_y, cv=5, scoring='accuracy', n_jobs=-1,)
rfc_all_cv = cross_val_score(RFC(), data_X, data_y, cv=5, scoring='accuracy', n_jobs=-1,)

print(f"0.01:\n\tMean:{np.mean(rfc_01_cv).round(5)}\tMin:{np.min(rfc_01_cv).round(5)}\tStd:{np.std(rfc_01_cv).round(5)}")
print(f"0.02:\n\tMean:{np.mean(rfc_02_cv).round(5)}\tMin:{np.min(rfc_02_cv).round(5)}\tStd:{np.std(rfc_02_cv).round(5)}")
print(f"0.04:\n\tMean:{np.mean(rfc_04_cv).round(5)}\tMin:{np.min(rfc_04_cv).round(5)}\tStd:{np.std(rfc_04_cv).round(5)}")
print(f"All:\n\tMean:{np.mean(rfc_all_cv).round(5)}\tMin:{np.min(rfc_all_cv).round(5)}\tStd:{np.std(rfc_all_cv).round(5)}")
important_df = av_fi_df.loc[features_01].sort_values(by='importance', ascending=False)
important_df
rfc_01_best = None
rfc_01_best_score = 0

splits = KFold(n_splits=10, shuffle=True)
for test_idx, train_idx in tqdm(splits.split(data_X)):
    train_X, train_y = data_X.iloc[train_idx][features_01], data_y.iloc[train_idx]
    test_X, test_y = data_X.iloc[test_idx][features_01], data_y.iloc[test_idx]
    
    rfc = RFC()
    rfc.fit(train_X, train_y)
    rfc_01_score = rfc.score(test_X[features_01], test_y)
    if rfc_01_score > rfc_01_best_score:
        rfc_01_best_score = rfc_01_score
        rfc_01_best = rfc
dtc_df = pd.Series({idx: dtc.score(data_X[features_01], data_y) for idx, dtc in enumerate(rfc_01_best.estimators_) }).to_frame('Score').sort_values(by='Score', ascending=False)

dtc_df,
dtc_df['Score'].hist()
plt.show()
from sklearn.tree import export_graphviz

dtc_best = rfc_01_best.estimators_[dtc_df.index[0]]

export_graphviz(
    dtc_best,
    out_file='dtc_best.dot',
    feature_names=features_01,
    class_names=['p', 'e'],
    rounded=True,
    filled=True
)
    

os.getcwd(), os.listdir() # See, that the saved .dot file is in our /working directory
! dot -Tpng dtc_best.dot -o dtc_best.png
os.listdir() # Convert this .dot file to .png
from IPython.display import Image

# Display this .png

img = 'dtc_best.png'
Image(url=img, embed=False)
# We need to access the predictor's index in order to access its depth, number of leaves and mean probability
if 'index' not in dtc_df.columns:
    dtc_df.reset_index(inplace=True, drop=False)

dtc_df['Depth'] = dtc_df['index'].apply(lambda idx: rfc_01_best.estimators_[idx].get_depth())
dtc_df['Leaves'] = dtc_df['index'].apply(lambda idx: rfc_01_best.estimators_[idx].get_n_leaves())
dtc_df['Mean proba'] = dtc_df['index'].apply(lambda idx: rfc_01_best.estimators_[idx].predict_proba(data_X[features_01]).max(axis=1).mean())

dtc_df
dtc_df['Mean proba'].value_counts()
fig, ax = plt.subplots(1,2, figsize=(15, 9))

dtc_df['Depth'].hist(bins = dtc_df['Depth'].max() - dtc_df['Depth'].min() +1, ax=ax[0] )
ax[0].set_title('Depth')
dtc_df['Leaves'].hist(bins = dtc_df['Leaves'].max() - dtc_df['Leaves'].min() +1, ax=ax[1] )
ax[1].set_title('Leaves')

plt.show()
DEPTH = np.arange(2, 15)
LEAVES = np.arange(5, 35)

accuracy_matrix = np.zeros((len(DEPTH), len(LEAVES)))
probas_matrix = np.zeros((len(DEPTH), len(LEAVES)))

splits = KFold(n_splits=10, shuffle=True)
for i, depth in tqdm(enumerate(DEPTH)):
    for ii, leaves in enumerate(LEAVES):
        #print(f"{np.round(100*(i/len(DEPTH)+(ii/len(LEAVES))/len(DEPTH)), 2)} %")
        for test_idx, train_idx in splits.split(data_X):
            train_X, train_y = data_X.iloc[train_idx][features_01], data_y.iloc[train_idx]
            test_X, test_y = data_X.iloc[test_idx][features_01], data_y.iloc[test_idx]

            rfc = RFC(max_depth=depth, max_leaf_nodes=leaves)
            rfc.fit(train_X, train_y)
            
            accuracy_matrix[i,ii] += rfc.score(test_X, test_y)
            probas_matrix[i,ii] += rfc.predict_proba(test_X).max(axis=1).mean()

accuracy_matrix /= 10
probas_matrix /= 10
fig, ax = plt.subplots(figsize=(15,9))

accuracy_heatmap = ax.imshow(accuracy_matrix)

ax.set_yticks(np.arange(len(DEPTH)))
ax.set_yticklabels(DEPTH)

ax.set_xticks(np.arange(len(LEAVES)))
ax.set_xticklabels(LEAVES)

ax.set_title("Accuracy")

fig.tight_layout()

plt.colorbar(accuracy_heatmap)

plt.show()
fig, ax = plt.subplots(figsize=(15,9))

probas_heatmap = ax.imshow(probas_matrix)

ax.set_yticks(np.arange(len(DEPTH)))
ax.set_yticklabels(DEPTH)

ax.set_xticks(np.arange(len(LEAVES)))
ax.set_xticklabels(LEAVES)

ax.set_title("Probas")

fig.tight_layout()

plt.colorbar(probas_heatmap)

plt.show()
print("\tMaximally constrained forest:")
print(f"Accuracy: {accuracy_matrix[0,0].round(3)}")
print(f"Probas: {probas_matrix[0,0].round(3)}")
# Find the best constrained forest
rfc_constrained = None
rfc_constrained_score = 0

splits = KFold(n_splits=10, shuffle=True)
for test_idx, train_idx in tqdm(splits.split(data_X)):
    train_X, test_X = data_X.iloc[train_idx][features_01], data_X.iloc[test_idx][features_01]
    train_y, test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]
    
    rfc = RFC(max_depth=2, max_leaf_nodes=5)
    rfc.fit(train_X, train_y)
    rfc_score = rfc.score(test_X, test_y)
    if rfc_score > rfc_constrained_score:
        rfc_constrained_score = rfc_score
        rfc_constrained = rfc

# Plot the data about decision trees

constrained_dtc_df = pd.Series({idx: dtc.score(data_X[features_01], data_y) for idx, dtc in enumerate(rfc_constrained.estimators_)}).to_frame('Score').sort_values(by='Score', ascending=False)
constrained_dtc_df.reset_index(inplace=True, drop=False)
constrained_dtc_df['Probas'] = constrained_dtc_df['index'].apply(lambda idx: rfc_constrained.estimators_[idx].predict_proba(data_X[features_01]).max(axis=1).mean())


# Compare their average individual accuracy and certainty to that of the whole ensemble (tested on the entire dataset)
print(f"Mean DTC accuracy:\t{np.round(constrained_dtc_df['Score'].mean(), 4)}")
print(f"Mean DTC probas:\t{np.round(constrained_dtc_df['Probas'].mean(), 4)}")
print(f"Mean ensemble accuracy:\t{np.round(rfc_constrained.score(data_X[features_01], data_y), 4)}")
print(f"Mean ensemble probas:\t{np.round(rfc_constrained.predict_proba(data_X[features_01]).max(axis=1).mean())}")

fig, ax = plt.subplots(1,2,figsize=(15,9))

constrained_dtc_df['Score'].hist(ax=ax[0])
ax[0].set_title('Score')
constrained_dtc_df['Probas'].hist(ax=ax[1])
ax[1].set_title('Probas')

plt.show()
dtc_constrained = rfc_constrained.estimators_[0]

export_graphviz(
    dtc_constrained,
    out_file='dtc_constrained.dot',
    feature_names=features_01,
    class_names=['p', 'e'],
    rounded=True,
    filled=True
)
! dot -Tpng dtc_constrained.dot -o dtc_constrained.png
img = 'dtc_constrained.png'
Image(url=img, embed=False)
feat_accuracies = np.zeros((av_fi_df.shape[0],))
feat_probas = np.zeros((av_fi_df.shape[0],))

splits = KFold(n_splits=10, shuffle=True)
for n_features in tqdm(range(av_fi_df.shape[0])):
    for test_idx, train_idx in splits.split(data_X):
        features = av_fi_df.index[:n_features+1]
        train_X, test_X = data_X.iloc[train_idx][features], data_X.iloc[test_idx][features]
        train_y, test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]
        rfc = RFC(max_depth=10, max_leaf_nodes=30)
        rfc.fit(train_X, train_y)
        
        feat_accuracies[n_features] += rfc.score(test_X, test_y)
        feat_probas[n_features] += rfc.predict_proba(test_X).max(axis=1).mean()
        
feat_accuracies /= 10
feat_probas /= 10
feat_df = pd.DataFrame([feat_accuracies, feat_probas]).T
feat_df.columns = ['Accuracies', 'Probas']

fig, ax = plt.subplots(figsize=(15,9))

sns.lineplot(data=feat_df, ax=ax)

ax.set_xlabel('Number of features')
ax.set_title('Accuracy and certainty')

plt.show()
model = models.Sequential(name='shroom_classifier',layers=[
    layers.Dense(32, activation='relu', kernel_regularizer='l2', input_shape=(train_X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(.1),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(.1),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)


train_X, test_X, train_y, test_y = tts(data_X, data_y_bin, test_size=.9, random_state=42, shuffle=True)

history = model.fit(
    train_X, train_y,
    validation_split=.1, shuffle=True,
    batch_size=32, epochs=5
)