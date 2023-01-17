import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
DATA_PATH = '../input'
df_train = pd.read_csv(DATA_PATH + '/train.csv', encoding='cp1252')
df_train.shape
df_train['ciphertext_len'] = df_train['ciphertext'].apply(lambda x: len([y.encode() for y in x]))
df_train.head()
from sklearn.feature_extraction.text import TfidfVectorizer
%%time
vect = TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_features=30000)
X_train_features_sparse = vect.fit_transform(df_train['ciphertext'])
X_train_features_sparse
from scipy.sparse import hstack
X_train = X_train_features_sparse.tocsr()
X_train
y_train = df_train['target']
df_test = pd.read_csv(DATA_PATH + '/test.csv', encoding='cp1252')
%%time
X_test_features_sparse = vect.transform(df_test['ciphertext'])
X_test = X_test_features_sparse.tocsr()
X_test
del(vect)
diffs = list(range(1, 5))
from sklearn.model_selection import train_test_split
def split_idx_by_column(df, column, valid_size=None):
    idxs, idxs_valid = {}, {}
    for d in diffs:
        idx = df.index[df[column] == d]
        if valid_size is None:
            idxs[d] = idx
        else:
            idx, idx_valid = train_test_split(idx, random_state=42, 
                                              test_size=valid_size, stratify=df['target'][idx])
            idxs[d] = idx
            idxs_valid[d] = idx_valid
    if valid_size is None:
        return idxs
    else:
        return idxs, idxs_valid
train_idxs = split_idx_by_column(df_train, 'difficulty')
train_part_idxs, valid_idxs = split_idx_by_column(df_train, 'difficulty', valid_size=0.1)
test_idxs = split_idx_by_column(df_test, 'difficulty')
print('train part sizes:', [z.shape[0] for z in train_part_idxs.values()])
print('valid sizes:', [z.shape[0] for z in valid_idxs.values()])
print('test sizes:', [z.shape[0] for z in test_idxs.values()])
y_valid_to_concat = []
for d in diffs:
    y_valid_to_concat.append(y_train.loc[valid_idxs[d]])
y_valid = pd.concat(y_valid_to_concat)
y_valid.sort_index(inplace=True)
y_valid.index
for d in diffs:
    plt.figure()
    plt.title(f'Difficulty {d}')
    idx = train_part_idxs[d].values
    plt.hist(y_train[idx], bins=20, normed=False, alpha=0.5)
    idx = valid_idxs[d].values
    plt.hist(y_train[idx], bins=20, normed=False, alpha=0.5)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
pipes = {}
for d in diffs:
    pipe = Pipeline(memory=None, steps=[
        ('scaler', MaxAbsScaler(copy=False)),
        ('clf', LogisticRegression(solver='lbfgs', multi_class='multinomial', verbose=2, n_jobs=-1))
    ])
    pipes[d] = pipe
def train(models, X, y, diff_idxs):
    for d in diffs:
        idx = diff_idxs[d].values
        print(f'difficulty = {d}, samples = {idx.shape[0]}')
        model = models[d]
        model.fit(X[idx], y.loc[idx])
    return models
%%time
train(pipes, X_train, y_train, train_part_idxs)
from sklearn.metrics import confusion_matrix
def predict(models, X, diff_idxs, show_graph=True, y_truth=None):
    y_preds = {}
    for d in diffs:
        idx = diff_idxs[d].values
        model = models[d]
        y_pred = model.predict(X[idx])
        y_preds[d] = pd.Series(data=y_pred, index=idx)
        print(f'difficulty = {d}, valid_preds = {y_preds[d].shape}')
        if show_graph:
            plt.figure(figsize=(12,4))
            plt.subplot(121)
            plt.title(f'Difficulty {d}')
            plt.hist(y_pred, bins=20, normed=False, label='pred', alpha=0.5)
            if y_truth is not None:
                plt.hist(y_truth[idx], bins=20, label='valid', alpha=0.5)
            plt.gca().set_xticks(range(20))
            plt.grid()
            plt.legend()
            if y_truth is not None:
                cm = confusion_matrix(y_truth[idx], y_pred)
                plt.subplot(122)
                plt.imshow(cm)
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
    y_pred_to_concat = []
    for d in diffs:
        y_pred_to_concat.append(y_preds[d])
    y_pred = pd.concat(y_pred_to_concat)
    y_pred.sort_index(inplace=True)
    return y_pred
y_valid_pred = predict(pipes, X_train, valid_idxs, y_truth=y_valid)
from sklearn.metrics import f1_score, precision_recall_fscore_support
f1_score(y_valid, y_valid_pred, average='macro')
precision_recall_fscore_support(y_valid, y_valid_pred, average='macro')
plt.hist(y_valid, bins=20, label='valid', alpha=0.5)
plt.hist(y_valid_pred, bins=20, label='valid_pred', alpha=0.5)
plt.gca().set_xticks(range(20))
plt.grid()
plt.legend()
pass
cm = confusion_matrix(y_valid, y_valid_pred)
plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
cv = StratifiedKFold(2)
params = {
    'clf__C': np.logspace(-2, 2, 5)
}
grids = {}
for d in diffs:
    pipe = pipes[d]
    grid = GridSearchCV(estimator=pipe, cv=cv, param_grid=params, 
                        scoring='f1_macro', return_train_score=True, verbose=2)
    grids[d] = grid
%%time
train(grids, X_train, y_train, train_idxs)
for d in diffs:
    print(f'Difficulty = {d}')
    print(grids[d].cv_results_)
models = {}
for d in diffs:
    model = grids[d].best_estimator_
    models[d] = model
    print(f'Difficulty = {d}, C={model.steps[1][1].C}')
%%time
y_test_pred = predict(models, X_test, test_idxs)
plt.hist(y_train, bins=20, label='train', alpha=0.5, density=True)
plt.hist(y_test_pred, bins=20, label='pred', alpha=0.5, density=True)
plt.gca().set_xticks(range(20))
plt.grid()
plt.legend()
pass
df_subm = pd.read_csv(DATA_PATH +'/sample_submission.csv')
df_subm['Predicted'] = y_test_pred
df_subm.head()
df_subm.to_csv('submission.csv', index=False)


