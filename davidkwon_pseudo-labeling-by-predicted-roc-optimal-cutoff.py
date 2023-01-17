import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
dpl = pd.read_csv('../input/siim-duplicates/siim_dpl2020.csv') 
#duplicate IDs including 9 more images as Chris mentioned here, https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/161943

train['is_train'] = 0
test['is_train'] = 1

#removing duplicates in train
train = train[~train['image_name'].isin(dpl['ISIC_id_paired'])]

train = train[['image_name', 'is_train', 'target']]
test = test[['image_name', 'is_train']]

path = '../input/models'

#predicted values for train and test, total 14 models
files_tr = [x for x in os.listdir(path) if x.endswith('_tr.csv')]
files_te = [x for x in os.listdir(path) if x.endswith('_te.csv')]

tr_models = {}
te_models = {}

for i in files_tr:
    tr_models[i] = pd.read_csv(path + '/' + i)
    tr_models[i]['predicted_' + i.replace('_tr.csv', '')] = tr_models[i]['pred']
    tr_models[i] = tr_models[i][['image_name', 'predicted_' + i.replace('_tr.csv', '')]]
    train = train.merge(tr_models[i], on = 'image_name', how = 'left')

for i in files_te:
    te_models[i] = pd.read_csv(path + '/' + i)
    te_models[i]['predicted_' + i.replace('_te.csv', '')] = te_models[i]['target']
    te_models[i] = te_models[i][['image_name','predicted_' + i.replace('_te.csv', '')]]
    
    test = test.merge(te_models[i], on = 'image_name', how = 'left')
    
test['target'] = 0

combine = pd.concat([train,test])

print(train.shape, test.shape)

train.columns
train.iloc[:,:5].head(5)
test.iloc[:, :5].head(5)
predicted_val = ['predicted_model' + str(x) for x in range(1,15)]
def ttest(col, *, data=combine, group="site", group_val=["1", "2"]):
    res = stats.ttest_ind(
        combine[combine[group] == group_val[0]][col],
        combine[combine[group] == group_val[1]][col],
        equal_var=False,
    )

    return np.round(res[1], 5)

def k2(col, *, data=combine, group="site", group_val=["1", "2"]):
    res = stats.ks_2samp(
        data[data[group] == group_val[0]][col], 
        data[data[group] == group_val[1]][col]
    )

    return np.round(res[1], 5)


#predicted target values in train and test have different distributions
#Kolmogorov-Smirnov test and T test are used

for i in predicted_val:
    print(f'model, {i}')
    print('--ks test p value --')
    print(k2(i, data = combine, group = 'is_train', group_val = [0, 1]))
    print('--t test p-value --')
    print(ttest(i, data = combine, group = 'is_train', group_val = [0, 1]))
    print()
def classify(y_true, y_pred, threshold):
    
    new_pred = y_pred >= threshold
    
    pos_pred = new_pred 
    neg_pred = 1-new_pred
    
    tp = np.sum(y_true * pos_pred)     #True Positive
    fp = np.sum((1-y_true) * pos_pred) #False Positive
    tn = np.sum((1-y_true) * neg_pred) #True Negative
    fn = np.sum(y_true * neg_pred)     #False Negative
    
    sensitivity = tp / (tp + fn)       #TP Ratio
    specificity = tn / (tn + fp)       #TN Ratio
    
    tpr = sensitivity
    fpr = fp/(fp + tn) #1 - specificity
    tnr = specificity
    
    return tpr, fpr, tnr, tp, tn

#simple ex
y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])

tpr, fpr, tnr, tp, tn = classify(y, scores, 0.35)

tpr, fpr, tnr, tp, tn


def opt_cutoff(train, feat, n_thresholds, do_print = True):
    #searching cutoff from min to max of predicted target values
    cutoff = np.linspace(train[feat].min(),
                         train[feat].max(),
                         n_thresholds)
    
    
    tpr = np.zeros(n_thresholds)
    fpr = np.zeros(n_thresholds)
    tnr = np.zeros(n_thresholds)
    
    for c, i in enumerate(cutoff):
        tpr1, fpr1, tnr1, tp, tn = classify(train['target'].values, train[feat].values, i)
        
        tpr[c] = tpr1
        fpr[c] = fpr1
        tnr[c] = tnr1
    
    optimal_idx = np.argmax(tpr + (-fpr))
    optimal_cutoff = cutoff[optimal_idx]
    
    
    new_tpr, new_fpr, new_tnr, tp, tn = classify(train['target'].values, train[feat].values, optimal_cutoff)
    
    if do_print:
        print(f'optimal cutoff, {optimal_cutoff}')
    
        plt.plot(fpr, tpr)
        plt.scatter(new_fpr, new_tpr)
        plt.title(f'{feat}, roc curve')
    
        print(f'tpr with cutoff {np.round(new_tpr, 5)}')
        print(f'fpr with cutoff {np.round(new_fpr, 5)}')
        print(f'tnr with cutoff {np.round(new_tnr, 5)}')
    
    feat_mean = train[feat].mean()
    feat_std = train[feat].std()
    
    return optimal_cutoff, feat_mean, feat_std
optimal_cutoff, col_avg, col_std = opt_cutoff(train, predicted_val[1], 5000) 
%%time
optimal_cutoff = np.zeros(len(predicted_val))
col_avg = np.zeros(len(predicted_val))
col_std = np.zeros(len(predicted_val))

for c, i in enumerate(tqdm(predicted_val)):
    oc, ca, cs = opt_cutoff(train, i, 10000, do_print = False)
    optimal_cutoff[c] = oc
    col_avg[c] = ca
    col_std[c] = cs
print('========================')
print('Real target value counts')
print(train['target'].value_counts())
print('========================')


tpr_s = []
fpr_s = []
tnr_s = []
sc = []

for c, i in enumerate(predicted_val):
    
    tpr, fpr, tnr, tp, tn = classify(train['target'].values, train[i].values, optimal_cutoff[c])
    score = np.round(roc_auc_score(train['target'].values, train[i].values),4)
    
    tpr_s.append(tpr)
    fpr_s.append(fpr)
    tnr_s.append(tnr)
    sc.append(score)
    
    print()
    print('---------------------')
    print(f'Model :: {i}')
    print(f'True Negative       : {tn}')
    print(f'True Negative Ratio : {np.int(np.round(tnr,2) * 100)}%')
    print()
    print(f'True Positive       : {tp}')
    print(f'True Positive Ratio : {np.int(np.round(tpr,2) * 100)}%')
    print(f'False Positive Ratio: {np.int(np.round(fpr,2) * 100)}%')
    print()
    print(f'OOF AUC score           : {score}')
    print('---------------------')
plt.figure(num=None, figsize=(10, 8), dpi=80)
plt.scatter(optimal_cutoff, col_avg)
for p, o, c in zip(predicted_val, optimal_cutoff, col_avg):
    plt.annotate(p, (o, c))
plt.title('Optimal Cutoff vs Predicted Value Average')

cor = np.round(np.corrcoef(optimal_cutoff, col_avg)[1,0], 7)


print(f'correlation between cutoff and prediction average, {cor}')
print('Linearity between those two seems acceptable, but not sure for now')
plt.figure(num=None, figsize=(10, 8), dpi=80)
plt.scatter(optimal_cutoff, col_avg + col_std)
for p, o, ca, cs in zip(predicted_val, optimal_cutoff, col_avg, col_std):
    plt.annotate(p, (o, ca + cs))
plt.title('Optimal Cutoff vs (Predicted Value Average + Standard Deviation)')

cor = np.round(np.corrcoef(optimal_cutoff, col_avg + col_std)[1,0], 7)
print(f'correlation between cutoff and prediction average + prediction std, {cor}')
print('No patterns')
plt.figure(num=None, figsize=(10, 8), dpi=80)
plt.scatter(optimal_cutoff-col_std, col_avg - col_std)
for p, o, ca, cs in zip(predicted_val, optimal_cutoff, col_avg, col_std):
    plt.annotate(p, (o - cs, ca - cs))

plt.title('(Optimal Cutoff - Std) vs (Predicted Value Average - Std)')

cor = np.round((np.corrcoef(optimal_cutoff-col_std, col_avg - col_std)[1,0]), 7)

print(f'correlation between cutoff - prediction std and prediction average - prediction std, {cor}')
print('Interesting pattern found here')
te_col_avg = np.zeros(len(predicted_val))
te_col_std = np.zeros(len(predicted_val))

for c, i in enumerate(tqdm(predicted_val)):
    te_col_avg[c] = test[i].mean()
    te_col_std[c] = test[i].std()

tr = pd.DataFrame({'cutoff' : optimal_cutoff, 'avg': col_avg, 'std' : col_std, 'cutoff-avg' : optimal_cutoff - col_avg, 
                   'cutoff-std': optimal_cutoff - col_std, 'avg+std':col_avg + col_std, 'avg-std':col_avg - col_std})
te = pd.DataFrame({'avg': te_col_avg, 'std' : te_col_std, 'avg-std' : te_col_avg - te_col_std})
plt.figure(num=None, figsize=(10, 8), dpi=80)
mask = np.zeros_like(tr.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(tr.corr(),mask = mask, annot = True)
def mae(x,y):
    return np.sum(np.abs(x - y))/len(x)

print(mae(tr['cutoff-std'], tr['avg-std']))
from xgboost import XGBRegressor

tree = XGBRegressor().fit(tr[['avg-std']], tr[['cutoff-std']])

score = np.int(np.round((tree.score(tr[['avg-std']], tr[['cutoff-std']]))*100, 0))

print(f'R2 score, {score}% variation explained')

tr_cutoff_std = tree.predict(tr[['avg-std']]).flatten()
te_cutoff_std = tree.predict(te[['avg-std']]).flatten()

#Since we predict the value of (ROC cutoff - target std) value,
#we add (target std) to the predicted (ROC cutoff - target std)
tr_cutoff = tr_cutoff_std + tr['std']
te_cutoff = te_cutoff_std + te['std']

tr_error = np.round(mae(tr_cutoff, optimal_cutoff), 8)
trte_error = np.round(mae(tr_cutoff, te_cutoff), 8)
trte_avg = np.round(mae(tr['avg'], te['avg']),8)

print(f'MAE,  predicted train optimal ROC cutoff vs true train optimal ROC cutoff     {tr_error}')
print(f'MAE,  predicted train optimal ROC cutoff vs predicted test optimal ROC cutoff {trte_error}')
print(f'MAE,  predicted target average for train vs predicted target average for test {trte_avg}')
print()
print('Gap of predicted ROC cutoff between train and test is simliar with the gap of predicted target average between train and test')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4), dpi=100)

sns.distplot(tr['avg'], hist = False, ax = ax1, label = 'train')
sns.distplot(te['avg'], hist = False, ax = ax1, label = 'test')
ax1.title.set_text('Average of Predicted target')

sns.distplot(tr_cutoff, hist = False, ax = ax2, label = 'train')
sns.distplot(te_cutoff, hist = False, ax = ax2, label = 'test')
ax2.title.set_text('ROC Cutoff of Predicted target')
pseudo_test = pd.DataFrame([])
pseudo_test['image_name'] = test['image_name']

for c, i in enumerate(predicted_val):
    pseudo_test[i + '_target'] = (test[i] >= te_cutoff[c]) * 1
    
pseudo_test['sum_by_cutoff'] = pseudo_test.drop(['image_name'], axis = 1).apply(lambda x: x.sum(), axis = 1)
pseudo_test[['image_name', 'sum_by_cutoff']].sort_values(['sum_by_cutoff'])
n_pos_pred = len(pseudo_test.loc[pseudo_test['sum_by_cutoff'] == 14, 'image_name'])

print(f'{n_pos_pred} confident predicted melanoma observations in test by all 14 models with predicted ROC cutoff')
n_neg_pred = len(pseudo_test.loc[pseudo_test['sum_by_cutoff'] == 0, 'image_name'])

print(f'{n_neg_pred} confident predicted benign observations in test by all 14 models with predicted ROC cutoff')