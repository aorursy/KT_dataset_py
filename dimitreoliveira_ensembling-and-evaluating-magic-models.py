import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.svm import SVC, NuSVC

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score



%matplotlib inline

sns.set_style("whitegrid")

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print('Train set shape:', train.shape)

print('Test set shape:', test.shape)

print('Train set overview:')

display(train.head())
N_FOLDS = 5
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic', 'preds']]

test['target_knn'] = 0

train['preds_knn'] = 0

test['target_mlp'] = 0

train['preds_mlp'] = 0

test['target_svc'] = 0

train['preds_svc'] = 0

test['target_nusvc'] = 0

train['preds_nusvc'] = 0

test['target_qda'] = 0

train['preds_qda'] = 0



# BUILD 512 MODELS

for i in range(512):

    print('wheezy-copper-turtle-magic {}\n'.format(i))

    

    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index

    idx2 = test2.index

    train2.reset_index(drop=True, inplace=True)

    

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]

    test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=0)

    counter = 0



    for train_index, val_index in skf.split(train3, train2['target']):

        counter += 1

        print('Fold {}\n'.format(counter))

        model_names = ['knn', 'mlp', 'svc', 'nusvc', 'qda']

        models = [KNeighborsClassifier(n_neighbors=17, p=2.9), 

                  MLPClassifier(random_state=3, activation='relu', solver='lbfgs', tol=1e-06, hidden_layer_sizes=(250, )), 

                  SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42), 

                  NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053), 

                  QuadraticDiscriminantAnalysis(0.1)]

        

        for i in range(len(model_names)):

            model = models[i]

            model_name = model_names[i]

            model.fit(train3[train_index,:], train2.loc[train_index]['target'])



            train_predictions = model.predict(train3[train_index,:])

            val_predictions = model.predict(train3[val_index,:])



            train_auc = roc_auc_score(train2.loc[train_index]['target'], train_predictions) * 100

            val_auc = roc_auc_score(train2.loc[val_index]['target'], val_predictions) * 100

            train_precision = precision_score(train2.loc[train_index]['target'], train_predictions) * 100

            val_precision = precision_score(train2.loc[val_index]['target'], val_predictions) * 100

            train_recall = recall_score(train2.loc[train_index]['target'], train_predictions) * 100

            val_recall = recall_score(train2.loc[val_index]['target'], val_predictions) * 100

            print('-----%s - Train----------' % model_name)

            print('AUC: %.2f Precision: %.2f Recall: %.2f \n' % (train_auc, train_precision, train_recall))

            print('-----%s - Validation-----' % model_name)

            print('AUC: %.2f Precision: %.2f Recall: %.2f \n' % (val_auc, val_precision, val_recall))



            # Make predictions

            train[('preds_%s' % model_name)].loc[idx1] += model.predict_proba(train3)[:,1] / N_FOLDS

            test[('target_%s' % model_name)].loc[idx2] += model.predict_proba(test3)[:,1] / N_FOLDS
train['preds_svcs'] = (train['preds_svc'] * 0.5) + (train['preds_nusvc'] * 0.5)

test['target_svcs'] = (test['target_svc'] * 0.5) + (test['target_nusvc'] * 0.5)



train['preds_avg'] = (train['preds_knn'] * 0.2) + (train['preds_mlp'] * 0.2) + (train['preds_svc'] * 0.2) + (train['preds_nusvc'] * 0.2) + (train['preds_qda'] * 0.2)

test['target_avg'] = (test['target_knn'] * 0.2) + (test['target_mlp'] * 0.2) + (test['target_svc'] * 0.2) + (test['target_nusvc'] * 0.2) + (test['target_qda'] * 0.2)



train['preds_avg2'] = (train['preds_knn'] * 0.2) + (train['preds_mlp'] * 0.05) + (train['preds_svc'] * 0.05) + (train['preds_nusvc'] * 0.7)

test['target_avg2'] = (test['target_knn'] * 0.2) + (test['target_mlp'] * 0.05) + (test['target_svc'] * 0.05) + (test['target_nusvc'] * 0.7)

f = plt.subplots(1, 1, figsize=(16, 5), sharex=True)

train_cnf_matrix = confusion_matrix(train['target'], [np.round(x) for x in train['preds_avg']])

train_cnf_matrix_norm = train_cnf_matrix / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=[0, 1], columns=[0, 1])

sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues")

plt.show()
f = plt.subplots(1, 1, figsize=(16, 5), sharex=True)

train_cnf_matrix = confusion_matrix(train['target'], [np.round(x) for x in train['preds_knn']])

train_cnf_matrix_norm = train_cnf_matrix / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=[0, 1], columns=[0, 1])

sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues")

plt.show()
f = plt.subplots(1, 1, figsize=(16, 5), sharex=True)

train_cnf_matrix = confusion_matrix(train['target'], [np.round(x) for x in train['preds_svc']])

train_cnf_matrix_norm = train_cnf_matrix / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=[0, 1], columns=[0, 1])

sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues")

plt.show()
print('KNN AUC %.2f' % roc_auc_score(train['target'], train['preds_knn']))

print('MLP AUC %.2f' % roc_auc_score(train['target'], train['preds_mlp']))

print('SVC AUC %.2f' % roc_auc_score(train['target'], train['preds_svc']))

print('NuSVC AUC %.2f' % roc_auc_score(train['target'], train['preds_nusvc']))

print('QDA AUC %.2f' % roc_auc_score(train['target'], train['preds_qda']))

print('SVCs AUC %.2f' % roc_auc_score(train['target'], train['preds_svcs']))

print('Averaged AUC %.2f' % roc_auc_score(train['target'], train['preds_avg']))

print('Averaged 2 AUC %.2f' % roc_auc_score(train['target'], train['preds_avg2']))
test[['id', 'target_avg', 'target_avg2', 'target_svcs', 'target_knn', 'target_mlp', 'target_svc', 'target_nusvc', 'target_qda']].head()
submission = test[['id', 'target_avg']]

submission.columns = ['id', 'target']

submission.to_csv('submission_avg.csv', index=False)

submission.head()
submission = test[['id', 'target_avg2']]

submission.columns = ['id', 'target']

submission.to_csv('submission_avg2.csv', index=False)

submission.head()
submission = test[['id', 'target_svcs']]

submission.columns = ['id', 'target']

submission.to_csv('submission_svcs.csv', index=False)

submission.head()
submission = test[['id', 'target_knn']]

submission.columns = ['id', 'target']

submission.to_csv('submission_knn.csv', index=False)

submission.head()
submission = test[['id', 'target_mlp']]

submission.columns = ['id', 'target']

submission.to_csv('submission_mlp.csv', index=False)

submission.head()
submission = test[['id', 'target_svc']]

submission.columns = ['id', 'target']

submission.to_csv('submission_svc.csv', index=False)

submission.head()
submission = test[['id', 'target_nusvc']]

submission.columns = ['id', 'target']

submission.to_csv('submission_nusvc.csv', index=False)

submission.head()
submission = test[['id', 'target_qda']]

submission.columns = ['id', 'target']

submission.to_csv('submission_qda.csv', index=False)

submission.head()