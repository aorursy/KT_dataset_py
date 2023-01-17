import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression



from ivis import Ivis
data = pd.read_csv('../input/creditcard.csv')

Y = data['Class']
train_X, test_X, train_Y, test_Y = train_test_split(data, Y, stratify=Y,

                                                    test_size=0.95, random_state=1234)
standard_scaler = StandardScaler().fit(train_X[['Time', 'Amount']])

train_X.loc[:, ['Time', 'Amount']] = standard_scaler.transform(train_X[['Time', 'Amount']])

test_X.loc[:, ['Time', 'Amount']] = standard_scaler.transform(test_X[['Time', 'Amount']])



minmax_scaler = MinMaxScaler().fit(train_X)

train_X = minmax_scaler.transform(train_X)

test_X = minmax_scaler.transform(test_X)
ivis = Ivis(embedding_dims=2, model='maaten',

            k=15, n_epochs_without_progress=5,

            classification_weight=0.80,

            verbose=0)

ivis.fit(train_X, train_Y.values)
ivis.save_model('ivis-supervised-fraud')
train_embeddings = ivis.transform(train_X)

test_embeddings = ivis.transform(test_X)
fig, ax = plt.subplots(1, 2, figsize=(17, 7), dpi=200)

ax[0].scatter(x=train_embeddings[:, 0], y=train_embeddings[:, 1], c=train_Y, s=3, cmap='RdYlBu_r')

ax[0].set_xlabel('ivis 1')

ax[0].set_ylabel('ivis 2')

ax[0].set_title('Training Set')



ax[1].scatter(x=test_embeddings[:, 0], y=test_embeddings[:, 1], c=test_Y, s=3, cmap='RdYlBu_r')

ax[1].set_xlabel('ivis 1')

ax[1].set_ylabel('ivis 2')

ax[1].set_title('Testing Set')
clf = LogisticRegression(solver="lbfgs").fit(train_embeddings, train_Y)
labels = clf.predict(test_embeddings)

proba = clf.predict_proba(test_embeddings)
print(classification_report(test_Y, labels))



print('Confusion Matrix')

print(confusion_matrix(test_Y, labels))

print('Average Precision: '+str(average_precision_score(test_Y, proba[:, 1])))

print('ROC AUC: '+str(roc_auc_score(test_Y, labels)))