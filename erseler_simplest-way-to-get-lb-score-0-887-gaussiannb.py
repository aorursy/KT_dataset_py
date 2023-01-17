import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

features = [c for c in train.columns if c not in ['ID_code', 'target']]



target = np.array(train['target'])

test_id_code = test['ID_code']

test = test[features]



# print (train.shape, target.shape, test.shape)

print ("Data is ready!")
sns.countplot(target)
# Subsample the data to balance each class sample size

n_sample = 20000

sub_train = train.groupby('target').apply(lambda x: x.sample(n_sample))

sub_target = np.array(sub_train['target'])

sub_train = np.array(sub_train.drop(["ID_code", "target"], axis=1))



sns.countplot(sub_target)
random_state = 15

gnb = GaussianNB()



folds = KFold(n_splits=5, shuffle=True, random_state=random_state)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(sub_train)):

    print ("Fold: %d" % (fold_ + 1))



    x_train, y_train = sub_train[trn_idx], sub_target[trn_idx]

    x_val, y_val = sub_train[val_idx], sub_target[val_idx]



    gnb.fit(x_train, y_train)

    y_pred = gnb.predict(x_val)

    y_score = gnb.predict_proba(x_val)[:,1]



    acc = gnb.score(x_val, y_val)

    auc = roc_auc_score(y_val, y_score)



    print ("ACC: %.4f, AUC: %.4f" % (acc, auc))



clf_preds_train = gnb.predict_proba(sub_train)[:,1]

clf_preds_test = gnb.predict_proba(test)[:,1]
frames = [test_id_code,  pd.DataFrame(clf_preds_test)]

results = pd.concat(frames, axis=1)

results.rename(columns = { results.columns[1]: "target" }, inplace=True)

results[0:10]
len(results[results['target']<=0.5])
pd.DataFrame(results).describe()
results.to_csv("submission.csv", index=False)