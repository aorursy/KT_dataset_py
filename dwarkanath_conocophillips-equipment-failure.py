import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.externals import joblib
plt.rcParams['figure.figsize'] = (10, 8)
train_og = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')
test_og = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
data = pd.concat([train_og, test_og], ignore_index=True, sort=False)
data.shape
data.head()
data.info(verbose=True, null_counts = True)
data = data.replace('na', np.nan)
data.info(verbose=True, null_counts = True)
data.columns
for c in data.columns:

    data[c] = pd.to_numeric(data[c])
data.info(verbose=True, null_counts = True)
data.head()
data.describe()
data_nonna = data.iloc[:,2:].fillna(data.mean())
data_nonna.head()
scaler = MinMaxScaler()
scaler.fit(data_nonna)
X = scaler.transform(data_nonna)
X.shape
y = data.target.values
y.shape
m = train_og.shape[0]
X_train, X_valid, y_train, y_valid = train_test_split(X[:m], y[:m], test_size=0.3, random_state = 0)
train_og.target.value_counts(normalize = True)
def get_f1_score(clf, X_train, X_valid, y_train, y_valid):

    """

    clf: classifier

    """

    y_train_preds = clf.predict(X_train)

    train_f1_score = f1_score(y_pred=y_train_preds, y_true=y_train)

    y_valid_preds = clf.predict(X_valid)

    valid_f1_score = f1_score(y_pred=y_valid_preds, y_true=y_valid)

    return train_f1_score, valid_f1_score
lrclassifier = LogisticRegression()

lrclassifier.fit(X_train, y_train)
get_f1_score(lrclassifier, X_train, X_valid, y_train, y_valid)
def select_sample(mult):

    sample_size = mult*y_train[y_train==1].shape[0]

    np.random.seed(2)

    sample_idx = np.concatenate([np.random.choice(np.where(y_train==0)[0], size = sample_size), np.where(y_train==1)[0]])

    X_train_sample = X_train[sample_idx]

    y_train_sample = y_train[sample_idx]

    return X_train_sample, y_train_sample
def get_f1_by_sample_lr(mult, is_sample=True):

    if is_sample:

        X_train_sample, y_train_sample = select_sample(mult)

    else:

        X_train_sample, y_train_sample = X_train, y_train

    lrclassifier = LogisticRegression()

    lrclassifier.fit(X_train_sample, y_train_sample)

    train_f1_score, valid_f1_score = get_f1_score(lrclassifier, X_train_sample, X_valid, y_train_sample, y_valid)

    

    return {'sample_frac': X_train_sample.shape[0]/X_train.shape[0], 'train_f1_score': train_f1_score, 'valid_f1_score': valid_f1_score}

num_mults = int(sum(y_train==0)/sum(y_train==1))
f1_mult_lr = []

mult = 1

while mult < num_mults:

    f1_mult_lr.append(get_f1_by_sample_lr(mult))

    if mult==1:

        mult+=9

    else:

        mult+=10

f1_mult_lr.append(get_f1_by_sample_lr(_, is_sample=False))
f1_mult_lr_df = pd.io.json.json_normalize(f1_mult_lr)
def plot_f1(df, x):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(x, 'train_f1_score', data = df)

    ax.plot(x, 'valid_f1_score', data = df)

    for i in range(df.shape[0]):

        ax.text(df.loc[i, x], df.loc[i, 'train_f1_score'], '%.2f' % df.loc[i, 'train_f1_score'], fontsize=12, ha='center', va='bottom')

        ax.text(df.loc[i, x], df.loc[i, 'valid_f1_score'], '%.2f' % df.loc[i, 'valid_f1_score'], fontsize=12, ha='center', va='bottom')

    ax.legend(fontsize = 12)

    plt.show()
plot_f1(f1_mult_lr_df, 'sample_frac')
def get_f1_by_sample_rf(mult, is_sample=True):

    if is_sample:

        X_train_sample, y_train_sample = select_sample(mult)

    else:

        X_train_sample, y_train_sample = X_train, y_train

    rfclassifier = RandomForestClassifier(n_estimators = 50, max_features = 10)

    rfclassifier.fit(X_train_sample, y_train_sample)

    train_f1_score, valid_f1_score = get_f1_score(rfclassifier, X_train_sample, X_valid, y_train_sample, y_valid)

    

    

    return {'sample_frac': X_train_sample.shape[0]/X_train.shape[0], 'train_f1_score': train_f1_score, 'valid_f1_score': valid_f1_score}

f1_mult_rf = []

mult = 1

while mult < num_mults:

    f1_mult_rf.append(get_f1_by_sample_rf(mult))

    if mult==1:

        mult+=9

    else:

        mult+=10

f1_mult_rf.append(get_f1_by_sample_rf(_, is_sample=False))
f1_mult_rf
f1_mult_rf_df = pd.io.json.json_normalize(f1_mult_rf)
plot_f1(f1_mult_rf_df, 'sample_frac')
X_train_sample, y_train_sample = select_sample(mult=50)

f1_est = []

num_estimators = 10

while num_estimators <= 100:

    rfclassifier = RandomForestClassifier(n_estimators = num_estimators, max_features = 10)

    rfclassifier.fit(X_train_sample, y_train_sample)

    train_f1_score, valid_f1_score = get_f1_score(rfclassifier, X_train_sample, X_valid, y_train_sample, y_valid)

    f1_est.append({'num_estimators': num_estimators, 'train_f1_score': train_f1_score, 'valid_f1_score': valid_f1_score})

    num_estimators+=10
f1_est
f1_est_df = pd.io.json.json_normalize(f1_est)
plot_f1(f1_est_df, 'num_estimators')
X_train_sample, y_train_sample = select_sample(mult=50)

f1_feat = []

max_features = 2

while max_features <= 20:

    rfclassifier = RandomForestClassifier(n_estimators = 60, max_features = max_features)

    rfclassifier.fit(X_train_sample, y_train_sample)

    train_f1_score, valid_f1_score = get_f1_score(rfclassifier, X_train_sample, X_valid, y_train_sample, y_valid)

    f1_feat.append({'max_features': max_features, 'train_f1_score': train_f1_score, 'valid_f1_score': valid_f1_score})

    max_features+=1
f1_feat_df = pd.io.json.json_normalize(f1_feat)
f1_feat_df
plot_f1(f1_feat_df, 'max_features')
final_classifier = RandomForestClassifier(n_estimators = 60, max_features = 19)

final_classifier.fit(X_train_sample, y_train_sample)
get_f1_score(final_classifier, X_train_sample, X_valid, y_train_sample, y_valid)
X_test = X[m:]
y_test_preds = final_classifier.predict(X_test)
submission = pd.DataFrame({'id': test_og.id, 'target': y_test_preds.astype(int)})
submission.head()
submission.shape
submission.to_csv('submission_dk_201910192153.csv',index=False)
test_sample = test_og.sample(10).reset_index(drop=True)
test_sample.replace('na', np.nan, inplace = True)
test_sample_nonna = test_sample.fillna(data.mean())
X_test_sample = scaler.transform(test_sample_nonna.iloc[:,1:])
X_test_sample.shape
y_test_sample_preds = final_classifier.predict(X_test_sample)
data.mean().to_pickle('data_means.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(final_classifier, 'clf.pkl')