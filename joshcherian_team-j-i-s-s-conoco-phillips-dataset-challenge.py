import pandas as pd

import numpy as np

import os

from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

# from sklearn import neighbors

from sklearn.metrics import f1_score

import seaborn as sns

import collections

import matplotlib.pyplot as plt



training_data = os.path.join(os.getcwd(), "tamu-datathon/equip_failures_training_neg_one.csv")

test_data = os.path.join(os.getcwd(), "tamu-datathon/equip_failures_test_set.csv")

training_data_cleaned = os.path.join(os.getcwd(), 'tamu-datathon/equip_failures_train_clean_drop_79.csv')

test_data_cleaned = os.path.join(os.getcwd(), 'tamu-datathon/equip_failures_test_clean_drop_79.csv')





def clean_data(df_clean):

    df_train = pd.read_csv(settings.training_data)

    df_test = pd.read_csv(settings.test_data)

    d1, d2 = clean_data(df_train, df_test)

    d1.to_csv(training_data_cleaned, index = False)

    d2.to_csv(test_data_cleaned, index = False)



# Clean Data

# clean_data(training_data)
import pickle

from sklearn.decomposition import NMF



df_training = pd.read_csv(training_data_cleaned)

df_drop = df_training.drop(columns=['id','target'])

df_drop = df_drop + 1



np_drop = df_drop.to_numpy()

model = NMF(n_components=2, init='random', random_state=0)

W = model.fit_transform(np_drop)



with open('new_matrix.pickle', 'wb') as handle:

    pickle.dump(W, handle)
def gen_output(predictions):

    columns = ['id', 'target']



    df = pd.read_csv(settings.test_data_cleaned)

    X_test = df.iloc[:, 1:]  # First one columns are id

    model = alg.fit(X, Y)

    predictions = model.predict(X_test)

    id = list(range(16002))

    id = id[1::]

    csv = pd.DataFrame()

    csv['id'] = id

    csv['target'] = predictions

    csv.to_csv('sample_drop79.csv', index = False)



print("Executing...")



df = pd.read_csv(training_data_cleaned)

X = df.iloc[:, 2:]  # First two columns are id and target

Y = np.array(df.iloc[:, 1])



alg = RandomForestClassifier(n_estimators=250)



cv = StratifiedKFold(n_splits=10)



fscores = []

for i, (train, test) in enumerate(cv.split(X, Y)):

    model = alg.fit(X.iloc[train], Y[train])

    Y_pred = model.predict(X.iloc[test])

    fscore = f1_score(Y[test], Y_pred, average='weighted', labels=np.unique(Y[test]))

    fscores.append(fscore)

    print('Fold', i, ':', fscore)



print('Average F-measure:', sum(fscores) / len(fscores))

#Plot Feature Importance

importances = list(model.feature_importances_)

column_headers = list(df.columns.values)

dicy = dict(zip(importances, column_headers))

dicysort = collections.OrderedDict(sorted(dicy.items()))

sns.set(style='whitegrid')

ax = sns.barplot(x=[dicysort[i] for i in dicysort.keys()], y=dicysort.keys(), data=dict(dicysort))

plt.show()