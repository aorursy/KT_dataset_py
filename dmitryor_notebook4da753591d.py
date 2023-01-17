# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

#sns.set()

from sklearn import preprocessing

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
df = pd.read_csv("../input/train.csv")
from sklearn import covariance

NUMERIC_FEATURES = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

FEATURE_COL = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']



def AddMahalanobis(df):

    df2 = df.copy()

    for t in set(df.type):

        df_of_a_type = df[df.type == t]

        mcd = covariance.MinCovDet()

        mcd.fit(df_of_a_type[NUMERIC_FEATURES])

        df2[t + '_md'] = mcd.mahalanobis(df[NUMERIC_FEATURES])

    return df2



def AddPerColorFeatures(df, feature_cols=NUMERIC_FEATURES, drop_originals=False):

    df2 = df.copy()

    for feature_col in feature_cols:

        for c in set(df['color']):

            df2[feature_col + '_color_' + c] = df[feature_col] * (df['color'] == c)

    if drop_originals:

        df2 = df2.drop(feature_cols, axis=1)

    return df2



def GetFeatures(df):

    df = AddMahalanobis(df)

    # Does not colorize Malahanobis distances -- should it?

    df = AddPerColorFeatures(df)

    #df = AddPerColorFeatures(df, NUMERIC_FEATURES + [t + '_md' for t in set(df.type)], False)

    df = df.drop(['id', 'type'], axis=1)

    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(df.T.to_dict().values()), vec



def GetFeaturesMahalanobisOnly(df):

    df = AddMahalanobis(df)

    df = df[[t + '_md' for t in set(df.type)] + ['color']]

    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(df.T.to_dict().values()), vec



def GetTarget(df):

    le = preprocessing.LabelEncoder()

    return le.fit_transform(df.type), le



x_all, vec = GetFeaturesMahalanobisOnly(df)  # GetFeatures(df)

y_all, le = GetTarget(df)



print(vec.get_feature_names())

print(x_all[0])

print(x_all.shape, y_all.shape)
x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(

    x_all, y_all, range(len(x_all)), test_size=0.2, random_state=0)
# RandomForestClassifier example

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=30)

clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(metrics.confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt

import pylab as pl

from sklearn import metrics



print(metrics.confusion_matrix(y_all, clf.predict(x_all)))



pred = clf.predict(x_test)

labels = list(le.classes_)

cm = metrics.confusion_matrix(y_test, pred)#, labels)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

pl.title('Confusion matrix of the classifier')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

pl.xlabel('Predicted')

pl.ylabel('True')

pl.show()
print(x_train.shape, y_train.shape)

print(x_all.shape, y_all.shape)
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)



def TrainLinear(x_train, y_train, x_test, y_test):

    clf = tf.contrib.learn.LinearClassifier(

        feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(x_train),

        n_classes=3,

        #optimizer=tf.train.FtrlOptimizer(

        #    learning_rate=0.1,

        #    l2_regularization_strength=0.001,

        optimizer=tf.train.AdagradOptimizer(

            learning_rate=0.5,

        ))

    for epoch in range(5):

        clf.fit(x_train, y_train, steps=500)

        y_pred = clf.predict(x_test)

        print('training {}'.format(clf.evaluate(x=x_train, y=y_train)))

        print('validation {}'.format(clf.evaluate(x=x_test, y=y_test)))

    #    print('all {}'.format(clf.evaluate(x=x_all, y=y_all)))



    print(metrics.classification_report(y_test, y_pred))

    print('confusion matrix\n', metrics.confusion_matrix(y_test, y_pred))

    return clf



#clf = TrainLinear(x_train, y_train, x_test, y_test)

clf = TrainLinear(x_all, y_all, x_all, y_all)
clf.get_variable_names()

clf.get_variable_value('linear/_weight/Adagrad')
df_goblin = df[df.type=='Goblin']

sns.pairplot(df.drop(['id', 'color'], axis=1), hue="type", diag_kind='kde')

sns.pairplot(df_goblin.drop(['id'], axis=1), hue="color", diag_kind='kde')
sns.pairplot(df_goblin.drop(['id'], axis=1), hue="color", diag_kind='hist')
df3 = AddMahalanobis(df)

df3['type'][np.array(id_test)[(y_test == 2) & (y_pred == 1)]] = 'test_Goblin_pred_Ghoul'

df3['type'][np.array(id_test)[(y_test == 2) & (y_pred == 0)]] = 'test_Goblin_pred_Ghost'

sns.pairplot(df3.drop(['id'], axis=1), hue="type")
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)



def TrainDNN(x_train, y_train, x_test, y_test):

    clf = tf.contrib.learn.DNNClassifier(

        feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(x_train),

        n_classes=3,

        hidden_units=[9],

        optimizer=tf.train.ProximalAdagradOptimizer(

            learning_rate=0.1,

            l2_regularization_strength=0.001,

        ))

    for epoch in range(5):

        clf.fit(x_train, y_train, steps=500)

        y_pred = clf.predict(x_test)

        print('training ', clf.evaluate(x=x_train, y=y_train))

        print('validation ', clf.evaluate(x=x_test, y=y_test))



    print(metrics.classification_report(y_test, y_pred))

    print('confusion matrix\n', metrics.confusion_matrix(y_test, y_pred))

    return clf

    

clf = TrainDNN(x_all, y_all, x_all, y_all)
le.inverse_transform([0,1,2])
df_subm = pd.read_csv("../input/test.csv")

x_subm, _ = GetFeatures(df_subm)

pred_subm = clf.predict(x_subm)

fdy_subm = pd.DataFrame()

fdy_subm['id'] = df_subm['id']

fdy_subm['type'] = le.inverse_transform(pred_subm)

fdy_subm.to_csv("submission.csv", index=False)
print(check_output(["cat", "submission.csv"]).decode("utf8"))