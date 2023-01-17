import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.1f}'.format
header_list = [ "id","ccf","age","sex","painloc","painexer","relrest","pncaden","cp","trestbps",
"htn","chol","smoke","cigs","years","fbs","dm","famhist","restecg","ekgmo",
"ekgday","ekgyr","dig","prop","nitr","pro","diuretic","proto","thaldur","thaltime",
"met","thalach","thalrest","tpeakbps","tpeakbpd","dummy","trestbpd","exang","xhypo","oldpeak",
"slope","rldv5","rldv5e","ca","restckm","exerckm","restef","restwm","exeref","exerwm",
"thal","thalsev","thalpul","earlobe","cmo","cday","cyr","num","lmt","ladprox",
"laddist","diag","cxmain","ramus","om1","om2","rcaprox","rcadist","lvx1","lvx2",
"lvx3","lvx4","lvf","cathef","junk","name" ]

df_raw = pd.read_csv("../input/data regex cleaned.csv", names=header_list, header=None)
df = df_raw.replace(-9, np.NaN)
print("Num cols 0 = %d" % len(df.columns))
author_specified_unused_cols = ['dummy', 'restckm', 'exerckm', 'thalsev', 'thalpul', 'earlobe', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk']
my_specified_useless_cols = ['id', 'ccf', 'pncaden', 'thaldur', 'name']
suspicious_cols = ['lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist']
cols_to_drop = author_specified_unused_cols + my_specified_useless_cols + suspicious_cols
df.drop(columns=cols_to_drop, inplace=True)
print("Num cols 1 = %d" % len(df.columns))
df.dropna(axis='columns', thresh=0.6*len(df.index), inplace=True)
print("Num cols 2 = %d" % len(df.columns))
df_raw.describe(include='all').head(10)
df.describe(include='all')
df[df.drop(columns=['thaltime']).isnull().any(axis='columns')].head(10)
df.describe(include='all')
df.loc[df.cigs.isnull() & df.years > 0, ['cigs']] = df.loc[df.cigs.isnull() & df.years > 0, ['cigs']].fillna({'cigs': df.cigs.mean()})
df.loc[df.years.isnull() & df.cigs > 0, ['years']] = df.loc[df.years.isnull() & df.cigs > 0, ['years']].fillna({'years': df.years.mean()})
df.loc[df.cigs.isnull() & df.years == 0, ['cigs']] = df.loc[df.cigs.isnull() & df.years == 0, ['cigs']].fillna({'cigs': 0})
df.loc[df.years.isnull() & df.cigs == 0, ['years']] = df.loc[df.years.isnull() & df.cigs == 0, ['years']].fillna({'years': 0})
# cigs = nan, years = nan are still left
df[['cigs', 'years', 'thaltime']] = df[['cigs', 'years', 'thaltime']].fillna(df[['cigs', 'years', 'thaltime']].mean())

df[['cigs']].plot.hist()
df[['years']].plot.hist()
df2 = df[:]

df2['dig'] = df['dig'].fillna(df['dig'].value_counts().idxmax())
df2['prop'] = df['prop'].fillna(df['prop'].value_counts().idxmax())
df2['nitr'] = df['nitr'].fillna(df['nitr'].value_counts().idxmax())
df2['pro'] = df['pro'].fillna(df['pro'].value_counts().idxmax())
df2['diuretic'] = df['diuretic'].fillna(df['diuretic'].value_counts().idxmax())

df2[df2.drop(columns=['thaltime']).isnull().any(axis='columns')]
df3 = df2[:]

df3.dropna(subset=['thal', 'ca'], axis='rows', inplace=True)

df3[df3.isnull().any(axis='columns')]

df3.describe()
df3.num.plot.hist()
df3.num = df3.num.apply(lambda x: 1 if x > 0 else 0)
df3.num.plot.hist()

X_train, X_test, y_train, y_test = train_test_split(
    df3.drop(columns=['num']), df3.num, test_size=0.25, random_state=1234)
def draw_roc(y_test, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
acc_dict = {}
def evaluate(classifier, drawTree = False):
    print("Evaluating {}".format(type(classifier).__name__))
    cl = classifier
    cl.fit(X_train, y_train)
    cl_pred = cl.predict(X_test)
    cl_confusion_matrix = confusion_matrix(y_test, cl_pred)
    tn, fp, fn, tp = cl_confusion_matrix.ravel()
    acc = accuracy_score(y_test, cl_pred)
    acc_dict[type(classifier).__name__] = acc
    
    print("Confusion matrix")
    print(cl_confusion_matrix)
    print("Accuracy %f" % acc)
    print("Recall %f" % recall_score(y_test, cl_pred))
    print("False alarm %f\n" % (fp / (fp+tn)))
    print("ROC curve")
    draw_roc(y_test, cl_pred)
    
    if drawTree and isinstance(classifier, tree.DecisionTreeClassifier):
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=list(X_train),
                                        class_names=["healthy", "ill"],
                                        filled=True,
                                        special_characters=True) 
        graph = graphviz.Source(dot_data)
        display.display(graph)
models = [tree.DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(n_neighbors=3), LogisticRegression()]
for m in models:
    evaluate(m)

plt.bar(range(len(acc_dict)), list(acc_dict.values()), tick_label=list(acc_dict.keys()))
plt.show()

y_pred = KMeans(n_clusters=2, random_state=1234).fit_predict(X_train)
y_pred = 1 - y_pred
tmpdf = pd.DataFrame({'real': y_train, 'clust': y_pred})
tmpdf['eq'] = np.where(tmpdf['real'] == tmpdf['clust'], 1, 0)
tmpdf['eq'].value_counts()
df3.describe()
#asoc = apriori(df3, min_support=0.1, use_colnames=True)
df_aso = df3.copy()

df_aso = df_aso.drop(columns=['cp', 'trestbps', 'restecg', 'ekgday', 'ekgmo', 'ekgyr', 'thaltime', 'met', 'thalach',
                     'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'oldpeak', 'rldv5e', 'ca', 'cmo',
                    'cday', 'cyr', 'trestbps', 'thal', 'years'])

df_aso.loc[:, 'age'] = df_aso['age'].apply(lambda x: 1 if x > 54.0 else 0)
df_aso.loc[:,'chol'] = df_aso['chol'].apply(lambda x: 1 if x > 277.0 else 0)
df_aso.loc[:,'cigs'] = df_aso['cigs'].apply(lambda x: 1 if x > 1.0 else 0)
df_aso.loc[:,'slope'] = df_aso['slope'].apply(lambda x: 1 if x > 0.0 else 0)
df_aso.describe()
aso_rules_apriori = apriori(df_aso, min_support=0.3, use_colnames=True)
rules = association_rules(aso_rules_apriori, metric='lift', min_threshold=1)
rules.loc[(rules.consequents.str.contains('num', regex=False))]