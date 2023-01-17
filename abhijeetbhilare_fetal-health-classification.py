import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc, roc_auc_score

from scipy import interp

from itertools import cycle

from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import precision_recall_fscore_support as score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/fetal-health-classification/fetal_health.csv")

print(df.shape)

df.head()
df.info()
df.describe()
sns.countplot(df.fetal_health)
y_orig = df.fetal_health

print(y_orig.unique())

y = label_binarize(y_orig, classes=[1,2,3])

n_classes = 3

# X = df.drop(["fetal_health"], axis=1)

X = df[["baseline value", "accelerations", "fetal_movement", "uterine_contractions", "light_decelerations",

        "severe_decelerations", "prolongued_decelerations", "abnormal_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability"]]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500, random_state=42))

y_score = clf.fit(x_train,y_train)

y_pred = clf.score(x_test,y_test)

print("Validation Accuracy",clf.score(x_test,y_test)*100,"%")
print(f1_score(y_test, clf.predict(x_test), average='macro'))

print(f1_score(y_test, clf.predict(x_test), average='micro'))

print(f1_score(y_test, clf.predict(x_test), average='weighted'))



precision, recall, fscore, support = score(y_test, clf.predict(x_test))



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
y_prob = clf.predict_proba(x_test)

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",

                                  average="macro")

weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",

                                     average="weighted")

macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",

                                  average="macro")

weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",

                                     average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "

      "(weighted by prevalence)"

      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))

print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "

      "(weighted by prevalence)"

      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

lw = 2

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Fetal Health Classification')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score



# For each class

precision = dict()

recall = dict()

average_precision = dict()

for i in range(n_classes):

    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],

                                                        y_prob[:, i])

    average_precision[i] = average_precision_score(y_test[:, i], y_prob[:, i])



# A "micro-average": quantifying score on all classes jointly

precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),

    y_prob.ravel())

average_precision["micro"] = average_precision_score(y_test, y_prob,

                                                     average="micro")

print('Average precision score, micro-averaged over all classes: {0:0.2f}'

      .format(average_precision["micro"]))  
plt.figure()

plt.step(recall['micro'], precision['micro'], where='post')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title(

    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'

    .format(average_precision["micro"]))
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])



plt.figure(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)

lw = 2

lines = []

labels = []

for f_score in f_scores:

    x = np.linspace(0.01, 1)

    y = f_score * x / (2 * x - f_score)

    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)

    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))



lines.append(l)

labels.append('iso-f1 curves')

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)

lines.append(l)

labels.append('micro-average Precision-recall (area = {0:0.2f})'

              ''.format(average_precision["micro"]))



for i, color in zip(range(n_classes), colors):

    l, = plt.plot(recall[i], precision[i], color=color, lw=2)

    lines.append(l)

    labels.append('Precision-recall for class {0} (area = {1:0.2f})'

                  ''.format(i, average_precision[i]))



fig = plt.gcf()

fig.subplots_adjust(bottom=0.25)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Fetal Health Classification')

plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))





plt.show()