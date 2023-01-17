import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import xgboost as xgb

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from itertools import combinations

from numpy import array,array_equal

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import classification_report, confusion_matrix

from xgboost import plot_tree

from matplotlib.pylab import rcParams
train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv")
test_id = test.ID
df = pd.DataFrame(train.TARGET.value_counts())

df['%'] = 100 * df['TARGET']/train.shape[0]

df
train = train.loc[:, (train != train.iloc[0]).any()]

test = train.loc[:, (train != train.iloc[0]).any()]
def remove_ducplicated_column(dataset):

    remove = []

    cols = dataset.columns

    for i in range(len(cols)-1):

        v = dataset[cols[i]].values

        for j in range(i+1,len(cols)):

            if np.array_equal(v,dataset[cols[j]].values):

                remove.append(cols[j])

    dataset.drop(remove, axis=1, inplace=True)
remove_ducplicated_column(train) # delete duplicate columns for trains

remove_ducplicated_column(test) # -- for test
X_train = train.iloc[:,:-1]

y_train = train.TARGET



X_test = test.iloc[:,:-1]

y_test = test.TARGET
# Create a random forest classifier

clf = RandomForestClassifier(random_state=0, n_jobs=-1)



# Train the classifier

clf.fit(X_train, y_train)



# Print the name and gini importance of each feature

for feature in zip(X_train.columns, clf.feature_importances_):

    print(feature)
# Create a selector object that will use the random forest classifier to identify

# features that have an importance of more than 0.15

sfm = SelectFromModel(clf, threshold=0.002)



# Train the selector

sfm.fit(X_train, y_train)



# Print the names of the most important features

print(len(sfm.get_support(indices=True)), "features selected")

for feature_list_index in sfm.get_support(indices=True):

    print(X_train.columns[feature_list_index], clf.feature_importances_[feature_list_index])
X_important_train = sfm.transform(X_train)

X_important_test = sfm.transform(X_test)
def confusio_matrix(y_test, y_predicted):

  cm = confusion_matrix(y_test, y_predicted)

  plt.figure(figsize=(15,10))

  plt.clf()

  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

  classNames = ['Negative','Positive']

  plt.title('Matrice de confusion')

  plt.ylabel('True label')

  plt.xlabel('Predicted label')

  tick_marks = np.arange(len(classNames))

  plt.xticks(tick_marks, classNames, rotation=45)

  plt.yticks(tick_marks, classNames)

  s = [['TN','FP'], ['FN', 'TP']]

  

  for i in range(2):

      for j in range(2):

          plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

  plt.show()
m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4, seed=1729)

m2_xgb.fit(X_important_train, y_train, eval_metric="auc", verbose = False, eval_set=[(X_important_test, y_test)])
y_predicted = m2_xgb.predict(X_important_test)
y_predicted
print(len(y_test))
confusio_matrix(y_test,y_predicted)
rcParams['figure.figsize'] = 50,20

plot_tree(m2_xgb)

plt.show()
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from inspect import signature



precision, recall, _ = precision_recall_curve(y_test, y_predicted)

print(precision, recall)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2, where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# calculate the auc score

print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict_proba(X_important_test)[:,1], average='macro'))
## # Submission

probs = m2_xgb.predict_proba(X_test)



#submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})

#submission.to_csv("submission.csv", index=False)