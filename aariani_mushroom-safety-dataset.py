# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

sns.set(font_scale = 2)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')

data.head()
# simple preprocessing of the dataset

target_var = data['class']

data.drop('class', axis = 1, inplace=True)

y = LabelEncoder().fit_transform(target_var) # p == 1, e == 0

data.head()
sns.set_style('whitegrid')

target_dist = target_var.value_counts()

sns.barplot(target_dist.index, target_dist.values)

plt.title('Target variable distribution');
sns.set_style('whitegrid')

odor = pd.DataFrame({'p':y, 'odor':data['odor']})

# count the 1 (poison) occurence on the dataset per odor

odor_poison = odor.groupby('odor').sum()

# get the count per odor

odor_poison_size = odor.groupby('odor').size()

odor_data = pd.DataFrame({'P': odor_poison['p']/odor_poison_size, 

                          'E':(odor_poison_size-odor_poison['p'])/odor_poison_size})

print(odor_data.head())

sns.set_style('whitegrid')

plt.figure(figsize=(12,6))

sns.set(font_scale = 2)

sns.set(style='whitegrid', context='notebook')

sns.heatmap(odor_data.T*100, fmt = '.1f', cmap = 'plasma', cbar = True, annot = True, 

            linewidth = 2, yticklabels=('Edible', 'Poisonous'))

plt.yticks(rotation=0)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin



class OdorDecision(BaseEstimator, TransformerMixin):

    def __init__(self, non_poison=['a', 'l', 'n']):

        self.non_poison = non_poison

    def fit(self, X, y=None):

        return self

    def predict(self, X):

        pred = [0 if i in self.non_poison else 1 for i in X]

        return pred

    

X = data['odor'].values

od = OdorDecision()

pred = od.predict(X)

print(classification_report(y, pred, target_names = ['edible', 'poisonous']))

print('Accuracy: ', accuracy_score(y, pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y, pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Odor rule',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()

from sklearn.model_selection import cross_val_predict

model_accuracy = []



caps = data[['cap-shape', 'cap-surface', 'cap-color']]

X_dum = pd.get_dummies(caps).values



X_train, X_test, y_train, y_test= train_test_split(X_dum, y, stratify=y,

                                                  test_size=0.2, random_state=101)



rfc_caps = RandomForestClassifier(n_estimators=100, random_state=42)

rfc_caps.fit(X_train, y_train)



# test in sample precision with CV



y_train_pred = cross_val_predict(rfc_caps, X_train, y_train, cv = 5)



print(classification_report(y_train, y_train_pred, target_names = ['edible', 'poisonous']))

print('Accuracy: ', accuracy_score(y_train, y_train_pred))

model_accuracy.append(accuracy_score(y_train, y_train_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_train, y_train_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Mushroom Cap',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
gills = data[['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']]



X_gills = pd.get_dummies(gills).values



X_train, X_test, y_train, y_test = train_test_split(X_gills, y, stratify=y,

                                                   test_size = 0.2, random_state = 101)



rfc_gills = RandomForestClassifier(n_estimators=100, random_state=42)



rfc_gills.fit(X_train, y_train)



# In sample precision with CV

y_train_pred = cross_val_predict(rfc_gills, X_train, y_train, cv = 5)

print(classification_report(y_train, y_train_pred))

print('Accuracy: ', accuracy_score(y_train, y_train_pred))

model_accuracy.append(accuracy_score(y_train, y_train_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_train, y_train_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Mushroom Gills',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
stalks = data[['stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring']]



X_stalks = pd.get_dummies(stalks).values



X_train, X_test, y_train, y_test = train_test_split(X_stalks, y, stratify=y,

                                                   test_size=0.2, random_state=101)



rfc_stalks = RandomForestClassifier(n_estimators=100, random_state=42)

rfc_stalks.fit(X_train, y_train)



# in sample

y_train_pred = cross_val_predict(rfc_stalks, X_train, y_train, cv = 5)

print(classification_report(y_train, y_train_pred))

print('Accuracy: ', accuracy_score(y_train, y_train_pred))

model_accuracy.append(accuracy_score(y_train, y_train_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_train, y_train_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Mushroom Stalks',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
veil = data[['veil-type', 'veil-color']]



X_veil = pd.get_dummies(veil).values



X_train, X_test, y_train, y_test = train_test_split(X_veil, y, stratify=y,

                                                   test_size=0.2, random_state=101)



rfc_veil = RandomForestClassifier(n_estimators=100, random_state=42)

rfc_veil.fit(X_train, y_train)



# in sample

y_train_pred = cross_val_predict(rfc_veil, X_train, y_train, cv = 5)

print(classification_report(y_train, y_train_pred))

print('Accuracy: ', accuracy_score(y_train, y_train_pred))

model_accuracy.append(accuracy_score(y_train, y_train_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_train, y_train_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Mushroom Veil',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
rings = data[['ring-number','ring-type']]



X_ring = pd.get_dummies(rings)



X_train, X_test, y_train, y_test = train_test_split(X_ring, y, stratify=y,

                                                   test_size=0.2, random_state=101)



rfc_ring = RandomForestClassifier(n_estimators=100, random_state=42)

rfc_ring.fit(X_train, y_train)



y_train_pred = cross_val_predict(rfc_ring, X_train, y_train, cv = 5)

print(classification_report(y_train, y_train_pred))

print('Accuracy: ', accuracy_score(y_train, y_train_pred))

model_accuracy.append(accuracy_score(y_train, y_train_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_train, y_train_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Mushroom Rings',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
part_names=['Caps', 'Gills','Stalks', 'Veils', 'Rings']



sns.set(font_scale = 2)

sns.set_style('whitegrid')

sns.barplot(part_names, model_accuracy)#, kwargs={'fontsize':18})

plt.ylabel('Accuracy Score')

plt.title('Mushroom Parts Evaluation', fontsize = 20);
combined_data = data[['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

                     'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

                      'stalk-surface-below-ring', 'stalk-color-above-ring',

                      'stalk-color-below-ring','ring-number','ring-type']]



X_comb = pd.get_dummies(combined_data)



X_train, X_test, y_train, y_test = train_test_split(X_comb, y, stratify=y,

                                                   test_size=0.2, random_state=101)



rf_comb = RandomForestClassifier(n_estimators=100, random_state=42)

rf_comb.fit(X_train, y_train)



y_pred = rf_comb.predict(X_test)

print(classification_report(y_test, y_pred))

print('Accuracy: ', accuracy_score(y_test, y_pred))

plt.figure(figsize = (6,6))

sns.heatmap(confusion_matrix(y_test, y_pred),

           cmap = 'plasma', annot = True,

            fmt = '.1f', cbar = True,linewidth = 2, 

            yticklabels=('Edible', 'Poisonous'),

            xticklabels=('Edible', 'Posonous')

           )

plt.title('Confusion Matrix',fontsize=20)

plt.yticks(rotation=0)

plt.xlabel('Predicted', fontsize=20)

plt.ylabel

plt.show()
combined_names = ['Gills', 'Stalks', 'Rings']

n = [part_names.index(i) for i in combined_names]

model_acc = [model_accuracy[i] for i in n]



model_acc.append(accuracy_score(y_test, y_pred))

combined_names.append('Combined')

sns.barplot(combined_names, model_acc)#, kwargs={'fontsize':18})

plt.ylabel('Accuracy Score')

plt.title('Mushroom Parts Evaluation', fontsize = 20);
# create subsets

from scipy.stats import mode

gills = data[['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']]

stalks = data[['stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

                      'stalk-surface-below-ring', 'stalk-color-above-ring',

                      'stalk-color-below-ring']]

rings = data[['ring-number','ring-type']]



rf_gills = RandomForestClassifier(n_estimators=100, random_state=42)

rf_stalks = RandomForestClassifier(n_estimators=100, random_state=42)

rf_rings = RandomForestClassifier(n_estimators=100, random_state=42)



# create function



def majorityClassifier(model, data, y):

    ys = []

    pred_class = []

    for clf, d in zip(model, data): # run in parallels model and data

        X_dum = pd.get_dummies(d)

        X_train, X_test, y_train, y_test = train_test_split(X_dum, y, stratify=y,

                                                   test_size=0.2, random_state=101)

        ys.append(y_test)

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)

        pred_class.append(preds)

    pred_class = np.array(pred_class).T

    maj_vote = mode(pred_class, axis = 1)[0]

    return maj_vote,ys

preds,y_all = majorityClassifier([rf_gills, rf_stalks, rf_rings],

                      [gills, stalks, rings], y)
print(classification_report(y_test, preds))