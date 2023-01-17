# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
c_encoder = LabelEncoder().fit(train['color'])

t_encoder = LabelEncoder().fit(train['type'])
train['color'] = c_encoder.transform(train['color'])

train['type'] = t_encoder.transform(train['type'])

test['color'] = c_encoder.transform(test['color'])
train.shape
colors = {ii: np.random.rand(3) for ii in set(train['type'])}



def sub_scatter(f1, f2, i):

    plt.subplot(2,2, i)

    plt.title("{0} v {1}".format(f1, f2))

    plt.xlabel(f1)

    plt.ylabel(f2)

    handles = []

    for type_ in set(train['type']):

        df = train[train['type'] == type_]

        x = df[f1]

        y = df[f2]

        handles.append(patches.Patch(color = colors[type_], label=t_encoder.inverse_transform(type_)))

        plt.scatter(x, y, color=colors[type_])

        plt.legend(handles=handles)

    
plt.figure(figsize=(12, 10))

sub_scatter('bone_length', 'rotting_flesh', 1)

sub_scatter('bone_length', 'hair_length', 2)

sub_scatter('bone_length', 'has_soul', 3)

sub_scatter('rotting_flesh', 'hair_length', 4)



plt.show()
plt.figure(figsize=(12, 10))

sub_scatter('rotting_flesh', 'has_soul', 1)

sub_scatter('has_soul', 'hair_length', 2)



plt.show()
feats = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']

train_attrs = train[feats].as_matrix()

train_labels = train['type'].as_matrix()
clf = GaussianNB()

clf.fit(train_attrs, train_labels)
test_attrs = test[feats].as_matrix()
subs = pd.DataFrame({'id': test['id'], 'type':t_encoder.inverse_transform(clf.predict(test_attrs))})
subs.to_csv('gg_submission.csv', index=False)