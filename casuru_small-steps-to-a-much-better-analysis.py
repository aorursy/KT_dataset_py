# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import BernoulliNB

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
frame = pd.read_csv('../input/cleanedmrdata.csv')
frame = frame.dropna()

frame.head()
percents = frame.iloc[:, 7:].mean() * 100

plt.bar(range(len(percents)), percents)

plt.title("Blog Post Tags")

plt.ylabel("Percentage of Blog Post With Tag")

plt.gca().set_xticklabels(percents.index)

plt.gca().set_xticks(np.arange(len(percents)) + .45)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10, 8))

t_percents = frame[frame['author'] == 'Tyler Cowen'].iloc[:, 7:].mean() * 100

a_percents = frame[frame['author'] == 'Alex Tabarrok'].iloc[:, 7:].mean() * 100

labels = frame[frame['author'] == 'Tyler Cowen'].iloc[:, 7:].mean().index

t_color = np.random.rand(3)

a_color = np.random.rand(3)

handles = [patches.Patch(label='Alex Tabarook', color=a_color), patches.Patch(label='Tyler Cowen', color=t_color)]

ind = np.arange(len(t_percents))

plt.bar(ind, t_percents, width=.45, color=t_color)

plt.bar(ind+.45, a_percents, width=.45, color=a_color)

plt.gca().set_xticklabels(labels)

plt.gca().set_xticks(ind + .45)

plt.legend(handles=handles)

plt.xticks(rotation=90)

plt.title("Blog Post Tags")

plt.ylabel("Percentage of Blog Post With Tag")

plt.show()
sns.boxplot(x='author', y='wordcount', data=frame)

plt.show()
sns.boxplot(x='author', y='comment.count', data=frame)

plt.show()
vectorizer = TfidfVectorizer().fit(frame['text'])

feature_vect = vectorizer.transform(frame['text'])

target_vect = LabelEncoder().fit_transform(frame['author'])
train_features = feature_vect[:8000]

test_features = feature_vect[8000:]

train_targets = target_vect[:8000]

test_targets = target_vect[8000:]
clf = BernoulliNB()

clf.fit(train_features, train_targets)
accuracy_score(test_targets, clf.predict(test_features))
net = MLPClassifier(hidden_layer_sizes = (500, 250))

net.fit(train_features, train_targets)
accuracy_score(test_targets, net.predict(test_features))