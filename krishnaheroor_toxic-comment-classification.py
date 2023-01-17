# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

df.info()
df.head()
# dropping id and comment text

df_toxic = df.drop(['id', 'comment_text'], axis=1)

# calculating total count of each category comments

counts = []

categories = list(df_toxic.columns.values)

for i in categories:

    counts.append((i, df_toxic[i].sum()))

df_stats = pd.DataFrame(counts, columns=['category', 'count'])

df_stats
# df_toxic.sum().plot(kind="bar")



sns.set(style="whitegrid")

sns.barplot(x='category', y='count', data=df_stats, palette="summer")

plt.title("Number Of Comments For Each Tag")

plt.show()
rowsums = df_toxic.iloc[:,:].sum(axis=1)

valcount = rowsums.value_counts()

valcount.plot.bar()

plt.xlabel("# of labels tagged to")

plt.ylabel("# of comments")

plt.title("Comments that have multiple labels tagged")

plt.show()



print(valcount[0]*100/sum(valcount),"% comments have no labels associated to them.")
lens = df.comment_text.str.len()

sns.distplot(lens)

plt.title("Distribution for Lengths of Comments")

plt.show()
print("# Of Vacant Comments : ", df['comment_text'].isnull().sum())
df['comment_text'][0]
sns.heatmap(df.corr(), square=True, cmap='nipy_spectral')

plt.show()
df.to_pickle('cleaned_data.pkl')
from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot
df = pd.read_pickle('cleaned_data.pkl')

df.head()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


train, test = train_test_split(df, test_size=0.33, random_state=42, shuffle=True)
train.shape, test.shape
X_train = train['comment_text']

X_test = test['comment_text']
accuracies = [[],[],[]]
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

cvec = CountVectorizer()

tvec = TfidfTransformer()

model1 = MultinomialNB()
X_train = tvec.fit_transform(cvec.fit_transform(X_train))

X_test = tvec.transform(cvec.transform(X_test))
X_train.shape, X_test.shape


for category in labels:

    model1.fit(X_train, train[category])

    accuracy = model1.score(X_test, test[category])

    accuracies[0].append(accuracy)

    print("Accuracy For {0} Class Is {1}%".format(category,round(accuracy*100,2)))


from sklearn.svm import LinearSVC

model2 = LinearSVC()

for category in labels:

    model2.fit(X_train, train[category])

    accuracy = model2.score(X_test, test[category])

    accuracies[1].append(accuracy)

    print("Accuracy For {0} Class Is {1}%".format(category,round(accuracy*100,2)))
from sklearn.linear_model import LogisticRegression

model3 = LogisticRegression(n_jobs=1, solver='liblinear')

for category in labels:

    model3.fit(X_train, train[category])

    accuracy = model3.score(X_test, test[category])

    accuracies[2].append(accuracy)

    print("Accuracy For {0} Class Is {1}%".format(category,round(accuracy*100,2)))
accuracies = pd.DataFrame(accuracies)

fig = accuracies.plot.bar(figsize=(16, 5), grid=True)

plt.xticks(np.arange(3),('Multinomial Naive Bayes','Linear Support Vector Classifier','Logistic Regression'),rotation=0)

plt.legend(labels)

plt.show()
for i in range(3):

    print("Model -",i+1,"... Aggregate Accuracy -",np.mean(accuracies.iloc[i,:]))