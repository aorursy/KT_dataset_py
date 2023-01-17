# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns


color = sns.color_palette()

%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.head())
for i in train.columns:

    print (i,len(train[i]),train[i].notnull().sum(),(train[i].notnull().sum())/len(train[i]))
for i in train.columns:

    print(i,train[i].isnull().sum())
df = train.dropna()
print (df)
df.head()
print('Total number of question pairs for training: {}'.format(len(df)))
print('Duplicate pairs: {}%'.format(round(df['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()

is_dup = df['is_duplicate'].value_counts()

plt.figure(figsize=(8,4))

sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Duplicate', fontsize=12)

plt.show()

is_dup / is_dup.sum()
data = df[:5000]
print(data)
data['is_duplicate'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
data.head()
print(data)
dummies =  pd.get_dummies(data,columns=['qid1','qid2','question1','question2'])
dummies.head()
y=dummies['is_duplicate']
final =dummies.drop(['id','is_duplicate'],axis='columns')
y.head()

feature_cols=final.columns

print(feature_cols)
X= final[feature_cols]
print(X.head())
print("*********************************************************")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# print the size of the traning set:
print(X_train.shape)
print(y_train.shape)

# print the size of the testing set:
print(X_test.shape)
print(y_test.shape)