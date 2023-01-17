# loading required libraries for cancer treament analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To ignore warinings

import warnings

warnings.filterwarnings('ignore')
# let's check in which directory our data is available so that it will be easy to pull from specific source location

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# loading datasets

train_variants = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants')

test_variants = pd.read_csv('../input/msk-redefining-cancer-treatment/test_variants')

train_text = pd.read_csv('../input/msk-redefining-cancer-treatment/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text = pd.read_csv('../input/msk-redefining-cancer-treatment/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_variants.head()
train_text.head()
train_merge = pd.merge(train_variants,train_text,how='left',on='ID')

# let's pull train merge dataset and do the analysis on this

train_merge.head()
# Let's understand the type of values present in each column of our dataframe 'train_merge' dataframe.

train_merge.info()
# Histogram : To check class distribution

plt.figure(figsize=(12,8))

sns.countplot(x='Class',data=train_variants)

plt.ylabel('Frequency-Counts', fontsize=15)

plt.xlabel('Class',fontsize=13)

plt.xticks(rotation='vertical')

plt.title('Class Counts',fontsize=15)

plt.show()
train_merge["Text_num_words"] = train_merge["Text"].apply(lambda x: len(str(x).split()) )

train_merge["Text_num_chars"] = train_merge["Text"].apply(lambda x: len(str(x)) )
plt.figure(figsize=(12, 8))

sns.distplot(train_merge.Text_num_words.values, bins=50, kde=False, color='red')

plt.xlabel('Number of words in text', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title("Frequency of number of words", fontsize=15)

plt.show()
plt.figure(figsize=(12, 8))

sns.distplot(train_merge.Text_num_chars.values, bins=50, kde=False, color='brown')

plt.xlabel('Number of characters in text', fontsize=12)

plt.ylabel('log of Count', fontsize=12)

plt.title("Frequency of Number of characters", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x='Class', y='Text_num_words', data=train_merge)

plt.xlabel('Class', fontsize=12)

plt.ylabel('Text - Number of words', fontsize=12)

plt.show()
train_merge.describe()
# putting respon variable to y

#y = train_merge['Class'].values

#train_merge = train_merge.drop('Class',axis=1)
train_merge.head(3)
#y
test_merge = pd.merge(test_variants,test_text,how='left',on='ID')

test_merge.head(3)
pid = test_merge['ID'].values

pid
test_merge.describe()
# check total number of null/missing value present in whole datasets

train_merge.isnull().sum()
# find out percentage of "?" value present across the dataset

percent_missing = train_merge.isnull().sum() * 100 / len(train_merge)

percent_missing
# droping missing values

train_merge.dropna(inplace=True)



# let's check again whether we have any further missing values

train_merge.isnull().sum()
test_merge.isnull().sum()
# dropping missing values

test_merge.dropna(inplace=True)



# check if our data is clean or not

test_merge.isnull().sum()
from sklearn.model_selection import train_test_split

train,test = train_test_split(train_merge,test_size=0.2)

np.random.seed(0)

train
X_train = train['Text'].values

X_test = test['Text'].values

y_train = train['Class'].values

y_test = test['Class'].values
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm
text_classifier = Pipeline([('vect', CountVectorizer()),

                     ('tfidf', TfidfTransformer()),

                     ('clf', svm.LinearSVC())

])

text_classifier = text_classifier.fit(X_train,y_train)
y_test_predicted = text_classifier.predict(X_test)

np.mean(y_test_predicted == y_test)
X_test_final = test_merge['Text'].values

#X_test_final
predicted_class = text_classifier.predict(X_test_final)
test_merge['predicted_class'] = predicted_class
test_merge.head(5)
onehot = pd.get_dummies(test_merge['predicted_class'])

test_merge = test_merge.join(onehot)
test_merge.head(5)
submission_df = test_merge[["ID",1,2,3,4,5,6,7,8,9]]

submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']

submission_df.head(5)
submission_df.to_csv('submission.csv', index=False)