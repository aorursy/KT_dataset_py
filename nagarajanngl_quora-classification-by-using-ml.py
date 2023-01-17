# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation

#loading csv file
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
print("Train Shape",train_data.shape)
print("Test Shape",test_data.shape)
train_data.head()
test_data.head()
#data information & description for train data
print("\n train data info")
train_data.info()
print("\n train data description")
print(train_data.describe())
#data information & description for test data
print("\n test data info")
test_data.info()
print("\n test data description")
print(test_data.describe())
# 1 - Filtering target values of Pandas train data frame
#calculating target 0&1 counts for train data
x = train_data['target'].value_counts()[0]
print("train data target 0 count is",x)
y = train_data['target'].value_counts()[1]
print("train data target 1 count is",y)
print('{}% of the questions in the train set are tagged as insincere.'.format((y*100/(x + y)).round(2)))
#plotting fig
fig, ax = plt.subplots()
g = sns.countplot(train_data.target)
g.set_xticklabels(['Sincere(0)', 'Insincere(1)'])
g.set_yticklabels([''])
sincere = train_data[train_data["target"] == 0]
insincere = train_data[train_data["target"] == 1]
# function to show sincere and insincere on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Questions', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
plt.show()
#finding Question length for training data
train_data["quest_len"] = train_data["question_text"].apply(lambda x: len(x.split()))
print("\nTrain data with new column quest_length")
print(train_data.head())
max_ql_train = train_data["quest_len"].max()
print("maximum question length is",max_ql_train)
min_ql_train=train_data["quest_len"].min()
print("minimum question length is",min_ql_train)

#finding Question length for test data
test_data["quest_len"] = test_data["question_text"].apply(lambda x: len(x.split()))
print("\nTest data with new column quest_length")
print(test_data.head())
max_ql_test = test_data["quest_len"].max()
print("maximum question length is",max_ql_test)
min_ql_test=test_data["quest_len"].min()
print("minimum question length is",min_ql_test)
#Finding stop_words
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

def clean_text(question):
    question = question.lower()
    question = re.sub("\\n", "", question)
    question = re.sub("\'", "", question)
    words = tokenizer.tokenize(question)
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in stop_words and w not in punctuations]
    clean_sent = " ".join(words)
    return clean_sent
train_data["clean_question_text"] = train_data["question_text"].apply(lambda question: clean_text(question))
test_data["clean_question_text"] = test_data["question_text"].apply(lambda question: clean_text(question))
print(train_data.head())
print(test_data.head())

train_question_list = train_data["clean_question_text"]
test_question_list = test_data["clean_question_text"]
#CountVectorizer--converting text document into spare matrix
vectorizer  = CountVectorizer()
x_train =  vectorizer.fit_transform(train_question_list)
x_test =  vectorizer.transform(test_question_list)
y_train_tfidf = np.array(train_data["target"].tolist())
train_x, validate_x, train_y, validate_y = train_test_split(x_train, y_train_tfidf, test_size=0.3)

#Applying multinomial LogisticRegression Algorithm
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='sag').fit(train_x, train_y)
y_vad = mul_lr.predict(validate_x)

#calculating accuracy for logistic regression algorithm
print('accuracy = %.2f%%' % \
      (accuracy_score(validate_y, y_vad)*100))
#target prediction
y_predict = mul_lr.predict(x_test)
predict = pd.DataFrame(data = y_predict, columns=['prediction'])
predict = predict.astype(int)
id = test_data['qid']
id_df = pd.DataFrame(id)
# Join predicted into result dataframe and write result as a CSV file
result = id_df.join(predict)
result.to_csv("submission.csv", index = False)

result.head()