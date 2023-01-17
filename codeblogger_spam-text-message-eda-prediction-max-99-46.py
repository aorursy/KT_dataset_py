import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

data.head(10)
data.info()
data.describe().T
def bar_plot(variable):

    var = data[variable]

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Count")

    plt.title(variable)

    plt.show()

    print("{}: \n{}".format(variable,varValue))
bar_plot("Category")
ham_message_length = []

spam_message_length = []

for i in data.values:

    if(i[0] == "ham"):

        ham_message_length.append(len(i[1]))

    else:

        spam_message_length.append(len(i[1]))

        

# average

ham_average = sum(ham_message_length)/len(ham_message_length)

spam_average = sum(spam_message_length)/len(spam_message_length)

print("ham_average: ", ham_average)

print("spam_average: ", spam_average)
plt.figure(figsize=(9,3))

plt.bar(["ham_average","spam_average"], [ham_average,spam_average])

plt.show()
data.isnull().sum()
from sklearn.model_selection import train_test_split

import re

from nltk.corpus import stopwords

import nltk as nlp

from sklearn.feature_extraction.text import CountVectorizer 
data.Category = [1 if each == "ham" else 0 for each in data.Category]

df = data

data
# for example



example = data.Message[0]

print("before: ", example)



example = re.sub("[^a-zA-Z]"," ",example)

print("after:", example)
regularExpressionMessages = []

for message in data["Message"]:

    message = re.sub("[^a-zA-Z]"," ",message)

    regularExpressionMessages.append(message)



data["Message"] = regularExpressionMessages

data
# for example



example = data.Message[0]

print("before: ", example)



example = example.lower()

print("after:", example)
lowercaseMessages = []

for message in data["Message"]:

    message = message.lower()

    lowercaseMessages.append(message)



data["Message"] = lowercaseMessages

data
# for example



example = data.Message[0]

print("before: ", example)



example = nlp.word_tokenize(example)

print("after:", example)
splitMessages = []

for message in data["Message"]:

    message = nlp.word_tokenize(message)

    splitMessages.append(message)



data["Message"] = splitMessages

data
# for example



print("before: ", data["Message"][0] )



message1 = [message for message in data["Message"][0] if not message in set(stopwords.words("english"))]

print("after: ", message1)
stopwordsMessages = []

for i in data["Message"]:

    i = [message for message in i if not message in set(stopwords.words("english"))]

    stopwordsMessages.append(i)



data["Message"] = stopwordsMessages

data
# for example



example = data.Message[6]

print("before: ", example)



lemma = nlp.WordNetLemmatizer()

example = [ lemma.lemmatize(word) for word in example]

print("after:", example)
example = " ".join(example)

print("example: ", example)
joinMessages = []

for message in data["Message"]:

    message = " ".join(message)

    joinMessages.append(message)



data["Message"] = joinMessages

data
# for example

sentence1 = "I am coming from school"

sentence2 = "I am coming from Istanbul today"



from sklearn.feature_extraction.text import CountVectorizer 

count_vectorizer = CountVectorizer(max_features=5, stop_words="english")

sparce_matrix = count_vectorizer.fit_transform([sentence1,sentence2]).toarray()

sparce_matrix
count_vectorizer = CountVectorizer(max_features=10000, stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(np.array(data["Message"])).toarray()

sparce_matrix
sparce_matrix.shape
# text classification



y = data.iloc[:,0].values

x = sparce_matrix
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)



print("x_train",x_train.shape)

print("x_test",x_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)



print("Accuracy: ", nb.score(x_test,y_test)*100)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
logreg = LogisticRegression()

logreg.fit(x_train,y_train)

acc_log_train = round(logreg.score(x_train,y_train)*100,2)

acc_log_test = round(logreg.score(x_test,y_test)*100,2)

print("Training Accuracy: %{}".format(acc_log_train))

print("Testing Accuracy: %{}".format(acc_log_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
from sklearn.svm import SVC



svm = SVC(random_state=1)

svm.fit(x_train,y_train)



print("accuracy of svm algo: ", svm.score(x_test,y_test)*100)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train,y_train)



print("Decision Tree Score: ", dt.score(x_test,y_test)*100)