# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'retina'



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/spam.csv",encoding='latin-1')
df.head()
#Drop column and name change

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df = df.rename(columns={"v1":"label", "v2":"text"})

data=df
data.head(3)
data.label.value_counts()
plt.figure(figsize = (3,3))

sns.countplot(df['label'])
dic={'ham':1 , 'spam':0}

data['label']=data.label.map(dic)
data.head(3)
from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)

spam=data[data.label ==0]

ham=data[data.label ==1]

X_train_spam,X_test_spam,y_train_spam,y_test_spam = train_test_split(spam["text"],spam["label"], test_size = 0.2, random_state = 25)

X_train_ham,X_test_ham,y_train_ham,y_test_ham= train_test_split(ham["text"],ham["label"], test_size = 0.3, random_state = 25)

fm_tn=[X_train_spam,X_train_ham]

X_train=pd.concat(fm_tn)

fm_ts=[X_test_spam,X_test_ham]

X_test=pd.concat(fm_ts)

fm_lb_tn=[y_train_spam,y_train_ham]

y_train=pd.concat(fm_lb_tn)

fm_lb_ts=[y_test_spam,y_test_ham]

y_test=pd.concat(fm_lb_ts)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(X_train)
print(vect.get_feature_names()[0:20])

print(vect.get_feature_names()[-20:])
X_train_df = vect.transform(X_train)

X_test_df = vect.transform(X_test)
ham_words = ''

spam_words = ''

spam = data[data.label == 1]

ham = data[data.label==0]
import nltk

from nltk.corpus import stopwords
for val in spam.text:

    text = val.lower()

    tokens = nltk.word_tokenize(text)

    #tokens = [word for word in tokens if word not in stopwords.words('english')]

    for words in tokens:

        spam_words = spam_words + words + ' '

        

for val in ham.text:

    text = val.lower()

    tokens = nltk.word_tokenize(text)

    for words in tokens:

        ham_words = ham_words + words + ' '
# Generate a word cloud image

from wordcloud import WordCloud

spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)

ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)
#Spam Word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Ham word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
prediction = dict()

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_df,y_train)
prediction["Multinomial"] = model.predict(X_test_df)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,prediction["Multinomial"])
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_df,y_train)

prediction["Logistic"] = model.predict(X_test_df)
accuracy_score(y_test,prediction["Logistic"])
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_df,y_train)

prediction["knn"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["knn"])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train_df,y_train)

prediction["random_forest"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["random_forest"])
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()

model.fit(X_train_df,y_train)

prediction["adaboost"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["adaboost"])
print(classification_report(y_test, prediction["Multinomial"], target_names = ["Ham", "Spam"]))
conf_mat = confusion_matrix(y_test, prediction['Multinomial'])

conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (3,3))

sns.heatmap(conf_mat_normalized)

plt.ylabel('True label')

plt.xlabel('Predicted label')
print(conf_mat)
from sklearn.model_selection import GridSearchCV

k_range = np.arange(1,10)

param_grid = dict(n_neighbors=k_range)

#print(param_grid)

model = KNeighborsClassifier()

grid = GridSearchCV(model,param_grid)

grid.fit(X_train_df,y_train)
grid.best_estimator_
grid.best_params_
grid.best_score_
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=2)

model.fit(X_train_df,y_train)

prediction["knn_tune"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["knn_tune"])
grid.grid_scores_
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(750,750))

clf.fit(X_train_df,y_train)

prediction["NN"] = clf.predict(X_test_df)

accuracy_score(y_test,prediction["NN"])
conf_mat_nn = confusion_matrix(y_test, prediction['NN'])
print(conf_mat_nn)
pd.set_option('display.max_colwidth', -1)
X_test[y_test < prediction["Multinomial"] ].head(5)
#missclaasify as ham

X_test[y_test > prediction["NN"] ].head(5)