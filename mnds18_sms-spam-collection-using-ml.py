import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
%config InlineBackend.figure_format = 'retina'
data = pd.read_csv("../input/spam.csv",encoding='latin-1')
data.head()
#Drop column and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})
data.tail()
#Count observations in each label
data.label.value_counts()
# convert label to a numerical variable
data['label_num'] = data.label.map({'ham':0, 'spam':1})
data.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_test_df)
ham_words = ''
spam_words = ''
spam = data[data.label_num == 1]
ham = data[data.label_num ==0]
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
from wordcloud import WordCloud
# Generate a word cloud image
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
from sklearn.model_selection import GridSearchCV
k_range = np.arange(1,30)
k_range
param_grid = dict(n_neighbors=k_range)
print(param_grid)
model = KNeighborsClassifier()
grid = GridSearchCV(model,param_grid)
grid.fit(X_train_df,y_train)
grid.best_estimator_
grid.best_params_
grid.best_score_
grid.grid_scores_
print(classification_report(y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))
conf_mat = confusion_matrix(y_test, prediction['Multinomial'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(conf_mat)
pd.set_option('display.max_colwidth', -1)
X_test[y_test < prediction["Multinomial"] ]
X_test[y_test > prediction["Multinomial"] ]
