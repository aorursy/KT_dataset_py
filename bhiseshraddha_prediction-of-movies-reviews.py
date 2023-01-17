from google.colab import drive
drive.mount('/gdrive')

import os
os.chdir("/gdrive/My Drive/movies")
!pip install nltk

import pandas as pd 
import numpy as np

reviews=pd.read_csv("IMDB Dataset.csv")

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = stopwords.words('english')
print(stopwords.words('english'))



from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 50000):
    review = re.sub('[^a-zA-Z]', ' ', reviews['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stops)]
    review = ' '.join(review)
    corpus.append(review)

corp_str = str(corpus)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(relative_scaling=1.0).generate(corp_str)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
from wordcloud import STOPWORDS
mystopwrds = set(STOPWORDS)
mystopwrds.add("br")
wc = WordCloud(stopwords=mystopwrds,relative_scaling=1.0,background_color="white")
wordcloud = wc.generate(corp_str)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()
y = reviews.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2020)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))



from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()
y =reviews.iloc[:, 1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2020)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"don\'t", "do not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase




from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 50000):
    review = reviews['review'][i]
    review = decontracted(review)    
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
#    ps = PorterStemmer()
    review = review.split()
#    review = [word for word in review if not word in set(stops)]
    #review = ' '.join(review)
    #review = [review]
    corpus.append(review)



from gensim.models import  Word2Vec

 

model_r =  Word2Vec(corpus, min_count=1,sg=0)


means = pd.DataFrame()
for i in corpus :
    row_means = np.mean(model_r[i],axis=0)
    row_means = pd.Series(row_means)
    means = pd.concat([means, row_means],axis=1)
    

X = means.T
y = reviews.iloc[:, 1]



from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2020,
                                                    stratify=y)
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=2020,
                                  n_estimators=10,oob_score=True)
model_rf.fit( X , y )


y_pred = model_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


from sklearn.metrics import roc_curve, roc_auc_score
y_test = pd.get_dummies(y_test, drop_first=True)
y_pred_prob = model_rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)


# summarize the loaded model
print(model_r)

# summarize vocabulary
words = list(model_r.wv.vocab)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers=[RandomForestClassifier(random_state=2020),GaussianNB(),KNeighborsClassifier(),
            DecisionTreeClassifier(random_state=2020)]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]

    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)

    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)


result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()