# Import All Required Libraries

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline

posts = pd.read_csv('../input/newData.csv', encoding = 'latin-1')

posts.head(10)
print(posts.shape)
posts['Label'].value_counts()
posts.drop( posts[pd.isnull( posts['Label'] )].index, inplace=True)

posts.describe()
total_not_spam = posts['Label'].value_counts()[0]
total_spam     = posts['Label'].value_counts()[1]

labels  = ['Spam','Not Spam']
sizes   = [total_spam, total_not_spam]
colors  = ['#ff9999','#66b3ff']

explode = (0.05,0.05)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.8, explode = explode)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')  
plt.tight_layout()
plt.show()

def clean(data):
    
    data = re.sub(r"\n", " ", data)
    data = re.sub(r"<[^a].*?>", ' ', data)
    data = re.sub(r"&.*?;", ' ', data)
    data = re.sub(r"{.*?}", " ", data)
    data = re.sub(r"\\r", " ", data)
    data = re.sub(r"\\n", " ", data)
    data = re.sub(r"[0-9]+", " ", data)
    
    return data

posts['Data'] = posts.Data.apply(clean)
posts.head(10)

def lemmatise(message):
    message = message.lower()
    lemm = WordNetLemmatizer()
    
    words = TextBlob(message).words                       # Tokenization {Splitting into words}
    stop_words = stopwords.words('english')               # Importing List of Stop Words
    
    message_new = [word for word in words if word not in stop_words]
    message_new = [lemm.lemmatize(word) for word in message_new]
    
    return message_new

print('\n Before :- \n')
print(posts['Data'][550])

print('\n After :- \n')
print(lemmatise(posts['Data'][550]))
spam_words = ''
for post in posts[posts['Label']=='Spam']['Data']:
    spam_words = spam_words + ' '.join(lemmatise(post)) + ' '

not_spam_words = ''
for post in posts[posts['Label']=='Not Spam']['Data']:
    not_spam_words = not_spam_words + ' '.join(lemmatise(post)) + ' '

spam_wordcloud = WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(spam_words)
plt.figure( figsize=(11,8) )
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

most_spam_words = Counter(spam_words.split()).most_common(25)

df = pd.DataFrame.from_dict(most_spam_words, orient='columns')
df.plot(kind='barh', figsize=(15,10), align='center', color='coral', grid=True)

plt.yticks(np.arange(25), df[0], fontsize=20)
plt.xticks(fontsize=20)

plt.title('Most Common \'Spam\' Words', fontsize=25)
plt.legend(('Counts',), fontsize=20)
plt.show()

most_not_spam_words = Counter(not_spam_words.split()).most_common(25)

df = pd.DataFrame.from_dict(most_not_spam_words, orient='columns')
df.plot(kind='barh', figsize=(15,10), align='center', color='coral', grid=True)

plt.yticks(np.arange(25), df[0], fontsize=20)
plt.xticks(fontsize=20)

plt.title('Most Common \'Not Spam\' Words', fontsize=25)
plt.legend(('Counts',), fontsize=20)
plt.show()

posts_train, posts_test, label_train, label_test = train_test_split(posts['Data'], posts['Label'], test_size = 0.25, random_state=42)

print(' Train Data :- {} \n Test Data :- {} \n Total :- {} \n'.format(len(posts_train), len(posts_test), len(posts_train)+len(posts_test)))
print('Spam Test Data :\n\n{}'.format(posts_test[label_test == 'Spam']))

print('Train Data')
print('Total %9d' % posts_train.shape[0])
print(label_train.value_counts())
print('\nTest Data')
print('Total %9d' % posts_test.shape[0])
print(label_test.value_counts())
BoW_vect = CountVectorizer(analyzer=lemmatise, stop_words="english")

posts_train_BoW = BoW_vect.fit_transform(posts_train)
posts_test_BoW  = BoW_vect.transform(posts_test)

tf_idf_vect = TfidfVectorizer(analyzer=lemmatise, stop_words="english")

posts_train_tfidf = tf_idf_vect.fit_transform(posts_train)
posts_test_tfidf  = tf_idf_vect.transform(posts_test)

mnb      = MultinomialNB()
lin_svm  = LinearSVC()
svm      = SVC()
dtc      = DecisionTreeClassifier()
lr       = LogisticRegression()
rfc      = RandomForestClassifier()

classifiers = {
                'MNB' : mnb, 
                'LinearSVC' : lin_svm, 
                'SVC' : svm, 
                'DTC' : dtc, 
                'LR'  : lr, 
                'RFC' : rfc
              }
def train(model, train_posts, train_labels):
    model.fit(train_posts, train_labels)

def test(model, test_posts):
    return model.predict(test_posts)
tfidf_scores = []
BoW_scores = []

for k,v in classifiers.items():
    train(v, posts_train_BoW, label_train)
    results = test(v, posts_test_BoW)
    BoW_scores.append((k,[accuracy_score(label_test, results)]))

for k,v in classifiers.items():
    train(v, posts_train_tfidf, label_train)
    results = test(v, posts_test_tfidf)
    tfidf_scores.append((k,[accuracy_score(label_test, results)]))

df = pd.DataFrame.from_items(BoW_scores, orient='index', columns=['BoW'])
df2 = pd.DataFrame.from_items(tfidf_scores, orient='index', columns=['TF-IDF'])

df = pd.concat([df,df2], axis=1)
df
df.plot(kind='bar', ylim=(0.90,1.0), figsize=(15,8), align='center', colormap="tab20")

plt.xticks(np.arange(6), df.index, rotation='horizontal', fontsize=18)
plt.yticks(fontsize=15)

plt.ylabel('Accuracy Score', fontsize=20)
plt.xlabel('Classifiers', fontsize=20)
plt.title('Comparison of Classifiers', fontsize=25)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=20, borderaxespad=0.)
plt.show()
pipeline = Pipeline([
                        ('BoW', BoW_vect),
                        ('nb', MultinomialNB()),
                    ])

params_mnb = {
                'nb__alpha' : (0.001, 0.01, 0.1, 1, 10)
             }

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    
    grid_mnb = GridSearchCV(pipeline, params_mnb, refit=True, n_jobs=-1, scoring='accuracy', cv=10)
    
    %time mnb_detector = grid_mnb.fit(posts_train, label_train)

    print('\nFitting & Cross Validating Train Data....\nDone!')
    print('\nTraining Best Score Based on Cross Validation :- {}\n'.format(mnb_detector.best_score_))
    
    predictions_mnb = mnb_detector.predict(posts_test)
    
    print('Running the Model on Test Data....\nDone!, Here are the Reports...\n')
    print('Confusion Matrix :-\n{}\n'.format(confusion_matrix(label_test, predictions_mnb)))
    print('Classification Report :- \n{}'.format(classification_report(label_test, predictions_mnb)))
    print('Accuracy Score :- {}\n'.format(accuracy_score(label_test, predictions_mnb)))
    
    best_parameters_mnb = mnb_detector.best_estimator_.get_params()
    print('Best Parameters :')
    for param_name in sorted(params_mnb.keys()):
        print("%s: %r" % (param_name, best_parameters_mnb[param_name]))

cnf_matrix = confusion_matrix(label_test, predictions_mnb)

plt.matshow(cnf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix\n', fontsize=20)

plt.xticks(np.arange(2),['Not Spam', 'Spam'])
plt.yticks(np.arange(2),['Not Spam', 'Spam'])

plt.ylabel('Expected', fontsize=18)
plt.xlabel('Predicted', fontsize=18)

plt.colorbar()
plt.show()

pipeline_svm = Pipeline([
                            ('tfidf', tf_idf_vect),
                            ('svc', SVC()),
                        ])

param_svm = {
                'svc__gamma' : (0.05, 0.1, 1),
                'svc__kernel' : ('linear', 'sigmoid'),
            }

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    
    grid_svm = GridSearchCV(pipeline_svm, param_svm, refit=True, n_jobs=-1, scoring='accuracy', cv=10)

    %time svm_detector = grid_svm.fit(posts_train, label_train)
    
    print('\nFitting & Cross Validating Train Data....\nDone!')
    print('\nTraining Best Score Based on Cross Validation :- {}\n'.format(svm_detector.best_score_))
    
    predictions_svm = svm_detector.predict(posts_test)
    
    print('Running the Model on Test Data....\nDone!, Here are the Reports...\n')
    print('Confusion Matrix :-\n{}\n'.format(confusion_matrix(label_test, predictions_svm)))
    print('Classification Report :- \n{}'.format(classification_report(label_test, predictions_svm)))
    print('Accuracy Score :- {}'.format(accuracy_score(label_test, predictions_svm)))
    
    best_parameters = svm_detector.best_estimator_.get_params()
    print('\nBest Parameters :- \n')
    for param_name in param_svm.keys():
        print("%s: %r" % (param_name, best_parameters[param_name]))

cnf_matrix = confusion_matrix(label_test, predictions_mnb)
plt.matshow(cnf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix\n', fontsize=20)

plt.xticks(np.arange(2),['Not Spam', 'Spam'])
plt.yticks(np.arange(2),['Not Spam', 'Spam'])

plt.ylabel('Expected', fontsize=18)
plt.xlabel('Predicted', fontsize=18)

plt.colorbar()
plt.show()

pred_scores = []
krnl = {'rbf' : 'rbf','polynominal' : 'poly', 'sigmoid': 'sigmoid', 'linear':'linear'}
for k,v in krnl.items():
    for i in np.linspace(0.05, 1, num=20):
        svc = SVC(kernel=v, gamma=i)
        svc.fit(posts_train_tfidf, label_train)
        pred = svc.predict(posts_test_tfidf)
        pred_scores.append((k, [i, accuracy_score(label_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Gamma','Score'])
df['Score'].plot(kind='line', figsize=(15,10), ylim=(0.9,1.0),colormap='tab20')

plt.title('Variation of Accuracy with Different Kernels', fontsize=20)
plt.xlabel('Kernels', fontsize=20)
plt.ylabel('Accuracy Scores', fontsize=20)
plt.yticks(fontsize=15)

df[df['Score'] == df['Score'].max()].head(3)

from sklearn.externals import joblib

final_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer=lemmatise, stop_words='english')),
        ('svc', SVC(kernel='linear', gamma=0.05)),
    ])

predictor = final_pipeline.fit(posts['Data'], posts['Label'])

joblib.dump(predictor, 'Trained-Model.pkl')

pp=joblib.load('Trained-Model.pkl')
pp