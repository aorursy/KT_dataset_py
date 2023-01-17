#------------------------------------------Libraries---------------------------------------------------------------#

####################################################################################################################

#-------------------------------------Boiler Plate Imports---------------------------------------------------------#

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

#---------------------------------------Text Processing------------------------------------------------------------#

import regex

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords 

from nltk.tokenize import WordPunctTokenizer

from string import punctuation

from nltk.stem import WordNetLemmatizer

#------------------------------------Metrics and Validation---------------------------------------------------------#

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

#-------------------------------------Models to be trained----------------------------------------------------------#

from sklearn.ensemble import StackingClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import xgboost

#####################################################################################################################
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
train.isna().sum()
sns.barplot(x = train.keyword.value_counts()[:10].index, y = train.keyword.value_counts()[:10])
sns.barplot(x = train.target.value_counts().index, y = train.target.value_counts())
def make_wordcloud(words,title):

    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)

    plt.figure(figsize=(20,20))

    plt.imshow(cloud, interpolation="gaussian")

    plt.axis("off") 

    plt.title(title, fontsize=60)

    plt.show()
wordnet_lemmatizer = WordNetLemmatizer()



stop = stopwords.words('english')



for punct in punctuation:

    stop.append(punct)



def filter_text(text, stop_words):

    word_tokens = WordPunctTokenizer().tokenize(text.lower())

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and w.find('http') == -1 and len(w) >= 3]

    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 

    return " ".join(filtered_text)
train["filtered_text"] = train.text.apply(lambda x : filter_text(x, stop)) 

train.head()
all_text = " ".join(train[train.target == 1].filtered_text) 

make_wordcloud(all_text, 'True')
count = pd.DataFrame(all_text.split(), columns = ['words'])

top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10.words.value_counts().index,

            y = top_10.words.value_counts(), palette = sns.color_palette("mako"))
count['len'] = count.words.apply(lambda x : len(x))

top_10_len = count[count['len'].isin(list(count.len.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10_len.len.value_counts().index,

            y = top_10_len.len.value_counts(), palette = sns.color_palette("mako"))
all_text = " ".join(train[train.target == 0].filtered_text) 

make_wordcloud(all_text, 'False')
count = pd.DataFrame(all_text.split(), columns = ['words'])

top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10.words.value_counts().index,

            y = top_10.words.value_counts(), palette = sns.color_palette("mako"))
count['len'] = count.words.apply(lambda x : len(x))

top_10_len = count[count['len'].isin(list(count.len.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10_len.len.value_counts().index,

            y = top_10_len.len.value_counts(), palette = sns.color_palette("mako"))
tfidf = TfidfVectorizer(lowercase=False)

train_vec = tfidf.fit_transform(train['filtered_text'])

train_vec.shape
train['classification'] = train['target'].replace([0,1],['Disaster', 'Not a Disaster'])
x_train, x_val, y_train, y_val = train_test_split(train_vec,train['target'], stratify=train['target'], test_size=0.2)
C = np.arange(0, 1, 0.001)

max_iter = range(100, 500)

warm_start = [True, False]

solver = ['lbfgs', 'newton-cg', 'liblinear']

penalty = ['l2', 'l1']



params = {

    'C' : C,

    'max_iter' : max_iter,

    'warm_start' : warm_start,

    'solver' : solver,

    'penalty' : penalty

}



random_search = RandomizedSearchCV(

    estimator = LogisticRegression(random_state = 1),

    param_distributions = params,

    n_iter = 100,

    cv = 3,

    n_jobs = -1,

    random_state = 1,

    verbose = 1

).fit(x_train, y_train)



random_search.best_params_
model_lr = random_search.best_estimator_

model_lr.score(x_train, y_train)
predicted = model_lr.predict(x_val)



lr_acc = accuracy_score(y_val,predicted)

lr_cop = cohen_kappa_score(y_val,predicted)

lr = pd.DataFrame([lr_acc, lr_cop], columns = ['Logistic Regression with RandomizedSearchCV'])



print("Test score: {:.2f}".format(lr_acc))

print("Cohen Kappa score: {:.2f}".format(lr_cop))



plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
alpha = np.arange(0, 1, 0.001)

fit_prior = [True, False]



params = {

    'alpha' : alpha,

    'fit_prior' : fit_prior

}



random_search = RandomizedSearchCV(

    estimator = MultinomialNB(),

    param_distributions = params,

    n_iter = 100,

    cv = 3,

    n_jobs = -1,

    random_state = 1,

    verbose = 1

).fit(x_train, y_train)



random_search.best_params_
model_mnb = random_search.best_estimator_

model_mnb.score(x_train, y_train)
predicted = model_mnb.predict(x_val)



mnb_acc = accuracy_score(y_val,predicted)

mnb_cop = cohen_kappa_score(y_val,predicted)

mnb = pd.DataFrame([mnb_acc, mnb_cop], columns = ['MultinomialNB with RandomizedSearchCV'])



print("Test score: {:.2f}".format(mnb_acc))

print("Cohen Kappa score: {:.2f}".format(mnb_cop))



plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
model_sgd_hinge = SGDClassifier(

    loss='hinge',

    penalty='l2',

    alpha=0.0001,

    l1_ratio=0.15,

    fit_intercept=True,

    max_iter=10000,

    tol=0.001,

    shuffle=True,

    verbose=0,

    epsilon=0.1,

    n_jobs=-1,

    random_state=1,

    learning_rate='optimal',

    eta0=0.0,

    power_t=0.5,

    early_stopping=False,

    validation_fraction=0.1,

    n_iter_no_change=5,

    class_weight=None,

    warm_start=True,

    average=False).fit(x_train, y_train)



model_sgd_hinge.score(x_train, y_train)
predicted = model_sgd_hinge.predict(x_val)



sgd_hinge_acc = accuracy_score(y_val,predicted)

sgd_hinge_cop = cohen_kappa_score(y_val,predicted)

sgd_hinge = pd.DataFrame([sgd_hinge_acc, sgd_hinge_cop], columns = ['SGDClassifier with Squared Hinge Loss'])



print("Test score: {:.2f}".format(sgd_hinge_acc))

print("Cohen Kappa score: {:.2f}".format(sgd_hinge_cop))

plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
criterion = ['gini', 'entropy']

splitter = ['best', 'random']

max_depth = range(5, 200)

max_features = ['auto', 'sqrt', 'log2']



params = {

    'criterion' : criterion,

    'splitter' : splitter,

    'max_depth' : max_depth,

    'max_features' : max_features

}



random_search = RandomizedSearchCV(

    estimator = DecisionTreeClassifier(random_state = 1),

    param_distributions = params,

    n_iter = 100,

    cv = 3,

    n_jobs = -1,

    random_state = 1,

    verbose = 1

).fit(x_train, y_train)



random_search.best_params_
model_dt = random_search.best_estimator_

model_dt.score(x_train, y_train)
predicted = model_dt.predict(x_val)



dt_acc = accuracy_score(y_val,predicted)

dt_cop = cohen_kappa_score(y_val,predicted)

dt = pd.DataFrame([dt_acc, dt_cop], columns = ['DecisionTreeClassifier with RandomizedSearchCV'])



print("Test score: {:.2f}".format(dt_acc))

print("Cohen Kappa score: {:.2f}".format(dt_cop))

plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
n_neighbors = range(5, 100)

weights = ['uniform', 'distance']

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

leaf_size = range(30, 100)

p = range(1, 4)



params = {

    'n_neighbors' : n_neighbors,

    'weights' : weights,

    'algorithm' : algorithm,

    'leaf_size' : leaf_size,

    'p'  : p

}



random_search = RandomizedSearchCV(

    estimator = KNeighborsClassifier(n_jobs = -1),

    param_distributions = params,

    n_iter = 200,

    cv = 3,

    n_jobs = -1,

    random_state = 1,

    verbose = 1

).fit(x_train, y_train)



random_search.best_params_
model_kn = random_search.best_estimator_

model_kn.score(x_train, y_train)
predicted = model_kn.predict(x_val)



kn_acc = accuracy_score(y_val,predicted)

kn_cop = cohen_kappa_score(y_val,predicted)

kn = pd.DataFrame([kn_acc, kn_cop], columns = ['KNeighborsClassifier with RandomizedSearchCV'])



print("Test score: {:.2f}".format(kn_acc))

print("Cohen Kappa score: {:.2f}".format(kn_cop))

plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
estimators = [

    ('kn', model_kn),

    ('mnb', model_mnb),

    ('lr', model_lr),

    ('dt', model_dt)

]



estimators
model_voting = VotingClassifier(

    estimators = estimators,

    voting='soft', 

    n_jobs=-1,

    flatten_transform=True, 

    verbose=1).fit(x_train, y_train)



model_voting.score(x_train, y_train)
predicted = model_voting.predict(x_val)



voting_acc = accuracy_score(y_val,predicted)

voting_cop = cohen_kappa_score(y_val,predicted)

voting = pd.DataFrame([voting_acc, voting_cop], columns = ['Soft Voting Classifier'])



print("Test score: {:.2f}".format(voting_acc))

print("Cohen Kappa score: {:.2f}".format(voting_cop))



plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
model_stack = StackingClassifier(

    estimators=estimators+[('svm', model_sgd_hinge)],

    final_estimator=model_kn,

    n_jobs = -1,

    verbose = 1

)



model_stack.fit(x_train, y_train)



model_stack.score(x_train, y_train)
predicted = model_stack.predict(x_val)



stack_acc = accuracy_score(y_val,predicted)

stack_cop = cohen_kappa_score(y_val,predicted)

stack = pd.DataFrame([stack_acc, stack_cop], columns = ['Stacking Classifier'])



print("Test score: {:.2f}".format(stack_acc))

print("Cohen Kappa score: {:.2f}".format(stack_cop))



plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['Disaster', 'Not a Disaster']),

            yticklabels=(['Disaster', 'Not a Disaster']))
model_comp = pd.concat([lr, mnb, sgd_hinge, dt, kn, voting, stack], axis = 1)

model_comp
model = model_mnb

model.fit(train_vec, train['target'])

model.score(train_vec, train['target'])
test['filtered_text'] = test.text.apply(lambda x : filter_text(x, stop))

test.head()
test_vec = tfidf.transform(test.filtered_text)

test_vec.shape
sub.target = model.predict(test_vec)
sub.to_csv('submissions.csv', index= False)