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
!pip install pywsd
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import nltk
import re

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pywsd.utils import lemmatize, lemmatize_sentence
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict,  KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from scipy.stats import zscore, chi2_contingency,normaltest
from imblearn.over_sampling import SMOTE

color= "Spectral"
data = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
data.head(2)
data.info()
data.describe()
sns.boxplot(x = "airline_sentiment_confidence", data = data)
total = data.isnull().sum().sort_values(ascending = False)
perc = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, perc], axis = 1, keys = ['total_missing', 'perc_missing'])
missing_data
fig, ax = plt.subplots( figsize = (15, 8))
sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
ax.set_title('Missing Data')
plt.show()
data["retweet_count"].value_counts()[1]/len(data["retweet_count"])
data["retweet_count"].value_counts()
data.loc[data["retweet_count"]==44,["retweet_count",'airline_sentiment']]
ax, fig = plt.subplots(1, 1, figsize = (15, 5))
sns.boxplot(y = "airline", x = "retweet_count", data = data)
plt.xlabel('# of Retweets', size = 14)
plt.ylabel('Airline', size = 14)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.title('The Distribution of Number of Retweets for each Airline', fontsize = 14)

plt.show()
ax, fig = plt.subplots(1, 1, figsize = (15, 5))
sns.boxplot(y = "airline_sentiment", x = "retweet_count", data = data)
plt.xlabel('# of Retweets', size = 14)
plt.ylabel('Airline', size = 14)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.title('The Distribution of Number of Retweets for each Airline', fontsize = 14)

plt.show()
data['n_words'] = [len(t.split()) for t in data.text]
print("Positive # words mean =",data['n_words'][data['airline_sentiment']=='positive'].mean())
print("Neutral  # words mean =",data['n_words'][data['airline_sentiment']=='positive'].mean())
print("Negative # words mean =",data['n_words'][data['airline_sentiment']=='negative'].mean())
print()
print("Positive # words median =",data['n_words'][data['airline_sentiment']=='positive'].median())
print("Neutral  # words median =",data['n_words'][data['airline_sentiment']=='positive'].median())
print("Negative # words median =",data['n_words'][data['airline_sentiment']=='negative'].median())
fig = plt.figure(figsize = (15, 6))
sns.distplot(data['n_words'][data['airline_sentiment']=='positive'], color='g', label = 'positive')
sns.distplot(data['n_words'][data['airline_sentiment']=='negative'], color='r', label = 'negative')
sns.distplot(data['n_words'][data['airline_sentiment']=='neutral'], color='b', label = 'neutral')
plt.legend(loc='best')
plt.xlabel('# of Words', size = 14)
plt.ylabel('Count', size = 14)
plt.title('The Distribution of Number of Words for each Class', fontsize = 14)
plt.show()
ND = data['n_words'][data['airline_sentiment']=='neutral']
k2, p = normaltest(ND)
alpha = 1e-3
print("p = {:g}".format(p))
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected, (not normally distributed)")
else:
    print("The null hypothesis cannot be rejected, (normally distributed)")
data['tweet_created'] =  data['tweet_created'].str[:-6]
data['tweet_created'] = pd.to_datetime(data['tweet_created'],format='%Y-%m-%d')
data['tweet_created'] = pd.to_datetime(data.tweet_created.dt.strftime("%Y-%m-%d"))
data['tweet_created']#.asfreq(freq='30S')
sizes = data['airline_sentiment'].value_counts()
labels = data['airline_sentiment'].value_counts().index
colors = ['#ff6666', '#ffcc99', '#99ff99']
data['airline_sentiment'].value_counts()
plot = plt.pie(sizes, labels=labels, colors=colors, startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()
plt.show()
st = data.groupby([data.tweet_created.dt.date,data.airline,data.airline_sentiment]).count()#.agg({'airline_sentiment': 'count'}).reset_index()
st = st.iloc[:,:1].reset_index()
chart = sns.catplot(x="airline", hue="airline_sentiment", y="tweet_id", data=st,palette=sns.diverging_palette(10, 220, sep=80, n=3,center="dark"), kind="bar",ci=None,height=5,aspect=1.5)
(chart.set_xticklabels(st['tweet_created'].unique(), horizontalalignment='center', rotation=20).despine(left=True)) 
chart = sns.catplot(x="tweet_created", hue="airline_sentiment", y="tweet_id",col='airline',col_wrap=2,palette=sns.diverging_palette(10, 220, sep=80, n=3,center="dark"), data=st, kind="bar",aspect=1.5)
(chart.set_xticklabels(st['tweet_created'].unique(), horizontalalignment='center', rotation=20).despine(left=True)) 
data.groupby(['airline','airline_sentiment']).count().iloc[:,0].sort_values()#/data.groupby(['airline']).count().iloc[:,0].sort_values()

pvst = st.pivot_table(index='airline_sentiment', columns='tweet_created', values='tweet_id', aggfunc='sum').fillna(0)
pvst
stat, p, dof, expected = chi2_contingency(pvst)
alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('tweet_created and airline_sentiment are dependent') 
else: 
    print('tweet_created and airline_sentiment are Independent') 
ngr = data.groupby([data.negativereason,data.airline]).count()[['tweet_id']].reset_index()#.sort_values(by='tweet_id',ascending=False)
# ngr
chart = sns.catplot(x="airline", hue="negativereason", y="tweet_id", data=ngr,palette='Set3', kind="bar",ci=None,height=5,aspect=2)
# (chart.set_xticklabels(st['airline'].unique(), horizontalalignment='center', rotation=20).despine(left=True)) 
le = LabelEncoder()
le.fit(data['airline_sentiment'])
le.classes_
data['airline_sentiment'] = le.transform(data['airline_sentiment'])
data['airline_sentiment'].value_counts()
stp_wrds = stopwords.words("english")
stp_wrds.append("flight")
stp_wrds.append('get')
stp_wrds.remove('no')
stp_wrds.remove('don')
stp_wrds.remove('nor')
stp_wrds.remove('not')
stp_wrds.remove('now')
# to clean the tweets
def clean(tweet):
    tweet = tweet.lower() # lowercase
    tweet = re.sub(r'http\S*\b', '',tweet)# remove links
    tweet = re.sub(r'@\S*\b', '',tweet) # remove mentions 
    tweet = re.sub(r'[^a-zA-Z]', ' ',tweet) # only words
    tweet = TweetTokenizer().tokenize(tweet) 
    tweet = ' '.join([i for i in tweet if (i not in stp_wrds)])
#     tweet = ' '.join([PorterStemmer().stem(i) for i in tweet])
    tweet = lemmatize_sentence(tweet) #limmatizeing
    tweet = ' '.join([i for i in tweet if (i not in stp_wrds)])
#     print(tweet)
    return(tweet)
data['cl_text'] = data['text'].apply(clean)
neg_tweets = data[data['airline_sentiment'] == 0]
neg_string = []
for t in neg_tweets.cl_text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(neg_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()
neu_tweets = data[data['airline_sentiment'] == 1]
neu_string = []
for t in neu_tweets.cl_text:
    neu_string.append(t)
neu_string = pd.Series(neu_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(neu_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()
pos_tweets = data[data['airline_sentiment'] == 2]
pos_string = []
for t in pos_tweets.cl_text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()
data_ = pd.get_dummies(data[['retweet_count','airline','airline_sentiment_confidence','n_words']], drop_first=True)
data_['cl_text'] = data['cl_text']
X = data_
y = data['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify = y)
# stratfied k fold for preserving the percentage of samples for each class
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
## To get the best parameters for tfdif, but it takes times, so I just took the best parameters
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('clf', SGDClassifier()),
# ])
# parameters = {
#     'tfidf__max_df': (0.71, 0.8,0.9, 1.0),
#     'tfidf__max_features': (None, 5000, 10000),
#     'tfidf__ngram_range': ((1, 1), (1, 2),(2, 3)),  # unigrams or bigrams
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'tfidf__smooth_idf':(True, False),
#     'clf__max_iter': (20,),
#     'clf__alpha': (0.00001, 0.000001),
#     'clf__penalty': ('l2', 'elasticnet'),
#     'clf__max_iter': (10, 50, 80),
# }
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# print(parameters)
# grid_search.fit(X_train['cl_text'], y_train)
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
tfidf_vect = TfidfVectorizer(max_df=0.71, norm='l2',max_features=None, ngram_range=(1,2), smooth_idf=True ,use_idf= False)
X_train_tf = tfidf_vect.fit_transform(X_train['cl_text'])
X_test_tf = tfidf_vect.transform(X_test['cl_text'])
cols = sp.sparse.csr_matrix(X_train.drop('cl_text', axis=1).astype(float))
X_train_ = sp.sparse.hstack((X_train_tf, cols))
cols = sp.sparse.csr_matrix(X_test.drop('cl_text', axis=1).astype(float))
X_test_ = sp.sparse.hstack((X_test_tf, cols))
smt = SMOTE(random_state=777, k_neighbors=1)
X_SMOTE, y_SMOTE = smt.fit_sample(X_train_, y_train)
def metrics(model, kfold, X_train, X_test, y_train, y_test): 
    model.fit(X_train, y_train)
    train_score =  model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Training Score =', train_score)
    print('Testing Score =', test_score)
    ####
    cv = cross_val_score(model, X_train, y_train, cv = kfold)
    print("Cross Val Scores =", cv)
    print("Cross Val Standard Deviation =", cv.std())
    print('Cross Val Mean Score =', cv.mean()); 
    ###
    pred = model.predict(X_test)
    print('Confusion Matrix =\n',confusion_matrix(y_test, pred))
    print('Classification Report =\n',classification_report(y_test, pred))
    return pred, test_score, cv.mean()
##KNN WITHOUT OVER-SAMPLING (SMOTE)
print('K-Neighbors Classifier')
params = {
    "n_neighbors" : [5],#,15,25,30,35,40, 100],
    "weights" : ["distance"] #"uniform"
}
knn= GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, cv=10)
knn_pred, knn_test, knn_train = metrics(knn, kfold, X_train_, X_test_, y_train, y_test)
print(knn.best_params_)
# ##KNN WITH OVER-SAMPLING (SMOTE)
# # Worse
# print('K-Neighbors Classifier')
# params = {
#     "n_neighbors" : [5,15,25,30,35,40, 100],
#     "weights" : ["uniform" , "distance"]
# }
# knn= GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, cv=10)
# knn_pred, knn_test, knn_train = metrics(knn, kfold, X_SMOTE, X_test_, y_SMOTE, y_test)
print('SVC')
params = {
    'C':[10],
    'gamma':[0.1], 
    'kernel':['rbf']#'linear',
}

svc = SVC(random_state=42)
svc = GridSearchCV(svc, params, n_jobs=-1, cv=10)
svc_pred, svc_test, svc_train = metrics(svc, kfold, X_train_, X_test_, y_train, y_test)
print(svc.best_params_)
print('Decision Tree')
param_grid = {'max_depth':[2],#1,
              'min_samples_leaf':[3]}#,5
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree = GridSearchCV(decision_tree, param_grid=param_grid, cv=5, n_jobs=-1)
dt_pred, dt_test, dt_train = metrics(decision_tree, kfold, X_train_, X_test_, y_train, y_test)
print(decision_tree.best_params_)
print('Logistic Regression Model')
params = {
    "penalty": ["l2"],#"l1", 
    "C": [10000.0]#np.logspace(-2,4,10)
}
logistic_regression = GridSearchCV(LogisticRegression(random_state=42), params, n_jobs=-1, cv=10)
lg_pred, lg_test, lg_train = metrics(logistic_regression, kfold, X_train_, X_test_, y_train, y_test)
print(logistic_regression.best_params_)
print('Random Forest')
rf_params = {
        'max_features':[2]#, 7, 8],
        #'max_depth': [1, 2, 3, 4, 5, 8],
        #'criterion':['gini', 'entropy']
}
random_forest = RandomForestClassifier(random_state=42,n_estimators=80)
random_forest = GridSearchCV(random_forest, param_grid=rf_params, cv=5, n_jobs=-1)
rf_pred, rf_test, rf_train = metrics(random_forest, kfold, X_train_, X_test_, y_train, y_test)
print(random_forest.best_params_)
print('AdaBoost')
param_grid = { 
    'n_estimators': [80],#10,50, 
    'learning_rate':[0.1]#0.01,
}
ada_boost = AdaBoostClassifier(random_state=42)
ada_boost = GridSearchCV(ada_boost, param_grid=param_grid, cv=5, n_jobs=-1)
ab_pred, ab_test, ab_train = metrics(ada_boost, kfold, X_train_, X_test_, y_train, y_test)
print(ada_boost.best_params_)
print('Extra Trees')
rf_params = {
    'n_estimators': [10],#, 100, 400, 800, 1100, 1850],
    'max_features':['auto'],
    'max_depth': [1],#, 2, 3, 4, 5, 8],
    'criterion':['gini']
}
extra_trees = ExtraTreesClassifier(n_estimators=100,random_state=42)
gs = GridSearchCV(extra_trees, param_grid=rf_params, cv=5, n_jobs=-1)
et_pred, et_test, et_train = metrics(gs, kfold, X_train_, X_test_, y_train, y_test)
fig, (ax1) = plt.subplots(figsize=(10,6))
inds = range(1,8)
labels = ["KNN", "Logistic Regression", "Decision Tree", "Random Forest",'Extra Trees', 'AdaBoost', 'SVC' ]
scores_all = [knn_train, lg_train, dt_train, rf_train, et_train, ab_train, svc_train]
scores_predictive = [knn_test, lg_test, dt_test, rf_test, et_test, ab_test, svc_test]    
ax1.bar(inds, scores_all, color=sns.color_palette(color)[5], alpha=0.3, hatch="x", edgecolor="none",label="CrossValidation Set")
ax1.bar(inds, scores_predictive, color=sns.color_palette(color)[0], label="Testing set")
ax1.set_ylim(0.4, 1)
ax1.set_ylabel("Accuracy score")
ax1.axhline(0.626913, color="black", linestyle="--")
ax1.set_title("Accuracy scores for basic models", fontsize=17)
ax1.set_xticks(range(1,8))
ax1.set_xticklabels(labels, size=12, rotation=40, ha="right")
ax1.legend()