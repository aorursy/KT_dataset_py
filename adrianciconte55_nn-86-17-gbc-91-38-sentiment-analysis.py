import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import datetime
%matplotlib inline
import time
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import average_precision_score
from sklearn import model_selection
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, regexp_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer 
from nltk.classify import NaiveBayesClassifier
from nltk.classify import ClassifierI
from nltk.sentiment import sentiment_analyzer, SentimentAnalyzer, SentimentIntensityAnalyzer
from collections import Counter
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import re
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
sns.set_style("white")
sns.set_style("ticks")
figsize = (22,4)
fontsize = 22
#read in data-set
df = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
#show first five lines
df.head()
#create Missing Values function to help determine what columns to keep and drop
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum().div(df.isnull().count()).sort_values(ascending=False))
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_data(df)
#drop unnamed [False Index] and Title // will focus on reviews for text mining and sentiment
df.drop(['Unnamed: 0', 'Title'], axis=1, inplace=True)

#show length of dataset before cleaning
print ('length of data frame before cleaning: ' + str(len(df)))

#drop na - percentage of review text is less than 4% so still sufficient records to work with
df.dropna(inplace=True)

#show length of dataset after cleaning
print ('length of data frame after cleaning: ' + str(len(df)))

#reduce the amount of age records and group in increments of 5
df['Age_Bin'] = pd.cut(df['Age'], range(15,105,5))

#assess word count for each review
df['Word_Count'] = df['Review Text'].apply(len)

#reset index to keep data-set clean
df.reset_index(inplace=True, drop=True)
#show new version of dataframe
df.head()
sns.set_style("white")
sns.set_style("ticks")

ax1 = plt.subplot(121)
df['Rating'].plot(kind='hist', bins=5, align='mid', 
               density=True, figsize=(figsize), ax=ax1, color='r', alpha=0.5, grid=True)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Frequency', fontsize=fontsize)

ax2 = plt.subplot(122)
df['Age'].plot(kind='hist', bins=30, align='mid', 
               density=True, figsize=(figsize), ax=ax2, color='r', alpha=0.5, grid=True)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Frequency', fontsize=fontsize)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(131)
df.groupby(['Division Name']).size().plot(kind='bar', 
                                          width=0.85, figsize=(22,7), ax=ax1, color='r', alpha=0.5, grid=True)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')

ax2 = plt.subplot(132)
df.groupby(['Department Name']).size().sort_values(ascending=False).plot(kind='bar', 
                                          width=0.85, figsize=(22,7), ax=ax2, color='r', alpha=0.5, grid=True)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')

ax3 = plt.subplot(133)
df.groupby(['Class Name']).size().sort_values(ascending=False).plot(kind='bar', 
                                          width=0.85, figsize=(22,7), ax=ax3, color='r', alpha=0.5, grid=True)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')

plt.tight_layout()
plt.show()
#remove isolated records < 2
df = df[df['Class Name'].str.contains(r'Casual bottoms|Chemises') == False]
plt.figure(figsize=(20,8))
sns.set(font_scale=1.3)

ax1 = plt.subplot(121)
sns.heatmap(pd.crosstab(df['Class Name'], df['Rating']), 
            annot=True, cmap='Greens', linewidths=1, fmt='.2f', 
            annot_kws={"size": 18}, ax=ax1, cbar_kws={'label':'', 'orientation':'vertical'})
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.ylabel('')

ax2 = plt.subplot(122)
sns.heatmap(pd.crosstab(df['Class Name'], df['Department Name']), 
            annot=True, cmap='Greens', linewidths=1, fmt='.2f', 
            annot_kws={"size": 18}, ax=ax2, cbar_kws={'label':'', 'orientation':'vertical'})
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=90)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()
plt.figure(figsize=(24,4))
sns.heatmap(pd.crosstab(df['Rating'], df['Age_Bin']), 
            annot=True, cmap='Greens', linewidths=1, fmt='.2f', annot_kws={"size": 16})
plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()
plt.figure(figsize=(24,7))
sns.heatmap(pd.crosstab(df['Class Name'], df['Age_Bin']), 
            annot=True, cmap='Greens', linewidths=1, fmt='.2f', annot_kws={"size": 14})
plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()
sns.set_style("white")
sns.set_style("ticks")

ax1 = plt.subplot(131)
df.groupby(['Age_Bin'])['Word_Count'].agg('median').plot(kind='bar', 
                                          width=0.85, figsize=(22,7), color='r', alpha=0.5, grid=True, ax=ax1)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.axhline(df.groupby(['Age_Bin'])['Word_Count'].agg('median').mean(), color='k', linestyle='--', alpha=0.5, linewidth=3)

ax2 = plt.subplot(132)
df.groupby(['Rating'])['Word_Count'].agg('median').plot(kind='bar', 
                                          width=0.85, figsize=(22,7), color='r', alpha=0.5, grid=True, ax=ax2)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.axhline(df.groupby(['Age_Bin'])['Word_Count'].agg('median').mean(), color='k', linestyle='--', alpha=0.5, linewidth=3)

ax3 = plt.subplot(133)
df.groupby(['Class Name'])['Word_Count'].agg('median').plot(kind='bar', 
                                          width=0.85, figsize=(22,7), color='r', alpha=0.5, grid=True, ax=ax3)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = 'k'
plt.grid(linewidth=1, color='lightgrey', linestyle='--', alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.axhline(df.groupby(['Age_Bin'])['Word_Count'].agg('median').mean(), color='k', linestyle='--', alpha=0.5, linewidth=3)

plt.tight_layout()
plt.show()
df.reset_index(inplace=True, drop=True)
#clean review text and remove any punctuation that will impact the sentiment analysis piece

cleaned = []
for i in range(len(df['Review Text'])):
    clean = df['Review Text'][i].translate(str.maketrans('', '', string.punctuation))
    clean = clean.lower()
    cleaned.append(clean)
#insert cleaned list into original dataframe
df.insert(loc=0, column="Cleaned", value=cleaned) 
sid = SentimentIntensityAnalyzer()
wnl = WordNetLemmatizer()

review_list = []

#tokenize each word and remove any stop words to further increase sentiment classification accuracy
for i in range(len(df["Cleaned"])):
    tokens = [w for w in word_tokenize(str(df["Cleaned"][i])) if w.isalpha()]
    no_stops = [t for t in tokens if t not in stopwords.words('english')]
    lemmatized = [wnl.lemmatize(t) for t in no_stops]
    lemmatized = str(lemmatized).replace("'", "")
    lemmatized = lemmatized.replace("[", "")
    lemmatized = lemmatized.replace("]", "")
    review_list.append(lemmatized)
#insert cleaned list into original dataframe
df.insert(loc=0, column="Revised_Reviews", value=review_list)
revised_polScores = []

for i in range(len(df["Revised_Reviews"])):
    polscores = sid.polarity_scores(df["Revised_Reviews"][i])
    revised_polScores.append(polscores)
    polscores_df = pd.DataFrame(revised_polScores)
df = df.join(polscores_df, how='outer')
df.rename(columns={'compound':'Compound', 'neg':'Negative', 'neu':'Neutral', 'pos':'Positive'}, inplace=True)
def sentiment(df):
    if df["Positive"] > df["Negative"]:
        return "Positive"
    elif df["Positive"] < df["Negative"]:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df.apply(sentiment, axis=1)
df.head(10)
from wordcloud import WordCloud, STOPWORDS
negative = df[df['Sentiment'] == 'Negative']
positive = df[df['Sentiment'] == 'Positive']
stopwords_1 = set(STOPWORDS)
k = (' '.join(negative["Revised_Reviews"]))
wordcloud = WordCloud(width = 1500, height = 500, collocations=False, 
                      max_words=300, stopwords=stopwords_1, relative_scaling=0.2).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.grid(False)
plt.tight_layout()
plt.show()
stopwords_1 = set(STOPWORDS)
k = (' '.join(positive["Revised_Reviews"]))
wordcloud = WordCloud(width = 1500, height = 500, collocations=False, 
                      max_words=300, stopwords=stopwords_1, relative_scaling=0.2).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.grid(False)
plt.tight_layout()
plt.show()
#assess correlations using various methods - all provide a similar consistancy
fig, (ax1, ax2, ax3) = plt.subplots(sharey=True, sharex=True, nrows=3, ncols=1, figsize=(24,20))

sns.set(font_scale=1.4)

sns.heatmap(df.corr(method='pearson'), annot=True, cmap='Greens', linewidths=1, fmt='.2f', ax=ax1, annot_kws={"size": 16})

ax1.set_title("Pearson")

sns.heatmap(df.corr(method='spearman'), annot=True, cmap='Greens', linewidths=1, fmt='.2f', ax=ax2, annot_kws={"size": 16})

ax2.set_title("Spearman")

sns.heatmap(df.corr(method='kendall'), annot=True, cmap='Greens', linewidths=1, fmt='.2f', ax=ax3, annot_kws={"size": 16})

ax3.set_title("Kendall")

# plt.tight_layout()
plt.show()
vectorizer = TfidfVectorizer("english")
X = vectorizer.fit_transform(df['Cleaned'])
y = df['Recommended IND']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
clf = clf = MLPClassifier(activation='logistic', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(45, 45, 45), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=True, warm_start=False)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 2, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
print('Standard Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
sns.set_style("white")
sns.set_style("ticks")

plt.figure(figsize=(20,6))
plt.plot(clf.loss_curve_, color='r', alpha=0.5, linewidth=4)
plt.title('Loss', fontsize=16)
plt.ylabel(s='Loss', fontsize=16)
plt.xlabel(s='Iteration', fontsize=16)
plt.tight_layout()
plt.show()
clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=50, max_depth=8, max_features='sqrt', subsample=0.8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='roc_auc')
print('Standard Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
