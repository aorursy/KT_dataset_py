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
from nltk.corpus import stopwords
import re
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
sns.set_style("white")
sns.set_style("ticks")
fontsize = 22
figsize = (24,6)
df = pd.read_csv("../input/spam.csv", encoding = "ISO-8859-1", engine='python')
#We can quickly see that there are 3 columns that aren't usable
df.head()
#First, we drop the 3 columns with NAN values. Axis=1 is referencing the column and when inplace=True is passed, the data is renamed in place (it returns nothing), 
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#We then rename our columns to something more user friendly - you can rename this to whatever you feel comfortable with
df.rename(columns={'v1':'Label', 'v2':'Text'}, inplace=True)
#Label column to have sentance case text
df['Label'] = df['Label'].str.title()
#We will need to create a new column to help define our target variable.
df['Target'] = df['Label'].map({'Ham':0, 'Spam':1})
#Re-Order Columns
df = df[['Text', 'Label', 'Target']]
#I thought it might be interesting to see the amount of words contained in each text - we can explore shortly
df['Word_Count'] = df['Text'].apply(len)
df.head()
ax1 = plt.subplot(141)
df.groupby(['Label']).size().plot(kind='bar', width=0.85, figsize=figsize, ax=ax1, colormap='viridis')
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.xlabel('')
plt.title('Spam Vs Ham Value Counts', fontsize=fontsize)

ax2 = plt.subplot(142)
df.groupby(['Label'])['Word_Count'].sum().div(df.groupby(['Label']).size()).plot(kind='bar', 
                                                                    width=0.85, figsize=figsize, ax=ax2, colormap='viridis')
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.xlabel('')
plt.title('Spam Vs Ham Words Per Text', fontsize=fontsize)

ax3 = plt.subplot(143)
df[df['Label'] == 'Spam']['Word_Count'].plot(kind='hist', bins=30, colormap='viridis', ax=ax3, figsize=figsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel('')
plt.title('Len Count // Spam', fontsize=fontsize)

ax4 = plt.subplot(144)
df[df['Label'] == 'Ham']['Word_Count'].plot(kind='hist', bins=30, colormap='viridis', ax=ax4, figsize=figsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel('')
plt.title('Len Count // Ham', fontsize=fontsize)

plt.tight_layout()
plt.show()
cleaned = []
for i in range(len(df['Text'])):
    clean = df['Text'][i].translate(str.maketrans('', '', string.punctuation))
    clean = clean.lower()
    cleaned.append(clean)
df.insert(loc=0, column="Cleaned", value=cleaned)
df.head()
#Instantiate the TfidfVectorizer
vectorizer = TfidfVectorizer("english")
#Create your features
X = vectorizer.fit_transform(df['Cleaned'])
#Create your target
y = df['Target']
#Split the data into training and testing // I typically like training on 80% of the data, but feel free to play around with the different amounts.
#Random States essentially allows you to re-produce the same results time and time again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
models = []
models.append(('LR1', LogisticRegression()))
models.append(('LR2', LogisticRegression(C=100, penalty='l2', solver='liblinear')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))
models.append(('DTC', DecisionTreeClassifier()))
# models.append(('GNB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
# models.append(('MLP', MLPClassifier()))
models.append(('ETC', ExtraTreeClassifier()))
# models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('BCL', BaggingClassifier()))
models.append(('SVC', SVC(kernel='sigmoid', gamma=1.0)))

# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'

parameters = ["accuracy", "average_precision", "f1", "f1_micro", 'f1_macro', 'f1_weighted', 'precision', "roc_auc"]

for name, model in models:
    kfold = model_selection.KFold(n_splits = 15, random_state = 7)
    cv_results = model_selection.cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize=(26,8))
fig.suptitle('Algorithm Comparison', fontsize=fontsize)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
#I just picked GBC becuase it's one of my favourites but feel free to choose the one you like working with the most.
clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, max_depth=8, max_features='sqrt', subsample=0.8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='roc_auc')
print('Standard Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
proba = clf.predict_proba(X_train)[:,1]

confusion_matrix = pd.DataFrame(
    confusion_matrix(y_test, predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"])
confusion_matrix
fpr, tpr, threshold = roc_curve(y_train, proba)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'k--', alpha=0.8)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
ml_results_df = pd.DataFrame(results).T
#Below are the results of each classifier using K-Folds, but column names arent helpful
ml_results_df
for number, name in enumerate([x[0] for x in models]):
    ml_results_df = ml_results_df.rename(columns={number:name})
ml_results_df
empty_list = []

for i in ml_results_df.columns:
    averages = ml_results_df[i].mean()
    empty_list.append(averages)
    averages_df = pd.DataFrame(empty_list, columns=['Score'])
ml_classifiers = [x[0] for x in models]
averages_df.insert(loc=0, column='ML_Classifier', value=ml_classifiers)
averages_df.groupby(['ML_Classifier']).sum().sort_values(by='Score', ascending=False).plot(kind='bar', width=0.85, figsize=figsize)
# plt.axvspan(xmax=-0.5, xmin=0.50, color='r', alpha=0.2)
plt.ylim(0.8, 1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel(s='Classifier', fontsize=fontsize)
plt.tight_layout()
plt.show()
from wordcloud import WordCloud, STOPWORDS
spam = df[df['Label'] == 'Spam']
ham = df[df['Label'] == 'Ham']
stopwords_1 = set(STOPWORDS)
k = (' '.join(spam['Cleaned']))
wordcloud = WordCloud(width = 2500, height = 500, collocations=False, 
                      max_words=300, stopwords=stopwords_1, relative_scaling=0.2).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.grid(False)
plt.tight_layout()
plt.show()
stopwords_1 = set(STOPWORDS)
k = (' '.join(ham['Cleaned']))
wordcloud = WordCloud(width = 2500, height = 500, collocations=True, 
                      max_words=300, stopwords=stopwords_1, relative_scaling=0.2).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.grid(False)
plt.tight_layout()
plt.show()
