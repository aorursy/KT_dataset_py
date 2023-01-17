import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

from string import punctuation
import nltk
from nltk.corpus import stopwords

#Preprocessing
from scipy.stats import uniform
from scipy import interp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


#predictive model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

#metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#get the data
df = pd.read_csv('../input/amazon-music-reviews/Musical_instruments_reviews.csv')
#show the dataframe
df.head()
#show feature data types
df.info()
#show decriptive stats of ratings
df.groupby('overall').describe()
df.isnull().sum()
plt.figure(figsize=(10,5))
plt.title('DISTRIBUTION OF RATINGS', fontsize=18)
sns.countplot(df.overall, palette='deep')
plt.xlabel('Rating')
#show columns
df.columns
df.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'], axis=1, inplace=True)
#show new dataframe
df.head()
#combining summary and reviewtext feature
df['review'] = df['reviewText'] + df['summary']
df.drop(['reviewText', 'summary'], axis=1, inplace=True)
#replace the ratings to sentiments

def num_to_sent(x):
    if (int(x) == 1 or int(x)==2 or int(x)==3):
        return 0
    else:
        return 1
df['overall'] = df.overall.apply(num_to_sent)
#show value counts
plt.figure(figsize=(10,5))
plt.title('COUNTPLOT OF LABELS')
sns.barplot(df.overall.value_counts().index,df.overall.value_counts().values, palette='deep')
#get the stopwords and punctuation
stop = stopwords.words('english')
punc = list(punctuation)
#remove stop words
text_clean = []
for i in range(len(df.review)):
    char_clean = []
    for char in str(df['review'][i]).split():
        char = char.lower()
        if char not in stop:
            char_clean.append(char)
        else:
            continue
    char_clean = ' '.join(char_clean)
    text_clean.append(char_clean)
df['review'] = text_clean
#remove punctuations
text_clean = []
for i in range(len(df.review)):
    char_clean = []
    for char in df['review'][i]:
        char = char.lower()
        if char not in punc:
            char_clean.append(char)
        else:
            continue
    char_clean = ''.join(char_clean)
    text_clean.append(char_clean)
df['review'] = text_clean
#show text sample
df.review[2]
#split the data
X = df.review
y = df.overall
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#RandomSearchCV
# define the parameters to tune
param_dist = {"learning_rate": uniform(0, 2),
              "gamma": uniform(1, 0.000001),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10),
              'n_jobs': range(1,5)}
#instance of RandomSearchCV
rs = RandomizedSearchCV(XGBClassifier(), param_distributions=param_dist, n_iter=3) #25 iterations
model  = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rs)
])
#fit the data
model.fit(X_train, y_train)
#predict the test data
predictions=model.predict(X_test)
print('Model Accuracy: ', round(accuracy_score(y_test, predictions)*100,2), '%')
print(classification_report(y_test, predictions))
#roc plot function
def plot_roc(X_df, y, estemator,n_splits, lns = 100):
    #creating an instance of KFold
    kfold = StratifiedKFold(n_splits=n_splits,shuffle=False)
    #define estemator
    rf = estemator
    #deifne figuresize
    plt.rcParams['figure.figsize'] = (10,5)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,lns)
    i = 1

    for train,test in kfold.split(X,y):
        #get prediction
        prediction = rf.fit(X.iloc[train],y.iloc[train]).predict_proba(X.iloc[test])
        #get the true pos. rate, false positive rate and thresh 
        fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        #get the area under the curve
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plot the tpr and fpr
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1

    #plot the mean ROC
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='gold',
    label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    #setup the labels
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('ROC PLOT', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

from sklearn.ensemble import RandomForestClassifier
model_rf  = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier())
])

#plotting roc curve with 5 number of splits
plot_roc(X, y, estemator=model_rf, n_splits=5)