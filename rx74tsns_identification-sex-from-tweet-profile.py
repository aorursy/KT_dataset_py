import pandas as pd

import re

import nltk # natural language tool kit

nltk.download("stopwords")      # corpus diye bir kalsore indiriliyor

from nltk.corpus import stopwords  # sonra ben corpus klasorunden import ediyorum



import nltk as nlp

lemma = nlp.WordNetLemmatizer()



import matplotlib

%matplotlib inline  

matplotlib.get_backend()



from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
# Module

import numpy as np 

import pandas as pd

import random

import time

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn import svm

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.stats import multivariate_normal as mvn

from sklearn.mixture import GaussianMixture

import scipy as sp

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

import math

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTEENN

from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.under_sampling import NeighbourhoodCleaningRule

from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import TomekLinks

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

from sklearn.model_selection import cross_val_score

import re

import string

import collections

from sklearn.feature_extraction.text import CountVectorizer

import nltk

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
# %% import twitter data

df = pd.read_csv("/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding = "latin1")

df_male = df[df["gender"] == "male"]

df_female = df[df["gender"] == "female"]

df = pd.concat([df_male,df_female])
def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s



df['Tweets'] = [cleaning(s) for s in df['text']]

df['Description'] = [cleaning(s) for s in df['description']]



from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

df['Tweets'] = df['Tweets'].str.lower().str.split()

df['Tweets'] = df['Tweets'].apply(lambda x : [item for item in x if item not in stop])
for i in range(df.shape[1]):

    df[df.columns[i]] = [cleaning(s) for s in df[df.columns[i]]]
df = df[df["tweet_location"]!="nan"]
df["tweet_location"].value_counts()
df["tweet_location"].str.contains("uk")
df["tweet_location"].replace("london","uk")
df.gender.value_counts()

male = df[df['gender'] == 'male']

female = df[df['gender'] == 'female']

male_words = pd.Series(' '.join(male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

female_words = pd.Series(' '.join(female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

male_words = male_words.iloc[1:]

female_words = female_words.iloc[1:]
fig = plt.figure()

fig.patch.set_facecolor('white')

#plt.style.use(['white_background'])

plt.title("Female Tweet Word Count")

plt.xlabel("Tweet Word")

plt.ylabel("Count")

female_words.plot.bar(color="salmon")
fig = plt.figure()

fig.patch.set_facecolor('white')

#plt.style.use(['dark_background'])

plt.title("male Tweet Word Count")

plt.xlabel("Tweet Word")

plt.ylabel("Count")

male_words.plot.bar(color="mediumturquoise")
import collections

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud

from wordcloud import STOPWORDS
Vectorize = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', max_features=35000)

X = Vectorize.fit_transform(df["Description"])

y = df.gender 

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.values)
#split dataset 

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
models = []

models.append(("k Nearest Neighbor",KNeighborsClassifier(n_neighbors=5)))

models.append(("Decision Tree",tree.DecisionTreeClassifier()))

models.append(("Random Forest",RandomForestClassifier(n_estimators=100, max_depth=2)))

models.append(("Logistic Regression",LogisticRegression()))

models.append(("Naive Bayes",MultinomialNB()))
def cross_validate_evaluate(algorithm):

    

    # Build model

    clf = algorithm

    # Not Need

    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),

               'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    

    # Stratified k-Fold 

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # Evaluation Indicator

    score_funcs = [

        'accuracy',

        'precision',

        'recall',

        'f1',

    ]

    # Cross Validation 

    scores = cross_validate(clf, X, y, cv=skf, scoring=score_funcs)

    print('accuracy:', scores['test_accuracy'].mean())

    print('precision:', scores['test_precision'].mean())

    print('recall:', scores['test_recall'].mean())

    print('f1:', scores['test_f1'].mean())

    

    #return scores



#if __name__ == '__main__':

#    main()

models_index=0

name_index=1

for models_index in range(len(models)):

    print("-----------"+str(models[models_index][name_index-1])+"-----------")

    cross_validate_evaluate(models[models_index][name_index])
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
###Model Lasso regression

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_X, train_y)

rmse_cv(model_lasso).mean()
index_list = []

for i in range(len(model_lasso.coef_)):

    index_list.append(i)
coef_dict = dict(zip(index_list,model_lasso.coef_))
tfidf_name = Vectorize.get_feature_names()

for i in range(len(coef_dict)):

    if coef_dict.get(i) > abs(0):

        print(tfidf_name[i] + " : " + str(coef_dict[i]))

        Vectorize.get_feature_names()
print(model_lasso.intercept_)