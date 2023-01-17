%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


!pip install -q wordcloud
import wordcloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
%load_ext autoreload
%autoreload 2
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('../input/fake-news-dataset/news.csv')
data.head()
data['label'].value_counts()   # No class imbalance
data = data.rename(columns = {'Unnamed: 0': 'Id'})     # Replacing the unnamed column with 'Id'
# Creating new feature : Number of words in news 'title'

def num_words(string):
    words = string.split()
    return len(words)

data['num_word_title'] = data['title'].apply(num_words)

print(data.groupby(data['label']).mean())

cols = ['title','num_word_title','text', 'label']
data = data[cols]
data.head()
data[data['num_word_title']>25].groupby('label').count()    # This clearly shows if title length is more than 25, it's highly likely to be a fake news.
# Function to split data into train and test set
def train_test_split(df, train_percent=.80, validate_percent=.20, seed=10):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    test = df.iloc[perm[train_end:]]
    return train, test
train, test = train_test_split(data[['num_word_title','text','label']], seed = 12)
train.shape, test.shape
# Necessary for lemmatization
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
# CountVectorizer
count_vectorizer = CountVectorizer(stop_words = 'english', tokenizer=LemmaTokenizer(), 
                                   ngram_range = (1,2), dtype = np.uint8)

count_train = count_vectorizer.fit_transform(train['text'].values)
count_test = count_vectorizer.transform(test['text'].values)
"""
We won't use TfidfVectorizer. However, if any one wants to use it, pre-processing step is similar.
# TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, tokenizer=LemmaTokenizer(), 
                                   ngram_range = (1,2), dtype = np.float32)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(train['text'].values)
# Transform the test data
tfidf_test = tfidf_vectorizer.transform(test['text'].values)
"""
from sklearn.decomposition import TruncatedSVD
lsa_count = TruncatedSVD(n_components = 400, random_state = 20)
lsa_count.fit(count_train)
print(lsa_count.explained_variance_ratio_.sum())          # Explained_variance = 84.66 %
#count_train_df = pd.DataFrame.sparse.from_spmatrix(count_train, columns = count_vectorizer.get_feature_names())
count_train_lsa = pd.DataFrame(lsa_count.transform(count_train))
count_test_lsa = pd.DataFrame(lsa_count.transform(count_test))
# Adding number of words in news title as a feature
count_train_lsa['num_word_title'] = train['num_word_title'] / data['num_word_title'].max()
count_test_lsa['num_word_title'] = test['num_word_title'] / data['num_word_title'].max()
count_train_lsa.fillna(count_train_lsa.mean(), inplace = True)
count_test_lsa.fillna(count_test_lsa.mean(), inplace = True)
count_train_lsa.shape
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
# Trying out with Gaussian Naive Bayes and CountVectorizer model
# Here we are iteratively doing grid search with different hyperparameters in an informed manner.
# params_NB = {'var_smoothing' : [1e-1, 1, 40, 100, 1000]} # var_smoothing = 40 gives the best result.
params_NB = {'var_smoothing' : [20,25,30,35,40,45, 50, 55, 60, 80]} # var_smoothing = 55 gives the best result.


clf_NB = GridSearchCV(estimator = GaussianNB(),param_grid = params_NB, cv = 3, refit = True, scoring = 'accuracy', n_jobs = 4)
clf_NB.fit(count_train_lsa, train['label'])
clf_NB.best_params_
clf_NB.best_score_
test_count_pred_NB = clf_NB.predict(count_test_lsa)
accuracy_score(test['label'], test_count_pred_NB)
cm_NB = confusion_matrix(test['label'], test_count_pred_NB, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_NB/ np.sum(cm_NB),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_NB = classification_report(test['label'], test_count_pred_NB, labels = ['FAKE','REAL'], output_dict = True)
count_report_NB = pd.DataFrame(count_report_NB).transpose()
count_report_NB
from sklearn.linear_model import LogisticRegression

params_LR = {'C' : [10, 5, 1,0.7, 0.5,0.3]}

# C = 0.5 gives the best result after Grid Search.


clf_LR = GridSearchCV(estimator = LogisticRegression(class_weight = 'balanced', random_state = 6),param_grid = params_LR, 
                      cv = 3, refit = True, scoring = 'accuracy', n_jobs = 4)
clf_LR.fit(count_train_lsa, train['label'])
clf_LR.best_score_
clf_LR.best_params_
test_count_pred_LR = clf_LR.predict(count_test_lsa)
cm_LR = confusion_matrix(test['label'], test_count_pred_LR, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_LR/ np.sum(cm_LR),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_LR = classification_report(test['label'], test_count_pred_LR, labels = ['FAKE','REAL'], output_dict = True)
count_report_LR = pd.DataFrame(count_report_LR).transpose()
count_report_LR
from sklearn.neighbors import KNeighborsClassifier
# Here we are iteratively doing grid search with different hyperparameters in an informed manner.
# params_knn = {'n_neighbors' : [2, 4, 8, 16] } # We obtain 4 as the best hyperparameter.
params_knn = {'n_neighbors' : [3,4,5,6,7] }     # 5 is the final tuned hyperparameter.
clf_knn = GridSearchCV(estimator = KNeighborsClassifier(algorithm = 'ball_tree'), param_grid = params_knn, 
                       scoring = 'accuracy', n_jobs = 4, cv = 3, refit = True, verbose = 3)
clf_knn.fit(count_train_lsa, train['label'])
clf_knn.best_score_
clf_knn.best_params_
test_count_pred_knn = clf_knn.predict(count_test_lsa)
cm_knn = confusion_matrix(test['label'], test_count_pred_knn, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_knn/ np.sum(cm_knn),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_knn = classification_report(test['label'], test_count_pred_knn, labels = ['FAKE','REAL'], output_dict = True)
count_report_knn = pd.DataFrame(count_report_knn).transpose()
count_report_knn
from sklearn.svm import SVC
# Here we are iteratively doing grid search with different hyperparameters in an informed manner.
# params_svc = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C' : [0.1, 1, 50]} # 'rbf' and 50 give the best combination
# params_svc = {'kernel' : ['rbf', 'sigmoid'], 'C' : [10, 30, 50,100]}  # 100 and 'rbf' give the best combination.
params_svc = {'kernel' : ['rbf', 'sigmoid'], 'C' : [100, 150, 200]}   # 100 and 'rbf' give the best combination.
clf_svc = GridSearchCV(estimator = SVC(class_weight = 'balanced', random_state = 6), param_grid = params_svc, 
                       scoring = 'accuracy', n_jobs = 4, cv = 3, refit = True, verbose = 2)
clf_svc.fit(count_train_lsa, train['label'])
clf_svc.best_params_
clf_svc.best_score_
test_count_pred_svc = clf_svc.predict(count_test_lsa)
cm_svc = confusion_matrix(test['label'], test_count_pred_svc, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_svc/ np.sum(cm_svc),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_svc = classification_report(test['label'], test_count_pred_svc, labels = ['FAKE','REAL'], output_dict = True)
count_report_svc = pd.DataFrame(count_report_svc).transpose()
count_report_svc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
params_LDA = {'solver' : ['svd', 'lsqr', 'eigen'], 'shrinkage' : ['auto', None]}

# (shrinkage = None and solver = svd) give the best result.
clf_LDA = GridSearchCV(estimator = LinearDiscriminantAnalysis(), param_grid = params_LDA, scoring = 'accuracy', n_jobs = 4,
                       cv = 3, refit = True, verbose = 2)
clf_LDA.fit(count_train_lsa, train['label'])
clf_LDA.best_params_
clf_LDA.best_score_
test_count_pred_LDA = clf_LDA.predict(count_test_lsa)
cm_LDA = confusion_matrix(test['label'], test_count_pred_LDA, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_LDA/ np.sum(cm_LDA),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_LDA = classification_report(test['label'], test_count_pred_LDA, labels = ['FAKE','REAL'], output_dict = True)
count_report_LDA = pd.DataFrame(count_report_LDA).transpose()
count_report_LDA
from sklearn.tree import DecisionTreeClassifier
params_dt = {'criterion' : ['entropy'], 'min_samples_split' : [2, 4, 8, 16, 32], 
             'min_samples_leaf' : [1,2,4,8],'max_depth' : [4, 7, 10], 'max_features' : ['sqrt', None],  'class_weight' : ['balanced']}

# (max_depth = 7, min_samples_leaf = 4, min_samples_split = 32, max_features = None) give the best result.
clf_dt = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 7), param_grid = params_dt, 
                      scoring = 'accuracy', n_jobs = 4, cv = 3, refit = True, verbose = 3)
clf_dt.fit(count_train_lsa, train['label'])
clf_dt.best_params_
clf_dt.best_score_
test_count_pred_dt = clf_dt.predict(count_test_lsa)
cm_dt = confusion_matrix(test['label'], test_count_pred_dt, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_dt/ np.sum(cm_dt),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_dt = classification_report(test['label'], test_count_pred_dt, labels = ['FAKE','REAL'], output_dict = True)
count_report_dt = pd.DataFrame(count_report_dt).transpose()
count_report_dt
from sklearn.ensemble import RandomForestClassifier
params_RF = {'n_estimators' : [400, 1000, 1600], 'criterion' : ['entropy'], 'min_samples_split' : [2, 4, 8, 16], 
             'min_samples_leaf' : [1,2], 'class_weight' : ['balanced']}  

# (min_samples_leaf = 1, min_samples_split = 4 , n_estimators = 1000) give the best result.
clf_RF = GridSearchCV(estimator = RandomForestClassifier(oob_score = True, random_state = 7), param_grid = params_RF, 
                      scoring = 'accuracy', n_jobs = 4, cv = 3, refit = True, verbose = 2)
clf_RF.fit(count_train_lsa, train['label'])
clf_RF.best_params_
clf_RF.best_score_
test_count_pred_RF = clf_RF.predict(count_test_lsa)
cm_RF = confusion_matrix(test['label'], test_count_pred_RF, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_RF/ np.sum(cm_RF),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_RF = classification_report(test['label'], test_count_pred_RF, labels = ['FAKE','REAL'], output_dict = True)
count_report_RF = pd.DataFrame(count_report_RF).transpose()
count_report_RF
from sklearn.ensemble import AdaBoostClassifier
# Here we are iteratively doing grid search with different hyperparameters in an informed manner.
# params_ada = {'n_estimators' : [50, 100, 500], 'learning_rate' : [1, 0.3]} # 500 and 1 give the best combination.
params_ada = {'n_estimators' : [500, 1000, 1500, 2000], 'learning_rate' : [1]} 
# (n_estimators = 1500, learning_rate = 1) is the best combination.
clf_ada = GridSearchCV(estimator = AdaBoostClassifier(random_state = 8), param_grid = params_ada, 
                       scoring = 'accuracy', n_jobs = 4, cv = 3, refit = True, verbose = 2)
clf_ada.fit(count_train_lsa, train['label'])
clf_ada.best_params_
clf_ada.best_score_
test_count_pred_ada = clf_ada.predict(count_test_lsa)
cm_ada = confusion_matrix(test['label'], test_count_pred_ada, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_ada/ np.sum(cm_ada),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_ada = classification_report(test['label'], test_count_pred_ada, labels = ['FAKE','REAL'], output_dict = True)
count_report_ada = pd.DataFrame(count_report_ada).transpose()
count_report_ada
import lightgbm as lgb
# Here we are iteratively doing grid search with different hyperparameters in an informed manner.
#params_lgb = {'n_estimators' : [100, 400, 800], 'learning_rate' : [0.03, 0.1], 'min_child_samples' : [4, 12, 24]}
# 800, 0.1 , 4 are the best ones.
#params_lgb = {'n_estimators' : [1200, 2400], 'learning_rate' : [ 0.1], 'min_child_samples' : [2,4]}  
# (n_estimators = 2400 and min_child_samples = 2) are the best ones.
# Let's just try with n_estimators 3600.
params_lgb = {'n_estimators' : [3600], 'learning_rate' : [ 0.1], 'min_child_samples' : [1,2]} 
# (n_estimators = 3600, learning_rate = 0.1, min_child_samples = 1) perform best.
clf_lgb = GridSearchCV(estimator = lgb.LGBMClassifier(), param_grid = params_lgb, scoring = 'accuracy', n_jobs = 4,
                       cv = 3, refit = True, verbose = 2)
clf_lgb.fit(count_train_lsa, train['label'])
clf_lgb.best_params_
print(clf_lgb.best_score_)
test_count_pred_lgb = clf_lgb.predict(count_test_lsa)   


cm_lgb = confusion_matrix(test['label'], test_count_pred_lgb, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_lgb/ np.sum(cm_lgb),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_lgb = classification_report(test['label'], test_count_pred_lgb, labels = ['FAKE','REAL'], output_dict = True)
count_report_lgb = pd.DataFrame(count_report_lgb).transpose()
count_report_lgb
import catboost as cb
params_cat = {'n_estimators' : [800,1000,2000] }   #  2000 is the best one
clf_cat = GridSearchCV(estimator = cb.CatBoostClassifier(task_type = 'GPU', learning_rate = 0.2, max_depth = 6), param_grid = params_cat,
                       scoring = 'accuracy', n_jobs = 1, cv = 3, refit = True, verbose = 2 )
clf_cat.fit(count_train_lsa, train['label'])
clf_cat.best_params_
test_count_pred_cat = clf_cat.predict(count_test_lsa)

cm_cat = confusion_matrix(test['label'], test_count_pred_cat, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_cat/ np.sum(cm_cat),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_cat = classification_report(test['label'], test_count_pred_cat, labels = ['FAKE','REAL'], output_dict = True)
count_report_cat = pd.DataFrame(count_report_cat).transpose()
count_report_cat
from tpot import TPOTClassifier
clf_tpot = TPOTClassifier(generations = 6, population_size = 6, random_state = 3, cv = 2, n_jobs = 4)
clf_tpot.fit(count_train_lsa, train['label'])
clf_tpot.fitted_pipeline_
test_count_pred_tpot = clf_tpot.predict(count_test_lsa)
cm_tpot = confusion_matrix(test['label'], test_count_pred_tpot, labels = ['FAKE', 'REAL'])
sns.heatmap(cm_tpot/ np.sum(cm_tpot),fmt='.2%', annot=True, cmap = 'Blues', xticklabels = ['FAKE', 'REAL'], yticklabels = ['FAKE', 'REAL'])

from sklearn.metrics import classification_report

count_report_tpot = classification_report(test['label'], test_count_pred_tpot, labels = ['FAKE','REAL'], output_dict = True)
count_report_tpot = pd.DataFrame(count_report_tpot).transpose()
count_report_tpot