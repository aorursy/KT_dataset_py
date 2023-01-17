#Manipulate data
import pandas as pd
import numpy as np

# make a prediction with a stacking ensemble
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

# sklearn 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
train1 = pd.read_csv('../input/stemming/train_praprocess_stem.csv')
test1 = pd.read_csv('../input/stemming/test_praprocess_stem.csv')
#count_vectorizer = CountVectorizer()
count_vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 1)
train_vectors = count_vectorizer.fit_transform(train1['text'])
test_vectors = count_vectorizer.transform(test1["text"])

## Keeping only non-zero elements to preserve space 
train_vectors.shape
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df = 2, max_df = 0.5)
train_tfidf = tfidf.fit_transform(train1['text'])
test_tfidf = tfidf.transform(test1["text"])

train_tfidf.shape
# # Create classifiers
# rf = RandomForestClassifier()
# et = ExtraTreesClassifier()
# knn = KNeighborsClassifier()
# svc = SVC()
# rg = RidgeClassifier()
# lg = LogisticRegression()
# nb = MultinomialNB()
# ada = AdaBoostClassifier()
# grad = GradientBoostingClassifier()
# xgb = XGBClassifier()
# clf_array = [rf, et, knn, svc, rg, lg, nb, ada, grad, xgb]

# define the base models
level0 = list()
level0.append(('svm', SVC()))
level0.append(('ext', ExtraTreesClassifier()))
level0.append(('ada', AdaBoostClassifier()))
level0.append(('grad', GradientBoostingClassifier()))
level0.append(('xgb', XGBClassifier()))
# define meta learner model
grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg = LogisticRegression()
level1 = GridSearchCV(logreg,grid,cv=5)
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(train_vectors, train1['label'])
# Cross Validation
scores = model_selection.cross_val_score(model, train_vectors, train1["label"], cv=5, scoring="f1")
# make a prediction for one example
print('Cross Validation: %.6f' % (scores.mean()))
test_read = pd.read_excel('../input/dataset-bdc/Data Uji BDC.xlsx')
template = pd.read_csv('../input/dataset-bdc/template jawaban BDC.csv')
df = template['ID']
sample_submission = pd.DataFrame()
sample_submission["ID"] = test_read["ID"]
sample_submission["prediksi"] = model.predict(test_vectors) 
sample_submission = pd.merge(df, sample_submission)
sample_submission.to_csv("stacking_(92.83).csv", index=False)
sample_submission.head()