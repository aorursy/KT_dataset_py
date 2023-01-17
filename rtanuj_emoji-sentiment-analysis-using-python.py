import numpy as np
import pandas as pd 
import sys
import emoji
import math
import pickle
from time import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import svm
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import re
import string
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
from collections import defaultdict
import os
import psutil
import os
print(os.listdir("../input"))

from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.feature_extraction.text import CountVectorizer
import re, string

comments2emoji_df=pd.read_csv("../input/comments2emoji_frequency_matrix_cleaned.csv")
reference_df=pd.read_csv("../input/ijstable.csv")
use_cols=['Char','Neg','Neut','Pos']
train_df=pd.read_csv("../input/ijstable.csv",usecols=use_cols)
train_df.head(20)
train_df.drop(train_df.index[0], inplace=True)
train_df.head()
train_df = train_df.reset_index(drop=True)
train_df.head()
train_df["Char"] = train_df["Char"].astype('category')
train_df["Neg"] = train_df["Neg"].astype('float')
train_df["Neut"] = train_df["Neut"].astype('float')
train_df["Pos"] = train_df["Pos"].astype('float')
train_df.info()
train_df["Char_cat"] = train_df["Char"].cat.codes
train_df.head()
dataset=train_df.drop(['Char'],axis=1)
dataset.head()
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
arrays = dataset.values
imp.fit(arrays)
array_imp = imp.transform(arrays)
array_imp.shape
X=array_imp[:,-1]
print(X.shape)
X=X.reshape(-1,1)
Y=array_imp[:,3]
print(Y.shape)
validation_size = 0.20
seed = 7
from sklearn import model_selection
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=7)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
svc = LogisticRegression()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#let's test
x = ["👍"]
arr=np.array(x)
x_encoded=arr.reshape(-1,1)
#x_encoded = x.cat.codes
svc.predict(x_encoded)