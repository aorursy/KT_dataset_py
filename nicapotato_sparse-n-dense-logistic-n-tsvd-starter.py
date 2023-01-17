import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Dimensionality Reduction
from sklearn.decomposition import TruncatedSVD

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix, vstack
from nltk.corpus import stopwords

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

df = pd.read_csv("../input/train.tsv", sep="\t", index_col = ["PhraseId"])#.sample(500) # Debugging..
trainlen = df.shape[0]
test_df = pd.read_csv("../input/test.tsv", sep="\t", index_col = ["PhraseId"])#.sample(500)
testdex = test_df.index
print("\nTrain Shape: ",df.shape)
print("Test Shape: ",test_df.shape)

y = df.Sentiment.copy()
df = pd.concat([df.drop("Sentiment",axis=1),test_df], axis=0)
print("All Data Shape: {} Rows, {} Columns".format(*df.shape))
del test_df

# Glimpse at Dataset 
print("Dataset Glimpse")
display(df.head(5))
print("Percent Representation by Sentiment Level")
print(y.value_counts(normalize=True)*100)
# Meta Text Features
df["Phrase"] = df["Phrase"].astype(str) 
df["Phrase"] = df["Phrase"].astype(str).fillna('missing') # FILL NA
df["Phrase"] = df["Phrase"].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
df["Phrase" + '_num_words'] = df["Phrase"].apply(lambda comment: len(comment.split())) # Count number of Words
df["Phrase" + '_num_unique_words'] = df["Phrase"].apply(lambda comment: len(set(w for w in comment.split())))
df["Phrase" + '_words_vs_unique'] = df["Phrase"+'_num_unique_words'] / df["Phrase"+'_num_words'] * 100 # Count Unique Words
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    dtype = np.float32,
    norm='l2',
    min_df=0,
    smooth_idf=False,
    max_features=15000)
# Fit and Transform
word_vectorizer.fit(df.iloc[0:trainlen,:]["Phrase"])
train_word_features = word_vectorizer.transform(df.iloc[0:trainlen,:]["Phrase"])
test_word_features = word_vectorizer.transform(df.iloc[trainlen:,:]["Phrase"])
sent_dummy = pd.get_dummies(df["SentenceId"])
df.drop("SentenceId", axis=1, inplace=True)
print("Before..")
display(df.describe())
dense_variables = [x for x in df.columns if x not in ["PhraseId","SentenceId","Phrase"]]
scaler = MinMaxScaler()
df[dense_variables] = scaler.fit_transform(df[dense_variables])
print("After..")
display(df.describe())
# Fill Missing Values with 0
print("Missing Values Before:\n", df.isnull().sum())
df.fillna(0,inplace=True)
# Sparse Matrix
dense_vars = [x for x in df.columns if x not in ["PhraseId","SentenceId","Phrase"]]
X = hstack([csr_matrix(df.iloc[0:trainlen,:][dense_vars].values), csr_matrix(sent_dummy.iloc[0:trainlen,:]),train_word_features])
test_df = hstack([csr_matrix(df.iloc[trainlen:,:][dense_vars].values),csr_matrix(sent_dummy.iloc[trainlen:,:]), test_word_features])
# del df, sent_dummy, train_word_features, test_word_features; gc.collect();
# Zero Proportion
zero_proportion = ((X.toarray() != 0).sum() / (X.shape[0]*X.shape[1]))
print("Portion of Data that has an value other than 0: {}%".format(round(zero_proportion, 5)))
print("Train Shape: {} Rows and {} Cols".format(*X.shape))
print("Test Shape: {} Rows and {} Cols".format(*test_df.shape))
# Define and Fit
model = LogisticRegression(multi_class = 'ovr', C=1, solver='sag')
model.fit(X,y)
# Predict and Submit
submission = model.predict(test_df)
submission_df = pd.Series(submission).rename("Sentiment")
submission_df.index = testdex
submission_df.to_csv("Logistic_sub.csv",index=True,header=True)
display(submission_df.head())

del model, submission, submission_df
# Define and Fit TruncatedSVD Dimensionality Reduction Model
svd = TruncatedSVD(n_components=50, n_iter=20, random_state=42)
svd.fit(X) 

# Transform
X = svd.transform(X)
test_df = svd.transform(test_df)

# Logistic
model = LogisticRegression(multi_class = 'ovr', C=1)
model.fit(X,y)

# Submit
submission = model.predict(test_df)
submission_df = pd.Series(submission).rename("Sentiment")
submission_df.index = testdex
submission_df.to_csv("TSVD_n_Logistic_sub.csv",index=True,header=True)
submission_df.head()
print("Notebook Runtime: %0.0f seconds"%((time.time() - notebookstart)))