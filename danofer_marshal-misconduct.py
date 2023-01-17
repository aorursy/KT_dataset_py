# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import sparse

from sklearn.preprocessing import LabelEncoder as LE # warning - using this can result in silly range features, vs using OHE from sklearn or Pandas's get_dummies(). 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path = '../input/'

filename = 'FederalAirMarshalMisconduct.csv'



df = pd.read_csv(path + filename,parse_dates=["Date Case Opened"],infer_datetime_format=True)
df.head()
df.columns

df.rename(columns={"Allegation ":"Allegation", "Field Office ":'FieldOffice'},inplace=True)

print(df.columns)
df["Allegation"].value_counts()
df['Final Disposition'].value_counts()
df['target'].value_counts()
print("A naive majority classifier would get: %.4f Accuracy" % (1833/df.shape[0]))
least_frequent_classes = df['target'].value_counts().tail(8).index
print(df.shape)

df = df.loc[~df.target.isin(least_frequent_classes)]

df.shape[0]
print("Check for nulls in the target column:")

print(df.isnull().sum())

# df.dropna(subset="target",inplace=True,axis=1) # This gives errors on kaggle for some reason ? 

df = df.loc[df.target.notnull()]

print("After cleaning:",df.isnull().sum())
df["Year"] = df['Date Case Opened'].dt.year

df["Month"] = df['Date Case Opened'].dt.month
df.head()
df = df[[ 'FieldOffice', 'Allegation', 'target', 'Year', 'Month']]
# # Encode OHE the FieldOffice:

df = pd.get_dummies( df, columns = ["FieldOffice"] )



# ### ALT:

# df["FieldOffice"] = LE.fit_transform(df["FieldOffice"])
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop("target",axis=1), df.target, random_state=42)
# Bag of words features on the text

tfidf = CountVectorizer(stop_words='english', max_features=200,min_df=3,ngram_range=(1, 2))

tr_sparse = tfidf.fit_transform(X_train["Allegation"])

te_sparse = tfidf.transform(X_test["Allegation"])
X_train = sparse.hstack([X_train.drop("Allegation",axis=1), tr_sparse]).tocsr()

X_test = sparse.hstack([X_test.drop("Allegation",axis=1), te_sparse]).tocsr()
fmodel = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=9, max_features=30,class_weight="balanced").fit(X_train, y_train)

prediction = fmodel.predict(X_test)
# Data is hihgly imbalanced, so accuracy is meaningless. let'sWe could have a look at the AUC, but it's tricker to define for multiclass, so we'll leave it for now) : 

# score = roc_auc_score(y_test, prediction)

# print("AUC on test set: %.2f" % score)



acc_score = accuracy_score(y_test, prediction)

print("Accuracy score on test set: %.2f" % (100.*acc_score))
print(classification_report(y_test, prediction))