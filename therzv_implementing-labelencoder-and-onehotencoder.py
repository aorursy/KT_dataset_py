import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # = "from matplotlib import pyplot as plt"
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline
training_variants_df = pd.read_csv("../input/training_variants")
training_text_df = pd.read_csv("../input/training_text", sep="\|\|", header=None, engine='python', skiprows=1, names=["ID","Text"])
training_variants_df.shape
training_variants_df.head(5)
## Check Unique Data in Training Data
print("Unique Classification : {} class".format(len(training_variants_df.Class.unique())))
print("Unique ID : {} ".format(len(training_variants_df.ID.unique())))
print("Unique Variation : {} ".format(len(training_variants_df.Variation.unique())))
print("Unique Gene : {} ".format(len(training_variants_df.Gene.unique())))
plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=training_variants_df, palette="GnBu_d")
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of Gen Mutation", fontsize=18)
plt.show()
training_variants_df.groupby('Class').size()
training_text_df.shape
training_text_df.head(10)
train_full = training_variants_df.merge(training_text_df, how="inner", left_on="ID", right_on="ID")
train_full.head()
train_full.isnull().sum()
train_full.info()
train_full.fillna('xxxxxxxx', inplace=True)
train_full.info()
train_full['Class'].unique()
train_full.info()
X = train_full[['ID', 'Gene', 'Variation', 'Class', 'Text']]
y = train_full['Class']
le = LabelEncoder()
#To Label Encoder
X2 = X.apply(le.fit_transform)
#To One Hot Encoder
ohe = OneHotEncoder()
X3 = ohe.fit_transform(X2).toarray()
X3.shape
X_train, X_test, y_train, y_test = train_test_split(X3, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
%time nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)
