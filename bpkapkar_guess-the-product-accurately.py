# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df= pd.read_csv("/kaggle/input/guess-the-product/train_set.csv")
df.shape
df.head() #to visualize the data
df.info() # to get detailed information ,type of data and check the null values

# data
#to identify unique Vendor Code count

df['Vendor_Code'].value_counts().count()
#to identify unique GL_Code

df['GL_Code'].unique()
fig_dims = (40, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x='Product_Category',data=df)
#to identify Product_Category count that we want to predict

df['Product_Category'].value_counts().count()
df['Product_Category'].value_counts()
fig_dims = (20, 4)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x='GL_Code',data=df)
fig_dims = (30, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x="Product_Category", y="Inv_Amt", data=df,palette='rainbow')
#To apply NLP we have text converted all the string to Lowercase

df["Item_Description"]=df["Item_Description"].str.lower()
textData=df["Item_Description"]
import string

def remove_punctuation(text):

    return text.translate(str.maketrans('','',string.punctuation))

text_clean=textData.apply(lambda text:remove_punctuation(text))
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
def stopwords_(text):

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

text_clean = text_clean.apply(lambda text: stopwords_(text))
text_clean.head()
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

def lemma(text):

    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
import nltk

from nltk.stem import WordNetLemmatizer   

lemmatizer = WordNetLemmatizer() 

text_clean=text_clean.apply(lambda text: lemma(text))
text_clean.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf = TfidfVectorizer(stop_words = 'english')
tfidf.fit(text_clean)
X = tfidf.transform(text_clean)
text_df = pd.DataFrame(X.toarray())
text_df = pd.DataFrame(X.toarray())
text_df = pd.DataFrame(X.toarray())
text_df.head()
from sklearn import preprocessing

# encode categorical variables using Label Encoder



# select all categorical variables

df_categorical = df[['Vendor_Code','GL_Code']]

df_categorical.head()
# apply Label encoder to df_categorical



le = preprocessing.LabelEncoder()

df_categorical = df_categorical.apply(le.fit_transform)

df_categorical.head()
df.drop(['Inv_Id','Item_Description','Vendor_Code','GL_Code'], axis=1, inplace=True)

df.head()
# concat df_categorical with original df

df = pd.concat([df_categorical,text_df,df], axis=1)

df.head()
# Importing train-test-split 

from sklearn.model_selection import train_test_split
X = df.drop(['Product_Category'],axis=1)



# Putting response variable to y

y = df['Product_Category']
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state = 99)

X_train.head()
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

Classifier = DecisionTreeClassifier()

Classifier.fit(X_train, y_train)
Classifier.feature_importances_
# Let's check the evaluation metrics of our default model for train



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred = Classifier.predict(X_train)



print(confusion_matrix(y_train,y_pred))

print(accuracy_score(y_train,y_pred))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = Classifier.predict(X_test)



# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
from sklearn.ensemble import RandomForestClassifier
Regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
Regressor.fit(X_train, y_train) 
Regressor.feature_importances_
# Let's check the evaluation metrics of our default model for train



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred = Regressor.predict(X_train)



print(confusion_matrix(y_train,y_pred))

print(accuracy_score(y_train,y_pred))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = Regressor.predict(X_test)



# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
# Accuracy Achieved is nearly 99.43 %