# IMPORTING ALL THE NECESSARY LIBRARIES AND PACKAGES

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import string

import math

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV

%matplotlib inline
data = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
data.head()
data = data.drop('Unnamed: 0',1)
data.head()
# Number of Columns and rows

data.shape
#----------------------Basic Statistics and Data Types-------------------------#

# Information on counts, columns, column data types, memory usage, etc

data.info()
# Descriptive Statistics for all columns 

data.describe(include='all')
# Descriptive Statistics for numeric columns 

data.describe()
# Column names

print("Column names:")

print(data.columns)

# Datatypes of each column

print("Datatype of each column:")

print(data.dtypes)
#Unique counts of records for each column

print("Unique Counts for each column:")

print(data.nunique())
# UNIQUE COUNTS OF NAN/NULL EACH COLUMN

print("Unique Counts of nan/null for each column:")

print(data.isnull().sum())
#Remove nan/null form review column 

data = data.dropna(subset=['Review Text'])
#-----------------------------'Review Text' Analysis along with Data visualization---------------------------#
data['Text_length'] = data['Review Text'].apply(len)

data.head()
Bar_plot = ["Rating","Recommended IND"]

adds = 0

f, ax = plt.subplots(1,len(Bar_plot), figsize=(14,4), sharex=False)

for i in range(len(Bar_plot)):

    sns.countplot(x=Bar_plot[adds], data=data,order=data[Bar_plot[adds]].value_counts().index, ax=ax[i])

    ax[i].set_title("Histogram - Distribution for\n{}".format(Bar_plot[adds]))

    ax[i].set_ylabel("Counts")

    ax[i].set_xlabel("{}".format(Bar_plot[adds]))

    adds += 1

ax[1].set_ylabel("")

plt.show()
Bar_plot = ["Division Name","Department Name"]

adds = 0

f, ax = plt.subplots(1,len(Bar_plot), figsize=(14,4), sharex=False)

for i in range(len(Bar_plot)):

    sns.countplot(x=Bar_plot[adds], data=data,order=data[Bar_plot[adds]].value_counts().index, ax=ax[i])

    ax[i].set_title("Histogram - Distribution for\n{}".format(Bar_plot[adds]))

    ax[i].set_ylabel("Counts")

    ax[i].set_xlabel("{}".format(Bar_plot[adds]))

    adds += 1

ax[1].set_ylabel("")

plt.show()
# Distribution Plot for 'Age'

sns.distplot(data['Age'], color="red")

plt.show()

sns.distplot(data['Recommended IND'], color="red")

plt.show()

sns.jointplot(data=data,x='Age', y='Rating',kind='kde')
sns.jointplot(data=data,x='Age', y='Positive Feedback Count',kind='kde')
#------------------------------Converting all the ratings into just 2 classes------------------------#

classification = [

    (data['Rating'] <= 3),

    (data['Rating'] > 3)]

meaning = ['Utmost_3','Greater_than_3']

data['Rating_Class'] = np.select(classification, meaning)


# if the rating are more than 3 stars then the Rating_Class_Ind is given as 1, or else if the stars 

# are less than or equal to 3 it is given as 0



data['Rating_Class_Ind'] = data['Rating_Class'].apply(lambda x: 0 if x=='Utmost_3' else 1)
data.head()
#---------------------------Model Development-----------------------------------#
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from plotly import tools

import xgboost

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings("ignore")
# Data Classification based on 'Rating_Class_Ind'

data_classes = data[(data['Rating_Class_Ind']==1) | (data['Rating_Class_Ind']==0)]

data_classes.head()

print(data_classes.shape)



# Seperate the dataset into X and Y for prediction

x = data_classes['Review Text']

y = data_classes['Rating_Class_Ind']

print(x.head())

print(y.head())
# Fucntion analyzer to remove stop words and non english words

def text_process(text):

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
vocab = CountVectorizer(analyzer=text_process).fit(x)
# Sparse matrix from vocab for indicating each occurence of the word

x = vocab.transform(x)
# train and test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
LR = LogisticRegression()

LR.fit(x_train,y_train)

LR_Pred = LR.predict(x_test)

print("Confusion Matrix for Logistic Regression:")

print(confusion_matrix(y_test,LR_Pred))

print("Score: ",round(accuracy_score(y_test,LR_Pred)*100,2))

print("ROC_AUC Score:", round(roc_auc_score(y_test,LR_Pred)*100,2))

print("Classification Report:")

print(classification_report(y_test,LR_Pred))

lr_cm=confusion_matrix(y_test.values, LR_Pred)

plt.figure(figsize=(10,8))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplot(2,2,1)

plt.title("Logistic Regression")

sns.heatmap(lr_cm, annot = True, cmap="Greens",cbar=False);
xgb = XGBClassifier()

xgb.fit(x_train,y_train)

predxgb = xgb.predict(x_test)

print("Confusion Matrix for XGBoost Classifier:")

print(confusion_matrix(y_test,predxgb))

print("Score: ",round(accuracy_score(y_test,predxgb)*100,2))

print("ROC_AUC Score:", round(roc_auc_score(y_test,predxgb)*100,2))

print("Classification Report:")

print(classification_report(y_test,predxgb))

xgb_cm=confusion_matrix(y_test.values, predxgb)

plt.figure(figsize=(10,8))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplot(2,2,1)

plt.title("XGBoost Classifier")

sns.heatmap(xgb_cm, annot = True, cmap="Greens",cbar=False);

mlp = MLPClassifier()

mlp.fit(x_train,y_train)

predmlp = mlp.predict(x_test)

print("Confusion Matrix for Multilayer Perceptron Classifier:")

print(confusion_matrix(y_test,predmlp))

print("Score:",round(accuracy_score(y_test,predmlp)*100,2))

print("ROC_AUC Score:", round(roc_auc_score(y_test,predmlp)*100,2))

print("Classification Report:")

print(classification_report(y_test,predmlp))
mlp_cm=confusion_matrix(y_test.values, predmlp)

plt.figure(figsize=(10,8))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplot(2,2,1)

plt.title("MLP Classifier")

sns.heatmap(mlp_cm, annot = True, cmap="Greens",cbar=False);
nb = MultinomialNB() 

nb.fit(x_train,y_train)

prednb = nb.predict(x_test)

print("Confusion Matrix for MultinomialNB Classifier:")

print(confusion_matrix(y_test,prednb))

print("Score: ",round(accuracy_score(y_test,prednb)*100,2))

print("ROC_AUC Score:", round(roc_auc_score(y_test,prednb)*100,2))

print("Classification Report:")

print(classification_report(y_test,prednb))

nb_cm=confusion_matrix(y_test.values, prednb)

plt.figure(figsize=(10,8))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplot(2,2,1)

plt.title("MultinomialNB Classifier")

sns.heatmap(nb_cm, annot = True, cmap="Greens",cbar=False);