# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



#importing sns 

import seaborn as sns

sns.set(style="darkgrid")

import matplotlib.pyplot as plt



#nltk stopwords 

from nltk.corpus import stopwords







import scikitplot as skplt

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/content_classification.csv',names=['Text','Label'])

df.head()
#removing stop words

stop = stopwords.words('english')

df['Text']=df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df.head()
labels = df.Label.unique()

labels
ax = sns.countplot(x="Label", data=df)
#different labels with its count  

count_Class=pd.value_counts(df.Label, sort= True)

#plotting different label value % in pie chart 

# Data to plot

sizes = [count_Class[0], count_Class[1],count_Class[2],count_Class[3],count_Class[4],count_Class[5],count_Class[6]]

colors = ['gold', 'yellowgreen','lightcoral', 'lightskyblue','yellow','green'] 

explode = (0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1)  # explode 1st slice

 

# Plot

plt.pie(sizes,colors=colors,explode=explode,labels=labels,

        autopct='%1.1f%%', shadow=True, startangle=240)

plt.axis('equal')

plt.show()
df_labels = df['Label']

df_labels.head()
df.head()
#splitting the test and train data 

trainset, testset, trainlabel, testlabel = train_test_split(df, df_labels, test_size=0.33, random_state=42)
#extracting n-grams from the text data

countvect= CountVectorizer(ngram_range=(2,2),)

x_counts = countvect.fit(trainset.Text)

#preparing for training set

x_train_df =countvect.transform(trainset.Text)

#preparing for test set

x_test_df = countvect.transform(testset.Text)
#Creating the model using naive bayes

clf=MultinomialNB()

clf.fit(x_train_df,trainset.Label)

predicted_values = clf.predict(x_test_df)

predictions=dict()

acurracy = accuracy_score(testset.Label,predicted_values)

predictions['Naive Bayes']=acurracy*100

confusionmatrix = confusion_matrix(testset.Label,predicted_values)

print("The accuracy of the model is {}%".format(acurracy*100 ))

print(confusionmatrix)

skplt.metrics.plot_confusion_matrix(testset.Label,predicted_values, normalize=True)

plt.show()