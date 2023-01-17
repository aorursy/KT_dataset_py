# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as np

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import nltk

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

countvec=CountVectorizer()

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

log=LogisticRegression()

knn=KNeighborsClassifier()

rf=RandomForestClassifier()

dt=DecisionTreeClassifier()

nom=MultinomialNB()

list_of_models=[log,knn,dt,rf,nom]
def clean_sentance(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)
data=pd.read_json('/kaggle/input/whats-cooking-kernels-only/train.json')

test=pd.read_json('/kaggle/input/whats-cooking-kernels-only/test.json')

#sample=pd.read_json('/kaggle/input/whats-cooking-kernels-only/sample_submission.csv')
Types_of_Foods = data['cuisine'].unique()

Number_of_Types_of_Foods=data['cuisine'].nunique()

print(Types_of_Foods)

print(Number_of_Types_of_Foods)
doc=data['ingredients']

doc=doc.apply(lambda x: str(x).upper())

doc=doc.apply(lambda x: str(x).replace('[^a-z]',' '))

doc=doc.apply(lambda x: str(x).replace(',',' '))

doc=doc.apply(lambda x: str(x).replace("'",' '))

doc=doc.apply(lambda x:str(x).replace('[0-9]',' '))

stopwords=nltk.corpus.stopwords.words('english')

stemmer=nltk.stem.PorterStemmer()

doc=doc.apply(clean_sentance)
x_train,x_test,y_train,y_test=train_test_split(doc,data['cuisine'],test_size=0.3,random_state=1)
countvec.fit(x_train)
x_train=countvec.transform(x_train)

x_test=countvec.transform(x_test)
pd.DataFrame(x_train.toarray(),columns=countvec.get_feature_names())
accuracy=[]

for i in list_of_models:

    i.fit(x_train,y_train)

    y_pred=i.predict(x_test)

    accuracy_01=accuracy_score(y_test,y_pred)

    accuracy.append(accuracy_01)

    print(i)

    print(accuracy)

    confusion=pd.DataFrame(confusion_matrix(y_test,y_pred),columns=Types_of_Foods,index=Types_of_Foods)

    plt.figure(figsize=(15,10))

    sns.heatmap(confusion,annot=True,fmt='d')

    plt.show()

    
df_accuracy=pd.DataFrame(['Logistic Regression','KNN','Decision Tree','Random Forest','Binomial'],columns=['Model Names'])

df_accuracy['Accuracy Score']=accuracy

df_accuracy
plt.figure(figsize=(15,6))

sns.barplot(df_accuracy['Model Names'],df_accuracy['Accuracy Score'],)