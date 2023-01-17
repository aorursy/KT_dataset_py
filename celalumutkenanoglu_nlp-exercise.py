# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
df.head()
df.info()
df.isnull().sum()
text_df = df[["title", "company_profile", "description", "requirements", "benefits"]]

text_df = text_df.fillna(' ')



text_df.head()
cat_df = df[["telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education", "industry", "function","fraudulent"]]

cat_df = cat_df.fillna("None")

cat_df["telecommuting"] = [0 if i=="None" else i for i in cat_df["telecommuting"]]

cat_df["has_questions"] = [0 if i=="None" else i for i in cat_df["has_questions"]]

cat_df["has_company_logo"] = [0 if i=="None" else i for i in cat_df["has_company_logo"]]



cat_df.head(10)
Counter(cat_df["employment_type"])

cat_df = pd.get_dummies(cat_df, columns=["employment_type"])
Counter(cat_df["required_experience"])
cat_df = pd.get_dummies(cat_df, columns=["required_experience"])
Counter(cat_df["industry"])
cat_df.drop(labels=["industry"],axis=1,inplace=True)
Counter(cat_df["function"])
cat_df.drop(labels=["function"],axis=1,inplace=True)
cat_df = pd.get_dummies(cat_df, columns=["required_education"])
cat_df.head()
import re

import nltk as nlp

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

def clean_tex(data,max_features):

    description_list = []

    for description in data:

        description = re.sub("[^a-zA-Z]"," ",description)

        description = description.lower()

        description = nlp.word_tokenize(description)

        description = [word for word in description if not word in set(stopwords.words("english"))]

        lemma = nlp.WordNetLemmatizer()

        description = [lemma.lemmatize(word) for word in description ]

        description =" ".join(description)

        description_list.append(description)

    count_vectorizer = CountVectorizer(max_features=max_features)

    sparce_matrix=count_vectorizer.fit_transform(description_list).toarray()

    return sparce_matrix
text_df.head()
text_matrix = clean_tex(text_df.title,50)

text_matrix.shape
company_matrix = clean_tex(text_df.company_profile,50)

print(company_matrix.shape)

description_matrix = clean_tex(text_df.description,200)

print(description_matrix.shape)

requirements_matrix = clean_tex(text_df.requirements,100)

print(requirements_matrix.shape)
benefits_matrix = clean_tex(text_df.benefits,20)

print(benefits_matrix.shape)
cat_x = cat_df.iloc[:,:-1].values

print(cat_x.shape)
x = np.concatenate((cat_x,text_matrix,company_matrix,description_matrix,requirements_matrix,benefits_matrix),axis=1)

print(x.shape)
Counter(cat_df["fraudulent"])
y = cat_df.iloc[:,4].values

y.shape
Counter(y)
from sklearn.model_selection import train_test_split

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



nb.fit(x_train,y_train)



#%%

y_pred = nb.predict(x_test)

print("score",nb.score(x_test,y_test))

from sklearn.neighbors import KNeighborsClassifier



nb = KNeighborsClassifier(n_neighbors=4)



nb.fit(x_train,y_train)



print("score",nb.score(x_test,y_test))