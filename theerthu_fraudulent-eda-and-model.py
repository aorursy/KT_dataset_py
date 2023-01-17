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
df = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()
df.isnull().sum()
round(100*(df.isnull().sum()/len(df.index)),2)
### Lets drop 'salary_range' 
df = df.drop('salary_range',axis=1)
df['department'].value_counts()
### nullifying NA values of Department with 'other'

df['department'] = df['department'].fillna(value='other')
import seaborn as sns

import matplotlib.pyplot as plt

df.columns
plt.figure(figsize=(20,8))

sns.countplot(hue='fraudulent',y='required_education',data=df)

plt.show()
df['required_education'].value_counts()
df['required_education'] = df['required_education'].fillna(value='other')
## Counting plot

plt.figure(figsize=(20,8))

sns.countplot(hue='fraudulent',y='required_education',data=df)

plt.show()
### education with 'high school or equivalent' has more fradulent cases

## Counting plot

plt.figure(figsize=(20,8))

sns.countplot(hue='fraudulent',y='required_experience',data=df)

plt.show()
df['has_company_logo'].value_counts()
## Counting plot

plt.figure(figsize=(20,8))

sns.countplot(hue='fraudulent',y='has_company_logo',data=df)

plt.show()
## copying all fradulent cases
df_f = df.loc[df['fraudulent']==0]
df_f.isnull().sum()
## drop na values that are not required

df_f = df_f.dropna()
## Getting most words said in Benefits for fraudulent cases

from wordcloud import WordCloud, STOPWORDS

stopwords = STOPWORDS

wordcloud = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join

                                                                                           (df_f['benefits']))
plt.figure(figsize=(20,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Most words said in benefits", fontsize=25)

plt.show()

## Getting most words said in Description for fraudulent cases

from wordcloud import WordCloud, STOPWORDS

stopwords = STOPWORDS

wordcloud = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join

                                                                                           (df_f['description']))
plt.figure(figsize=(20,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Most words said in benefits", fontsize=25)

plt.show()
## Dropping all Na values and unwanted column JOB_ID

df = df.dropna()

df = df.drop('job_id',axis=1)
## making categorical variable to numbers

cat_var = ['title','location','department','company_profile','employment_type','required_experience','required_education','industry',

          'function']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df[cat_var] = df[cat_var].apply(lambda x: le.fit_transform(x))
### Making it safe

df_copy = df.copy()
## Dropping Text columns which can't be converted 

df = df.drop(['company_profile','description','benefits','requirements'],axis=1)
from sklearn.model_selection import train_test_split

# Putting feature variable to X

X = df.drop('fraudulent',axis=1)



# Putting response variable to y

y = df['fraudulent']



# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
## import metrics

# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
## Accuracy is good