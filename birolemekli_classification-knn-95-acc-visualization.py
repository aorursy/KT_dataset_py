import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence,text
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()
df.describe()
df.info()
df.isna().sum()
labels=df.location.value_counts().index[:10]
values=df.location.value_counts().values[:10]
plt.figure(figsize = (15, 8))

ax = sns.barplot(x=labels, y=values)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)


for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.9, values[i],ha="center")
labels=df['department'].value_counts().index[:10]
values=df['department'].value_counts().values[:10]
irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                 'rgb(175, 49, 35)', 'rgb(36, 73, 147)']

fig = make_subplots(1, specs=[[{'type':'domain'}]],subplot_titles=['Department'])
fig.add_trace(go.Pie(labels=labels, values=values, pull=[0.1,0,0,0,0,0,0,0,0], hole=.15,name="Department Top 10",marker_colors=irises_colors), 1, 1)
fig.update_layout(title_text='Department Top 10')
fig.show()
df.fillna('',inplace=True)
df['text'] = df['title'].str.cat(df[['location', 'department','salary_range','company_profile','description','requirements',
                                    'benefits','employment_type','required_education','industry',
                                    'function']].astype(str), sep=' ')
df.drop(columns=['title','job_id','has_questions','required_experience','location', 'department','salary_range','company_profile','description','requirements',
                                    'benefits','telecommuting','has_company_logo','employment_type','required_education','industry',
                                    'function'],inplace=True)
df.text[0]
df.text = df.text.apply(lambda x: x.lower())
df['text']=df.text.str.replace(r'\W',' ',regex=True)
df['text']=df.text.str.replace(r'\b\d+','',regex=True)
df['text']=df.text.str.replace(r'\S{20,}',' ',regex=True)
df['text']=df.text.str.replace(r'\s{2,}',' ',regex=True)
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
wordcloud = WordCloud(
    width = 1600,
    height = 768,
    max_words=2500,
    background_color = 'black').generate(str(df.text.values))
fig = plt.figure(
    figsize = (15, 15),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.show()
sns.countplot(x="fraudulent", data=df)
df[df.text.duplicated()].count()
df = df.drop_duplicates()
print(len(df))
df.reset_index(drop=True, inplace=True)
df['text'].apply(lambda x: len(str(x).split())).mean()
num_max = 1000000
max_len = 400
## The process of enumerating words
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(df.text)

cnn_texts_seq = tok.texts_to_sequences(df.text)
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len,padding='post')
cnn_texts_mat
X=MinMaxScaler().fit_transform(cnn_texts_mat)
X
X_train,X_test,y_train,y_test=train_test_split(X,df.fraudulent,test_size=0.15, random_state=122)
(X_train.shape,X_test.shape)
knn=KNeighborsClassifier(n_neighbors=3,weights='uniform')
k_neightbors = list(range(1,9,2))
scores_ = []

for item in k_neightbors:
    knn = KNeighborsClassifier(n_neighbors=item)
    scores = cross_val_score(knn, X_train, y_train, scoring='accuracy')
    scores_.append(scores.mean())
scores.max()
MSE = [1 - x for x in scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_neightbors, MSE[:len(k_neightbors)])

plt.show()