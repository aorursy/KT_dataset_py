import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/ufcdata/preprocessed_data.csv')

raw_data = pd.read_csv('/kaggle/input/ufcdata/data.csv')

data.shape
raw_data.head(3)
data.head()
data.info()
fighters = pd.concat([raw_data['R_fighter'], raw_data['B_fighter']], ignore_index=True)

names = ' '

for name in fighters:

    name = str(name)

    names = names + name + ' '
from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black',  

                min_font_size = 10).generate(names) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
raw_data['weight_class'].unique()
plt.figure(figsize=(8,8))

plt.xticks(rotation=90)

sns.set()

sns.set(style="darkgrid")

ax = sns.countplot(x=raw_data['weight_class'], data=raw_data)
stances = pd.concat([raw_data['R_Stance'], raw_data['B_Stance']], ignore_index=True)

st_values = stances.value_counts().sort_values(ascending=False).head(10)

st_labels = st_values.index
plt.figure(figsize=(10,6))

sns.barplot(y=st_values, x=st_labels)
plt.figure(figsize=(8,8))

sns.set()

sns.set(style="darkgrid")

ax = sns.countplot(x=raw_data['no_of_rounds'], data=raw_data)
countsT = data['title_bout'].value_counts()

labels = 'False' ,'True'

sizes = countsT.values

explode = (0.1, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
countsT = data['Winner'].value_counts()

labels = 'Red' ,'Blue'

sizes = countsT.values

explode = (0.1, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
raw_data['country'] = raw_data['location'].apply(lambda x : x.split(',')[-1])
plt.figure(figsize=(8,8))

plt.xticks(rotation=90)

sns.set()

sns.set(style="darkgrid")

ax = sns.countplot(x=raw_data['country'], data=raw_data)
values = fighters.value_counts().sort_values(ascending=False).head(10)

labels = values.index
plt.figure(figsize=(15,8))

sns.barplot(x=values, y=labels)
raw_data['year'] = raw_data['date'].apply(lambda x : x.split('-')[0])
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

sns.set()

sns.set(style="darkgrid")

ax = sns.countplot(x=raw_data['year'], data=raw_data)
age = pd.concat([raw_data['R_age'], raw_data['B_age']], ignore_index=True)

age_values = age.value_counts()

age_labels = age_values.index

plt.figure(figsize=(15,8))

sns.barplot(x=age_labels,y=age_values)
data_num = data.select_dtypes(include=[np.float, np.int])



scaler = StandardScaler()

data[list(data_num.columns)] = scaler.fit_transform(data[list(data_num.columns)])
y = data['Winner']

X = data.drop(columns = 'Winner')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
%%time

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate predictions

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))