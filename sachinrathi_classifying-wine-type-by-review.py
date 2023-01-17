import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import train_test_split

%matplotlib inline
data = pd.read_csv('../input/winemag-data_first150k.csv')

data.head(5)
data[data.duplicated('description',keep=False)].sort_values('description').head(5)
data = data.drop_duplicates('description')

data = data[pd.notnull(data.price)]

data.shape
from scipy.stats import pearsonr

import statsmodels.api as sm

print("Pearson Correlation:", pearsonr(data.price, data.points))

print(sm.OLS(data.points, data.price).fit().summary())

sns.lmplot(y = 'price', x='points', data=data)

fig, ax = plt.subplots(figsize = (20,7))

chart = sns.boxplot(x='country',y='points', data=data, ax = ax)

plt.xticks(rotation = 90)

plt.show()
data.country.value_counts()[:17]
country=data.groupby('country').filter(lambda x: len(x) >100)

df2 = pd.DataFrame({col:vals['points'] for col,vals in country.groupby('country')})

meds = df2.median()

meds.sort_values(ascending=False, inplace=True)



fig, ax = plt.subplots(figsize = (20,7))

chart = sns.boxplot(x='country',y='points', data=country, order=meds.index, ax = ax)

plt.xticks(rotation = 90)



plt.show()
df3 = pd.DataFrame({col:vals['price'] for col,vals in country.groupby('country')})

meds2 = df3.median()

meds2.sort_values(ascending=False, inplace=True)



fig, ax = plt.subplots(figsize = (20,5))

chart = sns.barplot(x='country',y='price', data=country, order=meds2.index, ax = ax)

plt.xticks(rotation = 90)

plt.show()
# medians for the above barplot

print(meds2)
data = data.groupby('variety').filter(lambda x: len(x) >100)

list = data.variety.value_counts().index.tolist()

fig4, ax4 = plt.subplots(figsize = (20,7))

sns.countplot(x='variety', data=data, order = list, ax=ax4)

plt.xticks(rotation = 90)

plt.show()
data = data.groupby('variety').filter(lambda x: len(x) >200)



df4 = pd.DataFrame({col:vals['points'] for col,vals in data.groupby('variety')})

meds3 = df4.median()

meds3.sort_values(ascending=False, inplace=True)



fig3, ax3 = plt.subplots(figsize = (20,7))

chart = sns.boxplot(x='variety',y='points', data=data, order=meds3.index, ax = ax3)

plt.xticks(rotation = 90)

plt.show()
df5 = pd.DataFrame({col:vals['points'] for col,vals in data.groupby('variety')})

mean1 = df5.mean()

mean1.sort_values(ascending=False, inplace=True)



fig3, ax3 = plt.subplots(figsize = (20,7))

chart = sns.barplot(x='variety',y='points', data=data, order=mean1.index, ax = ax3)

plt.xticks(rotation = 90)

plt.show()
df6 = pd.DataFrame({col:vals['price'] for col,vals in data.groupby('variety')})

mean2 = df6.mean()

mean2.sort_values(ascending=False, inplace=True)



fig3, ax3 = plt.subplots(figsize = (20,7))

chart = sns.barplot(x='variety',y='price', data=data, order=mean2.index, ax = ax3)

plt.xticks(rotation = 90)

plt.show()
X = data.drop(['Unnamed: 0','country','designation','points','province','region_1','region_2','variety','winery'], axis = 1)

y = data.variety



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

wine =data.variety.unique().tolist()

wine.sort()

wine[:10]
output = set()

for x in data.variety:

    x = x.lower()

    x = x.split()

    for y in x:

        output.add(y)



variety_list =sorted(output)

variety_list[:10]

extras = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 'cab',"%"]

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.update(variety_list)

stop.update(extras)
from scipy.sparse import hstack



vect = CountVectorizer(stop_words = stop)

X_train_dtm = vect.fit_transform(X_train.description)

price = X_train.price.values[:,None]

X_train_dtm = hstack((X_train_dtm, price))

X_train_dtm
X_test_dtm = vect.transform(X_test.description)

price_test = X_test.price.values[:,None]

X_test_dtm = hstack((X_test_dtm, price_test))

X_test_dtm
from sklearn.linear_model import LogisticRegression

models = {}

for z in wine:

    model = LogisticRegression()

    y = y_train == z

    model.fit(X_train_dtm, y)

    models[z] = model



testing_probs = pd.DataFrame(columns = wine)
for variety in wine:

    testing_probs[variety] = models[variety].predict_proba(X_test_dtm)[:,1]

    

predicted_wine = testing_probs.idxmax(axis=1)



comparison = pd.DataFrame({'actual':y_test.values, 'predicted':predicted_wine.values})   



from sklearn.metrics import accuracy_score

print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")

comparison.head(5)