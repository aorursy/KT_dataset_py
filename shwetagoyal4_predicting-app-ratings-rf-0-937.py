import re

import sys



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import time

import datetime



import warnings

warnings.filterwarnings("ignore")



from sklearn import preprocessing



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso



from sklearn import metrics

from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error,r2_score

data = pd.read_csv("../input/googleplaystore.csv")
# Print first few rows from data



data.head()
# Print last few rows from data



data.tail()
# Shape of data 



data.shape
# Basic information of data



data.info()
# Checking null values



data.isnull().sum()
# Types of data



data.dtypes
# Describing our data



data.describe()
# Columns present in our data



data.columns
# The best way to fill missing values might be using the median instead of mean



data['Rating'] = data['Rating'].fillna(data['Rating'].median())
# Lets convert all the versions in the format number.number to simplify the data

# We have to clean all non numerical values & unicode charachters 



replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]

for i in replaces:

    data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))



regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

for j in regex:

    data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))



data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)

data['Current Ver'] = data['Current Ver'].fillna(data['Current Ver'].median())

# Count the number of unique values in category column 



data['Category'].unique()
# Check the record  of unreasonable value which is 1.9

i = data[data['Category'] == '1.9'].index

data.loc[i]
# Drop this bad column

data = data.drop(i)
# Removing NaN values

data = data[pd.notnull(data['Last Updated'])]

data = data[pd.notnull(data['Content Rating'])]
plt.figure(figsize=(20, 10))

sns.countplot(x='Category', data=data, palette='Set1')

plt.xticks(rotation=90)

plt.xlabel('Categories', fontsize=20)

plt.ylabel('Counts', fontsize=20)

plt.title('Count of application according to category', fontsize=15, color='r')
from pylab import rcParams



rcParams['figure.figsize'] = 11.7, 8.27

g = sns.kdeplot(data.Rating, color="Red", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of Rating',size = 20)
labels =data['Type'].value_counts(sort = True).index

sizes = data['Type'].value_counts(sort = True)





colors = ["lightblue","orangered"]

explode = (0.1,0)  # explode 1st slice

 

rcParams['figure.figsize'] = 5,5

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=150,)



plt.title('Percent of Free App in data', size = 15)

plt.show()
from wordcloud import WordCloud

wordcloud1 = WordCloud(max_font_size=350, collocations=False, max_words=33, width=1600, height=800, background_color="white").generate(' '.join(data['Category']))

plt.figure(figsize=(12,8))

plt.imshow(wordcloud1, interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize=(10,8))

ax = sns.countplot(y='Content Rating', data=data)
# App values encoding



LE = preprocessing.LabelEncoder()

data['App'] = LE.fit_transform(data['App'])
# Category features encoding



CategoryList = data['Category'].unique().tolist() 

CategoryList = ['cat_' + word for word in CategoryList]

data = pd.concat([data, pd.get_dummies(data['Category'], prefix='cat')], axis=1)
# Genres features encoding



LE = preprocessing.LabelEncoder()

data['Genres'] = LE.fit_transform(data['Genres'])
# Content Rating features encoding



LE = preprocessing.LabelEncoder()

data['Content Rating'] = LE.fit_transform(data['Content Rating'])

# Type encoding



data['Type'] = pd.get_dummies(data['Type'])
# Last Updated encoding



data['Last Updated'] = data['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))

# Price cleaning



data['Price'] = data['Price'].apply(lambda x : x.strip('$'))
# Installs cleaning



data['Installs'] = data['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
# Convert kbytes to Mbytes



k_indices = data['Size'].loc[data['Size'].str.contains('k')].index.tolist()



converter = pd.DataFrame(data.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))



data.loc[k_indices,'Size'] = converter
# Size cleaning



data['Size'] = data['Size'].apply(lambda x: x.strip('M'))

data[data['Size'] == 'Varies with device'] = 0

data['Size'] = data['Size'].astype(float)
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']

features.extend(CategoryList)



X = data[features]

y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)
from sklearn import tree
clf = tree.DecisionTreeRegressor(criterion='mae', max_depth=5, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

accuracy
model = RandomForestRegressor(n_estimators = 200, n_jobs=-1, random_state=10)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

acc
Pred = model.predict(X_test)

'Mean Absolute Error:', metrics.mean_absolute_error(y_test, Pred)
'Mean Squared Error:', metrics.mean_squared_error(y_test, Pred)
'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Pred))
X = data.loc[:,['Reviews', 'Size', 'Installs', 'Content Rating']]



x_train, x_cv, y_train, y_cv = train_test_split(X, data.Rating)



clf = tree.DecisionTreeRegressor(criterion='mae', max_depth=5, min_samples_leaf=5, random_state=42)
clf.fit(x_train, y_train)
ac = clf.score(x_cv, y_cv)

ac