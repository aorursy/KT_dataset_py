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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
%matplotlib inline
zf_train = zipfile.ZipFile('/kaggle/input/whats-cooking/train.json.zip')
df_train = pd.read_json(zf_train.open('train.json'))
print(df_train.head())
df_train = df_train.drop('id', axis = 1)
print(df_train.head())
print(df_train.shape)
cuisines = df_train.cuisine.unique()
print(cuisines.shape)
fig = plt.figure()
fig.set_size_inches(20, 6)
sns.barplot(data = df_train.groupby('cuisine').count(), x = df_train.groupby('cuisine').count().index, y = 'ingredients')
cuisine_ingredients = {}
for i, cuisine in enumerate(cuisines):
    df = df_train[df_train['cuisine'] == cuisine]
    ingredients = []
    df['ingredients'].apply(lambda x : ingredients.extend(x))
    ingredients = np.array(ingredients)
    unique_ingredients = np.unique(list(ingredients))
    cuisine_ingredients[cuisine] = unique_ingredients
print(cuisine_ingredients)
total_ingredients = []
for cuisine, ingredients in cuisine_ingredients.items():
    total_ingredients.extend(ingredients)
print(len(total_ingredients))
total_different_ingredients = list(np.unique(np.array(total_ingredients)))
print("Total different types of ingredients :", len(total_different_ingredients))
def make_columns(dataframe):
    for ingredient in total_different_ingredients:
        dataframe[ingredient] = 0
    return dataframe

df_train = make_columns(df_train)
print(df_train.shape)
df_train.head()
def fill_columns(dataframe):
    for index, row in dataframe.iterrows():
        ingredients = row['ingredients']
        print(index)
        for ingredient in ingredients:
            if ingredient in total_different_ingredients:
                dataframe.at[index, ingredient] = 1
    return dataframe

df_train = fill_columns(df_train)
df_grouped = df_train.groupby('cuisine').mean()
df_grouped
keep_percent = .2
top_cuisine_ingredients = {}
for cuisine in cuisines:
    a = df_grouped.loc[cuisine, :].sort_values() < keep_percent
    a = a[a == True]
    top_cuisine_ingredients[cuisine] = a.index

top_ingredients = []
for cuisine, ingredients in top_cuisine_ingredients.items():
    top_ingredients.extend(ingredients)
    
top_ingredients = np.array(top_ingredients)
unique_top_ingredients = ['cuisine']
unique_top_ingredients.extend(list(np.unique(top_ingredients)))
print(len(unique_top_ingredients))
df_train = df_train[unique_top_ingredients]
print(df_train.shape)
temp = df_train.groupby('cuisine').mean()
fig, ax = plt.subplots(10, 2)
ax = ax.flatten()
fig.set_size_inches(20, 50)
for index, cuisine in enumerate(cuisines):
    ax[index].title.set_text(cuisine)
    sns.barplot(x = top_cuisine_ingredients[cuisine][:-1][-5:], y = temp.loc[cuisine][top_cuisine_ingredients[cuisine]].values[:-1][-5:], ax = ax[index])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
Y = df_train.cuisine
X = df_train.drop(['cuisine'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.15)
rf = RandomForestClassifier(n_estimators = 100, max_depth = height).fit(X_train, y_train)
test_predictions = rf.predict(X_test)
train_predictions = rf.predict(X_train)
print("height is ", height)
print("Training Accuracy is : ", sum(train_predictions==y_train)/y_train.shape[0])
print("Testing Accuracy is : ", sum(test_predictions==y_test)/y_test.shape[0])
zf_test = zipfile.ZipFile('/kaggle/input/whats-cooking/test.json.zip')
df_test = pd.read_json(zf_test.open('test.json'))
df_test.head()
df_test = make_columns(df_test)
df_test.head()
df_test = fill_columns(df_test)
df_test.head()
id = df_test['id']
df_test = df_test[unique_top_ingredients[1:]]
df_test.head()
predictions = rf.predict(df_test)
df = pd.DataFrame({'id':id, 'cuisine':predictions})
df.to_csv('cooking_submission.csv', index = False)
zf_train = zipfile.ZipFile('/kaggle/input/whats-cooking/train.json.zip')
df_train = pd.read_json(zf_train.open('train.json'))
df_train.head()
def joiningre(x):
    return ' '.join(x)
df_train['all_ingredients'] = df_train['ingredients'].apply(joiningre)
df_train.head(2)
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(df_train['all_ingredients'])
len(count_vect.vocabulary_)
df_train_transform = count_vect.transform(df_train['all_ingredients'])
X = df_train_transform.toarray()
Y = df_train['cuisine']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.15)
rf = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
test_predictions = rf.predict(X_test)
train_predictions = rf.predict(X_train)
print("height is ", height)
print("Training Accuracy is : ", sum(train_predictions==y_train)/y_train.shape[0])
print("Testing Accuracy is : ", sum(test_predictions==y_test)/y_test.shape[0])
print(train_predictions)
zf_test = zipfile.ZipFile('/kaggle/input/whats-cooking/test.json.zip')
df_test = pd.read_json(zf_test.open('test.json'))
df_test['all_ingredients'] = df_test['ingredients'].apply(joiningre)
df_test.head()
df_test_transform = count_vect.transform(df_test['all_ingredients'])
predictions = rf.predict(df_test_transform.toarray())
print(predictions)
df = pd.DataFrame({'id':df_test.id, 'cuisine':predictions})
df.to_csv('cooking_submission.csv', index = False)