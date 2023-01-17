import pandas as pd
data = pd.read_csv('../input/sel_hn_stories.csv', header = None, parse_dates=[0])
data.head()
columns = ["submission_time", "upvotes", "url", "headline"]
data.columns = columns
data.head()
data.shape
data.isnull().sum()
data.info()
data.dropna(inplace=True)
data.shape
token_data = []
#let us split the strings on the space 



for words in data['headline']:

  token_data.append(words.split(" "))

token_data[0:5]
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]

clean_strings = []
for words in token_data:

  cleaned = [] 

  for word in words:

    word = word.lower()

    for punc in punctuation:

      word = word.replace(punc,"")

    cleaned.append(word)

  clean_strings.append(cleaned)
clean_strings[0:5]
from collections import Counter

import numpy as np



unique_words = [] 

single_words = []



#let us combine all the words in the lists into one single list



final_list = []



for words in clean_strings: 

  for word in words:

    final_list.append(word)
final_list[0:15]
word_counter = Counter(final_list)

print(word_counter['the'])

print(word_counter['google'])
for count in word_counter: 

  if word_counter[count] > 1: 

    unique_words.append(count)

  else:

    single_words.append(count)
unique_words[0:5]
single_words[0:5]
#creating the data frame 

#for now we will fill the df with 0

#be mindful when indexing. The index should be cleaned list of strings and not the individual words



counts = pd.DataFrame(data = 0, index = np.arange(len(clean_strings)), columns = unique_words)
counts.head()
#we will use the enumerate function. Remember each list in the clean_strings corresponds to the row. 



for index, words in enumerate(clean_strings): 

  for word in words:

    if word in unique_words:

      counts.iloc[index][word] += 1
counts.head(10)
counts.shape
column_filtering = counts.sum(axis = 0)
column_filtering.head()
columns_to_drop = column_filtering[(column_filtering < 5) | (column_filtering > 100)]
columns_to_drop.head()
counts.drop(columns = columns_to_drop.index, inplace = True)
counts.shape
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error





X_train, X_test, y_train, y_test = train_test_split(counts, data['upvotes'], test_size = 0.2, random_state = 1)
X_train.head()
y_train.head()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)

print(lr_mse)

lr_rmse = __import__('math').sqrt(lr_mse)

print(lr_rmse)
data['upvotes'].mean()
data['upvotes'].std()
from sklearn.model_selection import cross_val_score

lr2 = LinearRegression()

score = cross_val_score(lr2, counts, data['upvotes'], scoring = 'neg_mean_squared_error', cv = 10)

lr2_score = __import__('numpy').mean(score)

print(abs(lr2_score))