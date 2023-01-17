# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np 

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import kagglegym

train_df = pd.read_json("../input/train.json")

test_df = pd.read_json("../input/test.json")

train_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

int_level = train_df['interest_level'].value_counts()



plt.figure(figsize=(6,3))

sns.barplot(int_level.index, int_level.values, alpha=0.6, color=color[2])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()
plt.figure(figsize=(10,4))

bathroom = train_df['bathrooms'].value_counts()



sns.barplot(bathroom.index,bathroom.values,alpha=.8,color=color[1])

plt.ylabel('Number of Occurances',fontsize=12)

plt.xlabel('Number of Bathrooms',fontsize=12)

plt.show()
plt.figure(figsize=(8,4))

sns.countplot(x='bathrooms', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurances',fontsize=12)

plt.xlabel('Number of Bathrooms',fontsize=12)

plt.show()
plt.figure(figsize=(6,4))



sns.countplot(x='bedrooms',data=train_df)

plt.ylabel('Number of Occurances',fontsize=12)

plt.xlabel('Number of Bedrooms',fontsize=12)

plt.show()
plt.figure(figsize=(6,4))



sns.countplot(x='bedrooms',hue='interest_level',data=train_df)

plt.ylabel('Number of Occurances',fontsize=12)

plt.xlabel('Number of Bedrooms',fontsize=12)

plt.show()
plt.figure(figsize=(8,4))



plt.scatter(range(train_df.shape[0]), np.sort(train_df.price.values))

plt.ylabel('Price',fontsize=12)

plt.xlabel('Individual Listing')

plt.title('Distribution of House prices')

plt.show()
plt.figure(figsize=(8,4))



sns.distplot(np.log(train_df.price.values))

plt.ylabel('Price',fontsize=12)

plt.xlabel('Individual Listing')

plt.title('Distribution of House prices')

plt.show()
#transforms price to be logarithm

train_df.price = train_df.price.apply(lambda x: np.log(x))
#get number of features for each listing

train_df['num_features']  = train_df.features.apply(lambda x: len(x))

#plot distribution

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14,4))



sns.countplot(x='num_features',data=train_df,ax=ax1)

ax1.set_title('Distribution of number of features')



sns.countplot(x='num_features',hue='interest_level',data=train_df,ax=ax2)

ax2.set_title('Distribution of number of features by interest')

plt.show()
import itertools

from collections import Counter



#puts all features into one list and gets word count

featureCount =Counter(list(itertools.chain.from_iterable(train_df.features)))

featureCount = dict((k.replace(" ", "_"), v) for k, v in featureCount.items() if v >= 5)

sortedFeatures = sorted(featureCount, key=featureCount.get,reverse=True)

print('The top ten features are ' + str(sortedFeatures[:10]))



#transform feature column to be put in vectorizer

train_df.features = [','.join(x) for x in train_df.features]

train_df.features = train_df.features.apply(lambda x: x.replace(' ','_'))

train_df.features = train_df.features.apply(lambda x: x.replace(',',' '))



#transform features into a sparse matrix

from sklearn.feature_extraction.text import CountVectorizer

#fit count vectorizer

vectorizer = CountVectorizer().fit(sortedFeatures)#fit vectorizer with top features

featVec = vectorizer.fit_transform(train_df.features)#transform training data into sparse matrix