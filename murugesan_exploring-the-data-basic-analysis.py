import os

import pandas as pd

import numpy as np

import random

from matplotlib import pyplot as plt

import plotly.plotly as py

from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import seaborn as sns



%matplotlib inline
train_text = pd.read_csv("../input/en_train.csv")
##Checking Null values

print("Null values\n")

print(train_text.isnull().sum(axis = 0))

print("Total non-null values per column\n")

print(train_text.count())
train_text[train_text['before'].isnull()].head(10)
#Getting the location of tokens with NaN values. Mostly they appear at the beginning of the sentence

plt.hist(train_text[train_text['before'].isnull()]['token_id'])
#Analysing the dataframe

print(train_text.dtypes)
print(train_text.head())

print(train_text.tail())
train_text['class'].value_counts()



#This shows the unique classes and the count of each class

#Clearly Plain token dominates everything else
train_text['change'] = train_text['before'] != train_text['after']

train_text['change'].value_counts()



#Less han 10% of the data get normalized
train_text[train_text['change'] == True]['class'].value_counts()



#Interestingly Plain is not the highest here, and all the Date and Cardinal fields are changing
random.seed(0)

train_sample = train_text
class_value = train_sample['class'].value_counts()

train_sample['class'].value_counts().plot(kind  = "bar")
np.log(train_sample['class'].value_counts()).plot(kind  = "bar")
agg_count = train_sample.loc[:,['class','change']].groupby(['class','change'])['class'].count()

agg_count = agg_count.unstack(1)

agg_count_log = np.log(agg_count).unstack(1)





agg_count['Total']= agg_count[True].fillna(0) + agg_count[False].fillna(0)



agg_count.sort_values([0],ascending = [0]).plot(kind = "bar", title = "Analysis in terms of absolute values")



np.log(agg_count).sort_values([0],ascending = [0]).plot(kind = "bar", title = "Analysis in log scale")
((agg_count/ agg_count['Total'].sum(axis = 0)) * 100).sort_values(['Total'], ascending = [0]).fillna(0)
plt.hist(train_sample['token_id'], log = True, bins = 100);

#Note: we have transformed the scale to log
token_hist = plt.hist(train_sample['token_id'][train_sample['token_id'] < 50], bins = 100);
percent = ((token_hist[0]/train_sample.shape[0]) * 100)



for i in range(5):

    percent_data = round(sum(percent[0:((i+1)*10)]),2)

    token_size = token_hist[1][(i+1)*10]

    

    print(percent_data, "% of data with tokens less than " ,round(token_size))
fig, ax = plt.subplots()

fig.set_size_inches(18,10)

sns.boxplot( y ='token_id', x = 'class', data = train_sample , linewidth = 2)
fig, ax = plt.subplots()

fig.set_size_inches(18,10)

sns.boxplot( y =np.log(train_sample['token_id']), x = 'class', data = train_sample , linewidth = 2)
max_token = pd.DataFrame(train_sample[['class','sentence_id','token_id']].

                         groupby(['sentence_id'])['token_id'].agg({'max_token':'max'}))
train_sample_token = pd.merge(left = train_sample, right = max_token, right_index = True, how = 'left', left_on = 'sentence_id')



train_sample_token['relative_position'] = (train_sample_token['token_id'])/(train_sample_token['max_token'])

fig, ax = plt.subplots()

fig.set_size_inches(18,10)

sns.boxplot( y ='relative_position', x = 'class', data = train_sample_token , linewidth = 2)
train_sample_change = train_sample[train_sample['change'] == True]
train_sample_change['class'].unique()
#Sample of date class

class_val = 'DATE' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head())

print(train_wc.tail())

#Note: All of the date format is changed in the output
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'LETTERS' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head())

print(train_wc.tail())

#Note: All of the letter format is changed in the output
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'CARDINAL' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head())

print(train_wc.tail())

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'PLAIN' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000, background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'VERBATIM' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)).strip())

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000, background_color = "white").generate(' '.join(train_wc['after'].astype(str)).strip())

plt.imshow(WC)
#Sample of date class

class_val = 'MEASURE' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000, background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'ORDINAL' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'DECIMAL' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)
#Sample of date class

class_val = 'MONEY' 



train_wc = train_sample_change[train_sample_change['class'] == class_val]

train_all_wc = train_sample[train_sample['class'] == class_val]



print(train_wc.head(10))

print(train_wc.tail(10))

#Note: There are numbers/roman numerals
fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC_all = WordCloud(width = 2000, height = 1000).generate(' '.join(train_all_wc['before'].astype(str)))

plt.imshow(WC_all)



fig, ax = plt.subplots()

fig.set_size_inches(15,10)

random.seed(0)

WC = WordCloud(width = 2000, height = 1000,background_color = "white").generate(' '.join(train_wc['after'].astype(str)))

plt.imshow(WC)