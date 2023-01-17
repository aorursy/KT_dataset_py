#Importing libraries

import pandas as pd

import numpy as np

import scipy as sci

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime
dataset2 = pd.read_csv('../input/TS_Harvey_Tweets.csv',engine='python')

dataset1 = pd.read_csv('../input/Hurricane_Harvey.csv',engine='python')
dataset1.shape
#Assigninng values to discriminate csv's

dataset1['csv_type'] = 0

dataset2['csv_type'] = 1

dataset = pd.concat([dataset1,dataset2],axis=0)
import missingno as msno

msno.bar(dataset,sort=True)
#Dropping NA's

dataset = dataset.dropna()

dataset = dataset.reset_index(drop=True)
dataset.shape
#Extracting Some features

dataset['Time'] = pd.to_datetime(dataset.Time)



dataset['year'] = dataset['Time'].dt.year

dataset['month'] = dataset['Time'].dt.month

dataset['day'] = dataset['Time'].dt.day

dataset['hour'] = dataset['Time'].dt.hour

dataset['date'] = dataset['Time'].dt.date
dataset.tail()

from nltk.corpus import stopwords

import re
def tweet_to_words(raw_tweet):

    raw_tweet = " ".join(word for word in raw_tweet.split() if 'http' not in word and not word.startswith('@') and not word.startswith('pic.twitter') and word != 'RT')

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 

#Obtaining clean tweets , the length of a tweet and the length of a clean tweet

dataset['clean_tweet'] = dataset['Tweet'].map(lambda x: tweet_to_words(x))

dataset['tweet_length'] = dataset['Tweet'].map(lambda x: len(x))

dataset['clean_tweet_length'] = dataset['clean_tweet'].map(lambda x: len(x))
#Finding the hashtags in a tweet

def hashtag(tweet):

    with_hashtag = " ".join([word for word in tweet.split() if word.startswith('#')])

    with_hashtag = with_hashtag.lower().split()

    return with_hashtag
dataset['hashtag'] = dataset['Tweet'].map(lambda x: hashtag(x))

# how many hashtags in a tweet

dataset['no_of_hashtag'] = dataset['hashtag'].map(lambda x: len(x))
print(np.sum(dataset['no_of_hashtag'])/len(dataset))

#On average , 35%  of tweets include hashtag
#WordCloud

from wordcloud import WordCloud, STOPWORDS

words = ' '.join(dataset['Tweet'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,

                      background_color='black',min_font_size=6,

                      width=3000,collocations=False,

                      height=2500

                     ).generate(cleaned_word)
plt.figure(1,figsize=(20, 20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

wordcloud_words = wordcloud.words_

s  = pd.Series(wordcloud_words,index=wordcloud_words.keys())

s.head(5)
negatives = ['wicked','bad','terrible','fuck','fucking','scary','damage','loss','shatter','shatters',

             'destroying','destroyed','devastating','disruptive','fucked','dead','shattering',

              'victim','alert','warning','evacuation','destructive','flooded','catastrophe','catastrophic',

             'worst','devastation','tornado','danger','dangerous','evacuated','serious',

             'screwed','frightening','flood','flooding','impact','effect','landfall','hitting',

             'influence','devastate','closed','terrifying','disaster','evacuation','evacuations',

             'risky','risk','problem','problems','problematic','need','destructive','disruptive']

#Calculating the degree of Negativity

def whether_negative(tweet):

    mylist= re.sub("[^a-zA-Z]", " ",tweet).lower().split()

    mylist_wo_neg = [word for word in mylist if word not in negatives]

    return len(mylist) - len(mylist_wo_neg)



dataset['degree_of_negativity'] = dataset['Tweet'].map(lambda x: whether_negative(x))

print(np.sum(dataset['degree_of_negativity'])/len(dataset))
#I found it logical to create a 'breaking column'

dataset['breaking'] = dataset['clean_tweet'].map(lambda x: 1 if 'breaking' in x else 0)

np.sum(dataset['breaking'])

##Picture, which tweets include a picture

dataset['picture'] = dataset['Tweet'].map(lambda x: 1 if 'pic.twitter.com' in x else 0)

np.sum(dataset['picture'])
#20 percent of tweets include link .
##Link , which tweets include a link

dataset['link'] = dataset['Tweet'].map(lambda x: 1 if 'http' in x else 0)



np.sum(dataset['link'])
#approximately ,70 percent of tweets contain link

## RT , which tweets contain RT

dataset['RT'] = dataset['Tweet'].map(lambda x: 1 if 'RT' in x else 0)

np.sum(dataset['RT'])

##Trump , which tweets about Trump

dataset['Trump'] = dataset['Tweet'].map(lambda x: 1 if 'Trump' in x else 1 if 'trump' in x else 1 if 'TRUMP' in x else 0 )

np.sum(dataset['Trump'])
# Approximately ,20k tweet are mentioning about Trump .

## Let's get into details of hashtag



dataset['all_hashtag'] = dataset['hashtag'].map(lambda x: " ".join(x))

#some hashtags include pic.twitter or https: etc. we should remove these from hashtag .

dataset['all_hashtag_cleaned'] = dataset['all_hashtag'].map(lambda x: x.replace('pic.twitter',' '))

dataset['all_hashtag_cleaned'] = dataset['all_hashtag_cleaned'].map(lambda x: x.replace('http',' '))

all_hashtag = " ".join(dataset['all_hashtag_cleaned'])
#all_hashtag include components , some of them start with # , wihch we are interested.

# the other things include link extension or picture extension , we will discard them

all_hashtag = all_hashtag.split()

#we are taking only hashtag which start with '#' onlt because the other ones are scraps .

all_hashtag_with_sign = [hashtag for hashtag in all_hashtag if hashtag[0] == '#']
# let's find the most occurring hashtags

import collections as co

c = co.Counter(all_hashtag_with_sign)

most_common_50=c.most_common(50)

most_common_50
##Periscope

# I think that '#video' and '#periscope' parameters should be merged .

# I will analyze the tweets which has a hashtag of '#video' or '#periscope'

dataset['periscope'] = dataset['Tweet'].map(lambda x: 1 if '#periscope' in x.lower() else 1 if '#video' in x.lower() else 0 )



np.sum(dataset['periscope'])
##News

# I will merge '#news' , '#foxnews' ,'#smartnews'

dataset['news'] = dataset['Tweet'].map(lambda x: 1 if '#news' in x.lower() else 1 if '#foxnews' in x.lower() else 1 if 'smartnews' in x.lower() else 0)



np.sum(dataset['news'])
##Evening

# I will divide hour of day into 2 categories : evening or not



evening_hours = [19,20,21,22,23]

dataset['evening'] = dataset['hour'].map(lambda x: 1 if x in evening_hours else 0)



np.sum(dataset['evening'])
## Last year

# In the recent years, twitter usage got popular , therefore . I extracted a new feature : current_year

#Therefore , I found it logical having a current_year parameter to predict whether a tweet is retweed or not .

dataset['current_year'] = dataset['year'].map(lambda x: 1 if x == 2017 else 0)
### Visualizations:



##Hour of Day

plt.figure(figsize=(12,10))

sns.countplot(x="hour",data=dataset,order=dataset.hour.value_counts().index)

plt.xticks(size=13)

plt.yticks(size=13)

plt.ylabel("The number of tweets sent",size=20)

plt.xlabel("Hour of Day",size=20)

plt.show()
# I think most of twitter users are nighthawks :))
### 2017 Harvey Hurricane



harvey_hurricane_2017 = dataset[(dataset.year == 2017) & ((dataset.month == 8) | (dataset.month == 9)) & (dataset.csv_type == 0) & (dataset.day > 20)]

tropical_storm_2017 = dataset[(dataset.year == 2017) & ((dataset.month == 8) | (dataset.month == 9)) & (dataset.csv_type == 1) & (dataset.day > 20)]



plt.plot(harvey_hurricane_2017.groupby('date').count()[['ID']], 'o-', label='harvey_hurricane_2017' , color = 'orange')

plt.plot(tropical_storm_2017.groupby('date').count()[['ID']], 'o-', label='tropical_storm_2017' , color = 'purple')

plt.title('Harvey Hurricane and Tropical Storm')

plt.legend(loc=0)

plt.ylabel('number of tweets')

plt.show()
#Deleting redundant annd unimportant input variables

dataset_cloned = dataset.copy()

del dataset['Unnamed: 0']

del dataset['ID']

del dataset['Time']

del dataset['Tweet']

del dataset['year']

del dataset['month']

del dataset['day']

del dataset['hour']

del dataset['clean_tweet']

del dataset['hashtag']

del dataset['all_hashtag']

del dataset['all_hashtag_cleaned']

del dataset['date']
X = dataset.iloc[:,1:17]

y = dataset.iloc[:,0]
corr = X.corr()

# Set up the matplot figure

f,ax = plt.subplots(figsize=(12,9))

#Draw the heatmap using seaborn

sns.heatmap(corr, cmap='inferno', annot=True)
#According to tableau , we can say that :

#Tweet_length is correlated with clean_tweet_length .

#Tweet_length is correlated with link .

#Likes and replies are correlated .



#I will drop tweet_length and Replies
del X['tweet_length'],X['Replies']
#Fitting LightGBM

#splitting the dataset into training and test dataset

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train ,y_test = train_test_split(X,y ,test_size=0.2 ,random_state = 30)
#Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
y_train = y_train.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

y_train = y_train.values

y_test = y_test.values
# Light GBM

import lightgbm as lgb

train_data=lgb.Dataset(X_train,label=y_train)

#setting parameters for lightgbm

param ={'num_leaves':90, 'objective':'regression_l2', 'num_leaves':10,'max_depth':5,'learning_rate':0.3,'max_bin':400,'boosting':'dart'}

param['metric'] = ['auc', 'l2']



num_round=50

lgbm=lgb.train(param,train_data,num_round)
#predicting on test set

ypred2=lgbm.predict(X_test)

ypred2 = pd.Series(ypred2).map(lambda x: round(x))

y_test = pd.Series(y_test)

ypred2[0:5]  # showing first 5 predictions

y_diff = ypred2 - y_test

y_diff = pd.DataFrame(y_diff)

y_diff = y_diff.rename(columns = {0:'diff'})

len(y_diff[y_diff['diff'] == 0]) / len(y_diff)

len(y_diff[y_diff['diff'] == 0]) / len(y_diff)

#77 percent of tweets are correctly predicted in terms of being retweed or not .
