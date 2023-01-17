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
'''Questions:

    -When were the most number of tweets tweeted? -> Finals

    -Did FIFA gain traction over the two(?) months it was held? (Date vs RTs) -> yes it did(finals had high rts)

    -What was the most common source of the tweet? ->[Twitter for Android]

    -Does the length of the tweet come from a normal distribution? -> Histogram is right-skewed (i.e mean > median)

    and if number of bins is taken to be 15, then it is bimodal + Boxplot stats (outliers upto 281 which can't be ignored)

    -Correlation Matrix - Which were the columns which were strongly correlated. -> len & Tweet_Len, Followers and Likes

    -Likes vs. RTs - Is there any connection/correlation between the two? -> No linear relationship

    -What were the 10 most popular Hashtags -> barchart with "WorldCup", "FRA" and 'worldcup' being the top 3'''

    

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from math import isnan

from scipy.stats import norm,ranksums

from statistics import mode



tweet = pd.read_csv("/kaggle/input/world-cup-2018-tweets/FIFA.csv")



t1 = tweet.sort_values(by='RTs',ascending = False) #sorting by RTs to determine popular tweets

def hashtag_cleaning():

    tweet["Hashtags"] = tweet.Hashtags.fillna('NOHASHTAG') #removing NaNs and replacing with NOHASHTAG

print("Before Cleaning Hashtags")

tweet.Hashtags.head(30)

print("After Cleaning Hashtags")

hashtag_cleaning()

#tweet.Hashtags.head(30)



#analysing source of tweets

def source():

    c_i = c_a = c_w = c_m = c_l = 0

    for i in tweet.Source: 

        if(i=='Twitter for iPhone'):#"Twitter for iPhone"): 207576 #"Twitter for Android"): 231895 #"Twitter Web Client"): 40442

            c_i+=1

        elif(i=='Twitter for Android'):

            c_a +=1

        elif i=='Twitter Web Client':

            c_w+=1

        elif i=='Twitter Lite':

            c_l+=1

        else:

            c_m +=1

        c = []

        c.append(c_i)

        c.append(c_a)

        c.append(c_w)

        c.append(c_m)

        c.append(c_l)

    #print(list(tweet.Source.unique()))

    plt.figure(figsize = (15,8))

    plt.title("Source of Tweet")

    plt.pie(c,labels = ['Twitter for iPhone','Twitter for Android','Twitter Web Client','Twitter Lite','Miscellaneous Sources'],autopct = '%.2f', colors = ['lightgreen','red','lightblue','purple','yellow'])

source()

#analysing dates vs number of tweets

def dates():

   # plt.style.use('ggplot')

    fin = 0

    sem_fin1 = 0

    sem_fin2 = 0

    other = 0

    for i in tweet.Date:

        if '2018-07-15' in i:

            fin+=1

        elif '2018-07-10' in i:

            sem_fin1+=1;

        elif '2018-07-11' in i:

            sem_fin2+=1;

        else:

            other+=1

    

    c = {'Finals': fin,"Semis 1":sem_fin1,"Semis 2": sem_fin2, "Other Matches": other}

    #print(c['Others'])

    print("Count of Tweets on particular dates :\n",c)

    plt.figure(figsize = (8,7))

    plt.bar(range(len(c)),list(c.values()))

    plt.ylim(0,270000)

    plt.xlabel("Dates")

    plt.ylabel("Count")

    plt.title("Dates vs. Number of Tweets")

    plt.xticks(range(len(c)),list(c.keys()))

    plt.show()

dates()
#created Tweet_Len column to store length of the preprocessed tweet



def cleaned_len():

    tw_len = []

    tw_len = [str(i) for i in list(tweet.Tweet)]

    tw_len1 = [len(i) for i in tw_len]

    tweet["Tweet_Len"] = tw_len1



cleaned_len()

#print("Original Tweet Length vs Cleaned Tweet Length\n")

#print(tweet[['len','Tweet_Len']][:30])
#date plot to see if fifa gained traction

t2 = t1.drop(columns = ['Place','Name']) #dropped Name and Place because of too much noise

t2.Date = pd.to_datetime(t2['Date'])

#DATES VS. RETWEETS

t3 = t2.sample(200000)

plt.figure(figsize=(13,7))

plt.title("Dates vs. Retweets: Did the tournament gain traction?")

plt.xlabel('Dates')

plt.ylabel('Retweets')

plt.plot_date(t3.Date,t3.RTs,color='purple')
#most popular hashtags



hash_split = []

for i in list(tweet.Hashtags.sample(100000)):

    if i!="NOHASHTAG":

        hash_split+=i.split(',') #splitting multiple hashtags

hash_arr = np.unique(np.array(hash_split)) #the unique hashtags



hash_dict = {}

for i in hash_split:

    if i not in hash_dict.keys() and i not in ['WORLDCUP','worldcup']: #counting number of hashtags

        hash_dict[i] = 1

    elif i in ['WORLDCUP','worldcup']:

        hash_dict['WorldCup']+=1

    else:

        hash_dict[i]+=1



def get_key(val): 

    for key, value in hash_dict.items(): 

         if val == value: 

             return key



hash_val = sorted(hash_dict.values(),reverse =True) #sorting in descending order of count

hash_key = [get_key(i) for i in hash_val]



hash_val = hash_val[:10] #top 10 most occurring

hash_key = hash_key[:10]



plt.figure(figsize = (12,7))

plt.title("Popularity of Hashtags")

plt.bar(range(len(hash_val)),hash_val, color='blue')

plt.ylim(0,90000)

plt.xticks(range(len(hash_key)),hash_key)

plt.ylabel("Occurrences")

plt.xlabel("Hashtag")

plt.show()
twt = tweet.sample(10000)

plt.scatter(twt.Likes,twt.Followers)