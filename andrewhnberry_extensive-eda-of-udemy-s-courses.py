# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the holy trinity of data science packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Other Visualization packages

import seaborn as sns 



#Other Packages

import re

from wordcloud import WordCloud, STOPWORDS 

import warnings

warnings.filterwarnings("ignore")
#importing udemy data to dataframe

df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head(3)
df.info()
df.isna().sum()
#Droping Course_ID and URL, since they're irrelevant right now for my EDA

df = df.drop(columns = ['course_id','url'], axis = 1)
#Finding course with highest number of subscribers

df.loc[df.num_subscribers.idxmax()]
#Finding course with highest number of reviews

df.loc[df.num_reviews.idxmax()]
df.groupby(['is_paid']).mean()
df.is_paid.value_counts()
#Standardize True

df['is_paid'] = df['is_paid'].replace('TRUE', 'True')

#Standardize False

df['is_paid'] = df['is_paid'].replace('FALSE', 'False')



#Believe it our not but learning guitar to worship is a paid course 

#and currenlty 75% off.

df['is_paid'] = df['is_paid'].replace('https://www.udemy.com/learnguitartoworship/',

                                      'True')
df.is_paid.value_counts()
#Now I can run this code in peace 

df.groupby(['is_paid']).mean()
plt.figure(figsize = (10,4))

sns.countplot(y = df.subject, data = df, order = df.subject.value_counts().index)

plt.title("Top Subject in Udemy", fontsize = 14)

plt.show()
plt.figure(figsize = (10,4))

sns.countplot(y = df.level, data = df, order = df.level.value_counts().index)

plt.title("Udemy Courses by Levels", fontsize = 14)

plt.show()
# df.price.value_counts()



#Step 1: Converting Free

df.price = df.price.replace('Free', 0)



#Step 2: Delete the 1 row where price is = TRUE

that_one_element = df[df.price == 'TRUE'].index

df.drop(that_one_element, inplace = True, axis = 0)



#Step 3: Convert column to integer

df.price = pd.to_numeric(df['price'])
plt.figure()

plt.subplots(figsize = (14,2))

sns.boxplot(x = df.price, data = df, color = 'red')

plt.title('Boxplot of the price of courses in Udemy',fontsize = 14)

plt.xlabel('Price in USD') #I am assuming all prices are in USD



plt.subplots(figsize = (14,8))

sns.boxplot(x = df.price, y = df.subject, data = df)

plt.title('Boxplot of the price of courses in Udemy by Subject',fontsize = 14)

plt.xlabel('Price in USD') #I am assuming all prices are in USD

plt.show()
print(df.corr())



%matplotlib inline

plt.figure(figsize = (10,5))



plt.subplot(221)

sns.regplot(x=df.price, y=df.num_subscribers, data = df)

plt.title('Scatterplot between price and num_subscribers.')

plt.text(100, 210000,'Corr = 0.05')



plt.subplot(222)

sns.regplot(x=df.price, y=df.num_reviews,data = df)

plt.title('Scatterplot between price and num_reviewers')

plt.text(100, 22000,'Corr = 0.11')



plt.subplot(223)

sns.regplot(x=df.price, y=df.num_lectures,data = df)

plt.title('Scatterplot between price and num_lectures')

plt.text(100, 650,'Corr = 0.33')



plt.tight_layout()

plt.show()
df.content_duration.head()
#Removing character values, leaving the numbers and decimal point

decimal_num_only = re.compile(r'[^\d.]+')

for i in np.arange(0,len(df.content_duration)+1):

    if i == 2066: #Because I removed row 2066, include this so there is no error

        continue     

    df.content_duration[i] = decimal_num_only.sub('', df.content_duration[i])



#Converting the feature to numeric    

df.content_duration = pd.to_numeric(df.content_duration)    
plt.figure()

sns.regplot(x=df.price, y=df.content_duration,data = df)

plt.title('Scatterplot Between Price and Content Duration (hours).')

plt.ylabel('Content Duration (Hours)')

plt.show()
plt.figure()

plt.hist(x= df.price, bins = 40, color = 'seagreen')

plt.title('Histogram of the Prices of Udemy\'s Courses.')

plt.xlabel('Price in USD')

plt.ylabel('Count')

plt.show()
#Extracting year from date time. 

df['year'] = pd.to_datetime(df.published_timestamp).dt.year
plt.figure(figsize = (14,7))

sns.countplot(x = df.year, data = df)

plt.title("Amount of Udemy Courses Created per Year", fontsize = 14)

plt.show()
plt.figure(figsize = (14,7))

sns.countplot(x = df.year, data = df, hue = df.subject)

plt.title("Amount of Udemy Courses Created per Year by Subject", fontsize = 14)

plt.show()
%%time

wordcloud_word_list = ''

stopwords = set(STOPWORDS)



wordcloud =WordCloud(height = 1000, width = 1000).generate(str(df.course_title).lower())
#Plotting the wordcloud

plt.figure(figsize = (20,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()