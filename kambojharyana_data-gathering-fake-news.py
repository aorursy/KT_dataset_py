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
data1 = pd.read_csv('../input/fake-news/train.csv',index_col = 0)
print(data1.shape)
data1.head()
# it has some null values, but we are concerned about null values in text col
data1.isnull().sum()
#remove all the values where text is null
data1 = data1.dropna()
data1.shape
data1.reset_index(inplace = True)
# it has even share for both fake and true news
data1['label'].value_counts() / len(data1)
#its around 4552
np.mean([len(data1['text'][i]) for i in range(len(data1))])
#lets see the avegare of headlines
np.mean([len(data1['title'][i]) for i in range(len(data1))])
data1.dropna(inplace = True)
data1.shape
#encoding the cat values
data1['label'] = np.where(data1['label'] == 0,'True','Fake')
#renaming the cols
data1 = data1[['title','text','label']]
data1.head()
# getting all the information in one col
data1['text'] = data1['title'] 
# all the dataframe will be in the similar structure
data1 = data1[['text','label']]
print(data1.shape)
data1.head()
data2 = pd.read_csv('../input/india-headlines-news-dataset/india-news-headlines.csv')
print(data2.shape)
data2.head()
# selecting news from 2020
data2 = data2[data2['publish_date'] > 20200000]
print(data2.shape)
data2.head()
data2.reset_index(inplace=True)
data2['headline_category'].nunique()
cat = data2['headline_category'].unique()
print(cat)
# selecting news only from city category
def func1(x):
    if 'city' in x:
        return x
    return np.nan    

data2['headline_category'] = list(map(func1,data2['headline_category']))
data2.isnull().sum()
data2['headline_category'].unique()
data2 = data2.dropna()
data2.reset_index(inplace = True)
#avg word count
np.max([len(data2['headline_text'][i]) for i in range(len(data2))])
data2['label'] = 'True'
data2 = data2[['headline_text','label']]
print(data2.shape)
data2.head()
data2.columns = ['text','label']
print(data2.shape)
data2.head()
# selecting 5000 random rows
import random
random_sample = random.sample(list(data2.index),5000)
data2 = data2.iloc[random_sample]
print(data2.shape)
data2.head()
data2.reset_index(inplace=True,drop = True)
data2.head()
data3 = pd.read_csv('../input/hate-speech-and-offensive-language-dataset/labeled_data.csv',index_col = 0)
print(data3.shape)
data3.head()
data3.reset_index(inplace = True)
np.mean([len(data3['tweet'][i]) for i in range(len(data3))])
data3['label'] = 'Fake'
data3 = data3[['tweet','label']]
data3.head()
#selecting 5000 random rows
random_sample = random.sample(list(data3.index),5000)
data3 = data3.iloc[random_sample]
print(data3.shape)
data3.head()
data3.columns = ['text','label']
data3.head()

data4 = pd.read_csv('../input/covid19-india-news-headlines-for-nlp/raw_data.csv')
print(data4.shape)
data4.head()
data4.isnull().sum()
data4 = data4[['Headline','Description']]
data4['label'] = True
data4.shape
data4['text'] = data4['Headline'] 
data4 = data4[['text','label']]
print(data4.shape)
data4.head()
np.mean([len(data4['text'][i]) for i in range(len(data4))])
data5 = pd.read_csv('../input/source-based-news-classification/news_articles.csv')
print(data5.shape)
data5.head()
data5.dropna(inplace = True)
data5['language'].value_counts()
data5 = data5[data5['language'] == 'english']
data5.shape
data5 = data5[['title_without_stopwords','text_without_stopwords','label']]
data5.head()
data5['text'] = data5['title_without_stopwords'] 
data5 = data5[['text','label']]
data5.head()
np.mean([len(data5['text'][i]) for i in range(len(data5))])
data_final = pd.concat([data1,data2,data3,data4,data5])
data_final.shape
data_final.head()
data_final.isnull().sum()
data_final.reset_index(inplace = True,drop=True)
data_final.to_csv('data.csv')
data_final['label'].unique()
data_final['label'].value_counts() / len(data_final)
