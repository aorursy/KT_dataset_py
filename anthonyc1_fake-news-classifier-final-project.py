import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
fake_news = pd.read_csv("../input/fake-news/fake.csv")
real_news = pd.read_csv("../input/gathering-real-news-for-oct-dec-2016/real_news.csv")
#Here are the size of our datasets:
print(fake_news.shape)
print(real_news.shape)
# Let's see what columns we have
print(list(fake_news.columns))
print(list(real_news.columns))
# now let's obtain similar features for both datasets before we combine them
# Let's add our label to the dataset "real" for real news and "fake" for fake news

real_news2 = real_news[['title', 'content', 'publication']]
real_news2['label'] = 'real'
real_news2.head(15)
fake_news2 = fake_news[['title', 'text','site_url']]
fake_news2['label'] = 'fake'
fake_news2.head(15)
# let's obtain all the unique site_urls
site_urls = fake_news2['site_url']

# let's remove the domain extensions
site_urls2 = [x.split('.',1)[0] for x in site_urls]

# now let's replace the old site_url column
fake_news2['site_url'] = site_urls2
fake_news2.head()
# let's rename the features in our datasets to be the same
newlabels = ['title', 'content', 'publication', 'label']
real_news2.columns = newlabels
fake_news2.columns = newlabels

# let's concatenate the dataframes
frames = [fake_news2, real_news2]
news_dataset = pd.concat(frames)
news_dataset
news_dataset.to_csv('news_dataset.csv', encoding='utf-8')