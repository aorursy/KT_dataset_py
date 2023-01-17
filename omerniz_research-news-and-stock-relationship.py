import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
news_data = pd.read_csv('../input/all-the-news/articles1.csv')[:70000]
news_data = news_data.rename(columns={'date':'dateOfNews'})
news_data = news_data.set_index('dateOfNews')
news_data = news_data[['title', 'publication']]
dummies = pd.get_dummies(news_data['publication'])
news_data[dummies.columns] = dummies
news_data = news_data.drop(['publication'], axis=1)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english')
X = vect.fit_transform(news_data.pop('title')).toarray()
for i, col in enumerate(vect.get_feature_names()):
    news_data[col] = X[:, i]
    
news_data.index = news_data.index.astype('datetime64[ns]')
news_data = news_data.groupby('dateOfNews').mean()
appl_stock = pd.read_csv('../input/apple-stock/AAPL_data.csv')
appl_stock.date = appl_stock.date.astype('datetime64[ns]')
appl_stock = appl_stock.set_index('date')
summed = appl_stock['open'].rolling(2).sum()
appl_stock['isWentUp'] =  2*appl_stock['open'] > summed
appl_stock = appl_stock[['isWentUp']]
appl_stock.tail()
joined = news_data.join(appl_stock, how='inner')
joined.to_csv('apple-stock-result_70000_articles.csv')