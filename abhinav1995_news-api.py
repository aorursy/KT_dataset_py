import pandas as pd
from datetime import * 
import requests
from datetime import * 
import json
import io
from io import StringIO
import os
from pandas import DataFrame
import math

# Defining the Function for the API and creating a wrapper

# insert the api key after registering on https://newsapi.org/register

api_key = ''
def news_api_call(country, category):
    url = ('https://newsapi.org/v2/top-headlines?country='+country+'&category='+category+'&pageSize=100&apiKey'+api_key)
    response = requests.get(url)
    data = response.json()
    pt = []
    for i in range(0,data['totalResults']):
        publishtime = datetime.strptime(data['articles'][i]['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        pt.append(publishtime)
    print('Total news articles returned: ' + str(data['totalResults']))    
    print('Latest Article retrieved: ' + str(max(pt)) + ' GMT')
    data_size = data['totalResults']
    title_ = []
    desc_ = []
    url_ = []
    content_ = []
    publishtime_ = []
    if data_size <= 100:
        for i in range(0,data['totalResults']):
            title = data['articles'][i]['title']
            desc = data['articles'][i]['description']
            url = data['articles'][i]['url']
            content = data['articles'][i]['content']
            publishtime = str(datetime.strptime(data['articles'][i]['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'))
            
            title_.append(title)
            desc_.append(desc)
            url_.append(url)
            content_.append(content)
            publishtime_.append(publishtime)
    else:
        x = data['totalResults']/100
        t = math.ceil(x)
        title_ = []
        desc_ = []
        url_ = []
        content_ = []
        publishtime_ = []
        for i in range(1, t+1):
            url = ('https://newsapi.org/v2/top-headlines?country='+country+'&category='+category+'&pageSize=20&page='+str(i)+'&apiKey=5338e06cd1154f6b84eadcfbda2d09e6')
            response = requests.get(url)
            data = response.json()
            for t in range(0, len(data['articles'])):
                        title = data['articles'][t]['title']
                        desc = data['articles'][t]['description']
                        url = data['articles'][t]['url']
                        content = data['articles'][t]['content']
                        publishtime = str(datetime.strptime(data['articles'][t]['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'))
                        
                        title_.append(title)
                        desc_.append(desc)
                        url_.append(url)
                        content_.append(content)
                        publishtime_.append(publishtime)
    
    zip_data = list(zip(publishtime_, title_, desc_, content_, url_))
    df = pd.DataFrame(zip_data, columns = ['Publish_Time_GMT','title', 'description', 'Content', 'url'])
    df['Region'] = country
    df['Category'] = category
    return df
df_de_business = news_api_call('de', 'business')
list(df_de_business.columns)
# Save the output to a csv
df_de_business.to_csv(r'de_business_02072020_news.csv', index = None, header=True, encoding = 'utf-8-sig')