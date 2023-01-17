

import datetime 

import json 

import numpy as np 

from sklearn import covariance, cluster 

import pandas as pd

import pandas_datareader.data as pdd



input_file = "../input/json-company/company_symbol_mapping.json" 

with open(input_file, 'r') as f: 

    company_symbols_map = json.loads(f.read()) 

symbols = list(company_symbols_map.keys())

                   

QUANDL_API_KEY = '9x1tmjss9Wssuz6MSWWT'

assert QUANDL_API_KEY != 'xxxxxxxxxxxxxxxxxxxx', 'You must set a valid API KEY.'



start_date = datetime.datetime(2003, 7, 3) 

end_date = datetime.datetime(2007, 5, 4) 



quotes = []

names = []

for symbol in symbols:

    try:

        print('Loading', symbol, company_symbols_map[symbol], end='...')

        d = pdd.DataReader('WIKI/' + symbol, 'quandl', start_date, end_date,

                           access_key=QUANDL_API_KEY)

        print('done.')

        quotes.append(d)

        names.append(company_symbols_map[symbol])

    except:

        print('not found.')

names = np.array(names)



opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float) 

closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float) 

quotes_diff = closing_quotes - opening_quotes 



X = quotes_diff.copy().T 

X /= X.std(axis=0) 



edge_model = covariance.GraphicalLassoCV(cv=3) 



with np.errstate(invalid='ignore'): 

    edge_model.fit(X) 



_, labels = cluster.affinity_propagation(edge_model.covariance_) 

num_labels = labels.max() 



for i in range(num_labels + 1): 

    print("Cluster", i+1, "==>", ', '.join(names[labels == i]))