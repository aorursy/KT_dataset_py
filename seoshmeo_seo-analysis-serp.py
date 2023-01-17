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
june6 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-06-15.csv")

aug15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-08-15.csv")

mar1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-03-01.csv")

oct1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-10-01.csv")

jan1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-01-01.csv")

jan15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-01-15.csv")

may15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-05-15.csv")

feb15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2020-02-15.csv")

jul15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-07-15.csv")

feb1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-02-01.csv")

apr15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-04-15.csv")

jun1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-06-01.csv")

apr1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-04-01.csv")

mar15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-03-15.csv")

mar1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2020-03-01.csv")

sept15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-09-15.csv")

dec16 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2018-12-16.csv")

jul1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-07-01.csv")

may1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-05-01.csv")

feb15 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-02-15.csv")

sept1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-09-01.csv")

aug1 = pd.read_csv("/kaggle/input/search-engine-results-flights-tickets-keywords/flights_tickets_serp2019-08-01.csv")

june6.head(3)
june6.displayLink.value_counts()
all = pd.concat([june6, aug15, mar1, oct1, jan1, jan15, may15, feb15, jul15, feb1, apr15, jun1, apr1 , mar15, mar1, sept15, dec16, jul1, may1, feb15, sept1, aug1])

most_popular = all.displayLink.value_counts()



most_popular
mean_rank = all[['displayLink', 'rank']].groupby(['displayLink'], as_index=False).mean().sort_values(by='rank')

mean_rank
all.link.value_counts()