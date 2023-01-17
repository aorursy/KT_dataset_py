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
import pandas as pd

df = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')

tdf = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')



import re

from datetime import datetime

from numpy import nan

import numpy as np



capitals = {

    'Paris'     : 1, # столица франции

    'Stockholm' : 1, # столица Швеции

    'London'    : 1, # столица Великобритании

    'Berlin'    : 1, # столица Германии

    'Munich'    : 0, #         Германии

    'Oporto'    : 0, #         Португалии

    'Milan'     : 0, #         Италии

    'Bratislava': 1, # столица Словакии

    'Vienna'    : 1, # столица Австрии

    'Rome'      : 1, # столица Италии

    'Barcelona' : 0, #         Испании

    'Madrid'    : 1, # столица Испании

    'Dublin'    : 1, # столица Ирландии

    'Brussels'  : 1, # столица Бельгии

    'Zurich'    : 0, #         Швейцарии

    'Warsaw'    : 1, # столица Польши

    'Budapest'  : 1, # столица Венгрии

    'Copenhagen': 1, # столица Дании

    'Amsterdam' : 1, # столица Нидерландов

    'Lyon'      : 0, #         Франции

    'Hamburg'   : 0, #         Германии

    'Lisbon'    : 1, # столица Португалии

    'Prague'    : 1, # столица Чешской Республики

    'Oslo'      : 1, # столица Норвегии

    'Helsinki'  : 1, # столица Финляндии

    'Edinburgh' : 1, # столица Шотландии

    'Geneva'    : 0, #         Швейцарии

    'Ljubljana' : 1, # столица Словении

    'Athens'    : 1, # столица Греции

    'Luxembourg': 1, # столица Люксембург

    'Krakow'    : 0, #         Польши

}



format = '%m/%d/%Y'

now = datetime.now()



def find_recent_date(reviews):

    recent_date = None

    if isinstance(reviews, str):

        review_lists = eval(reviews)

        dates = review_lists[1] if len(review_lists) > 1 else [] 

        for date in dates:

            if recent_date == None or recent_date < datetime.strptime(date, format):

                recent_date = datetime.strptime(date, format)

    return recent_date



def find_min_date(reviews):

    min_date = None

    max_date = None

    review_lists = None

    try:

        review_lists = eval(reviews)

    except:

        review_lists = []

    dates = review_lists[1] if len(review_lists) > 1 else [] 

    for dt in dates:

        date = datetime.strptime(dt, format)

        if min_date == None or min_date > date:

            min_date = date

        if max_date == None or max_date < date:

            max_date = date

    return -1 if min_date == None and max_date == None else (max_date - min_date).days



def find_item(cell, item):

    if isinstance(cell, str):

        if item in pattern.findall(cell):

            return 1

    return 0



def cleandf(df):

    for k,v in capitals.items():

        df[k] = df['City'].apply(lambda x: int(x == k))

    

    df['capital'] = df['City'].map(capitals)

    

    for item in cuisines:

        df[item] = df['Cuisine Style'].apply(find_item, item=item)

    

    df['RecentDate'] = df['Reviews'].apply(find_recent_date)

    df['SinceLastReview'] = df['RecentDate'].apply(lambda x: (now - x).days if isinstance(x, datetime) else 0)

    df['SinceLastReview'] = df['SinceLastReview'].fillna(0)

    df['max_days_between'] = df.Reviews.apply(find_min_date)

    

    df['lowcost'] = df['Price Range'].apply(lambda x: 1 if x == '$' else 0)

    df['midcost'] = df['Price Range'].apply(lambda x: 1 if x == '$$ - $$$' else 0)

    df['higcost'] = df['Price Range'].apply(lambda x: 1 if x == '$$$$' else 0)



pattern = re.compile(r'\'([^\']+)\'')

cuisines = set()

df['Cuisine Style'].apply(lambda x: cuisines.update(pattern.findall(x)) if isinstance(x, str) else None)



cleardf = cleandf(df)

cleartdf = cleandf(tdf)



X_train = df.drop(['Restaurant_id', 'Rating', 'RecentDate'], axis = 1)

y_train = df['Rating']



X_pred = tdf.drop(['Restaurant_id', 'RecentDate'], axis = 1)



X_train = X_train.select_dtypes(exclude=['object'])

X_pred = X_pred.select_dtypes(exclude=['object'])

X_train = X_train.fillna(0)

y_train = y_train.fillna(0)

X_pred = X_pred.fillna(0)

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

regr = RandomForestRegressor(n_estimators=100)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_pred)

y_pred2 = pd.DataFrame({'Restaurant_id': tdf['Restaurant_id'], 'Rating': y_pred})

y_pred2.to_csv('solution.csv', index = False)



# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))