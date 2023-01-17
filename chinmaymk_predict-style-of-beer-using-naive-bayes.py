# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
beers = pd.read_csv('../input/beers.csv')

breweries = pd.read_csv('../input/breweries.csv')
breweries.head(10)
data = beers.merge(breweries, how='left', left_on='brewery_id', right_on='Unnamed: 0')

data.head(10)
data['style'].unique()
final = data[['abv', 'ibu', 'ounces', 'style']]

final['city_state'] = data['city'] + data['state']

final.head()
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
features = pd.get_dummies(final.drop('style', axis=1))

features = features.fillna(value=0)

final['style_codes'] = final['style'].astype('category').cat.codes
model.fit(features, final['style_codes'])
model.score(features, final['style_codes'])
final.groupby('style').size()
boiled_down = final.groupby('style').filter(lambda x: len(x) > 10)
features_bd = pd.get_dummies(boiled_down.drop(['style', 'style_codes'], axis=1))

features_bd.fillna(0, inplace=True)
model.fit(features_bd, boiled_down['style_codes'])
model.score(features_bd, boiled_down['style_codes'])
model.predict_proba(features_bd.iloc[1:2])
model.predict(features_bd.iloc[1:2])