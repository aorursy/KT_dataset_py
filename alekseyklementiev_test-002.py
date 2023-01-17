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
from sklearn.feature_extraction import DictVectorizer
nRows = 100000
events = pd.read_csv('../input/events.csv', nrows=nRows)

events.head()
len(events.display_id.unique())
geoUpdate = events.geo_location.ravel()

print(geoUpdate)
geoNumeric = DictVectorizer()



X_train_categ = geoNumeric.fit_transform(geoUpdate) #Обучающая

#X_test_categ = geoNumeric.transform(test) #проверочная

print(X_train_categ)
clicks_train = pd.read_csv('../input/clicks_train.csv', nrows=nRows)

clicks_train.head()
len(clicks_train.ad_id.unique())
page_views_sample = pd.read_csv('../input/page_views_sample.csv', nrows=nRows)

page_views_sample.head()
promoted_content = pd.read_csv('../input/promoted_content.csv', nrows=nRows)

promoted_content.head()
len(promoted_content.ad_id.unique())
len(promoted_content.document_id.unique())