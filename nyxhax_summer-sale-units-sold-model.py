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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.dummy import DummyClassifier
r_state = 16

pred_column = 'units_sold'
df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

df = df.drop(['price', 'merchant_id', 'merchant_has_profile_picture',

		'title', 'title_orig', 'currency_buyer','rating_count', 'product_variation_inventory',

       'badges_count', 'badge_local_product', 'badge_product_quality', 'tags', 

       'product_variation_size_id', 'shipping_option_name', 'shipping_is_express',

       'countries_shipped_to', 'inventory_total', 'has_urgency_banner',

       'urgency_text', 'origin_country', 'merchant_title', 'merchant_name',

       'merchant_info_subtitle', 'merchant_rating_count',

       'merchant_profile_picture', 'product_url', 'product_picture',

       'product_id', 'theme', 'crawl_month'], 1)



df = df.dropna()



le = LabelEncoder()

df['product_color'] = le.fit_transform(df['product_color'])
print(df.head())

print(df.columns)

print(df.shape)
X = np.array(df.drop([pred_column], 1))

y = np.array(df[pred_column])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, random_state=r_state)
scaler = MinMaxScaler().fit(X)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
dummy = DummyClassifier(strategy='most_frequent', random_state=r_state)

dummy.fit(X_train, y_train)

dummy_score = dummy.score(X_test, y_test)

print('Dummy {}'.format(dummy_score))
for algo in [BaggingClassifier, DecisionTreeClassifier, RandomForestClassifier]:

    model = algo(random_state=r_state)

    model.fit(X_train, y_train)



    score = model.score(X_test, y_test)

    print('{}: {}'.format(algo, score))