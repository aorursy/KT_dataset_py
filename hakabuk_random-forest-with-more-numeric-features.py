# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

pd.options.mode.chained_assignment = None  # default='warn'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(['ls', "../input"]).decode('utf8'))
df = pd.read_json(open("../input/train.json", 'r'))

df.head()
df['num_photos'] = df['photos'].apply(len)

df['num_features'] = df['features'].apply(len)

df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))

df['created'] = pd.to_datetime(df['created'])

df['created_year'] = df['created'].dt.year

df['created_month'] = df['created'].dt.month

df['created_day'] = df['created'].dt.day

df['created_hour'] = df['created'].dt.hour

df['created_minute'] = df['created'].dt.minute
# function to remove outliers from given percentile (we'll use 99/1)

def remove_outlier(data, col, percent_list):

    for item in percent_list:

        ulimit = np.percentile(data[col].values, item)

        if item > 50:

            data[col].ix[data[col] > ulimit] = ulimit

        else:

            data[col].ix[data[col] < ulimit] = ulimit

    return data
# price: removing values in 99 percentile

df = remove_outlier(df, 'price', [99])



# Latitude & Longitude:

# removing outliers: values in the 1/99 percentiles

df = remove_outlier(df, 'latitude', [1, 99])

df = remove_outlier(df, 'longitude', [1, 99])
num_feats = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos',

             'num_features', 'num_description_words', 'created_year', 'created_month',

             'created_day', 'created_hour', 'created_minute']

X = df[num_feats]

y = df['interest_level']

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.33,

                                                    random_state=0)
clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train, y_train)

y_val_pred = clf.predict_proba(X_test)

log_loss(y_test, y_val_pred)
# fitting the model on the entire data without split

clf.fit(X, y)
df = pd.read_json(open("../input/test.json", 'r'))

df['num_photos'] = df['photos'].apply(len)

df['num_features'] = df['features'].apply(len)

df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))

df['created'] = pd.to_datetime(df['created'])

df['created_year'] = df['created'].dt.year

df['created_month'] = df['created'].dt.month

df['created_day'] = df['created'].dt.day

df['created_hour'] = df['created'].dt.hour

df['created_minute'] = df['created'].dt.minute

X = df[num_feats]
y = clf.predict_proba(X)
labels2idx = {label: i for i, label in enumerate(clf.classes_)}

labels2idx
sub = pd.DataFrame()

sub['listing_id'] = df['listing_id']

for label in ['high', 'medium', 'low']:

    sub[label] = y[:, labels2idx[label]]

sub.to_csv("submission_rf_01.csv", index=False)