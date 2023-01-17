import pandas as pd

import json

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
with open('../input/train.json') as train_data:

    data = json.load(train_data)
features = [x['ingredients'] for x in data]
features_list = []

for feature in features:

    single_item = ''

    for item in feature:

        item= item.replace(' ','-')

        single_item = single_item + ' ' + item

    single_item = single_item[1:]

    features_list.append(single_item)
vectorizor = CountVectorizer()

X_features = vectorizor.fit_transform(features_list)
le = LabelEncoder()

labels = [x['cuisine'] for x in data]

y = le.fit_transform(labels)
with open('../input/test.json') as test_data:

    test_data = json.load(test_data)
features_test = [x['ingredients'] for x in test_data]
features_list_test = []

for feature in features_test:

    single_item = ''

    for item in feature:

#         item= item.replace(' ','-')

        single_item = single_item + ' ' + item

    single_item = single_item[1:]

    features_list_test.append(single_item)
test_features = vectorizor.transform(features_list_test)
model= LogisticRegression()

model.fit(X_features,y)
model.score(X_features, y)
df = pd.read_json('../input/test.json')
df['cuisine'] = le.inverse_transform(model.predict(test_features))
df= df[['id', 'cuisine']]

df.to_csv('submit.csv', index=False)