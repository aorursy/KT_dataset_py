import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

print(check_output(["ls", "../input"]).decode("utf8"))
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
df = pd.read_json('../input/train.json')

df.tail()
print('Unique: ', len(set(df['manager_id'])))

print('Total: ', len(df['manager_id']))

print(len(df['manager_id'])/len(set(df['manager_id'])))
df['address'] = df['display_address'].astype('category').cat.codes

df['street_address'] = df['street_address'].astype('category').cat.codes

df['building_id'] = df['building_id'].astype('category').cat.codes

df['manager_id'] = df['manager_id'].astype('category').cat.codes

df['num_features'] = df['features'].apply(len)

df['created'] = pd.to_datetime(df['created'])

df['created_year'] = df['created'].dt.year.astype('category').cat.codes

df['created_month'] = df['created'].dt.month.astype('category').cat.codes

df['len_description'] = df['description'].apply(lambda x: len(x.split(' ')))

df['num_pics'] = df['photos'].apply(len)
new_feat = ['price','address','manager_id','building_id',

            'num_features','created_year','created_month',

            'len_description','latitude','longitude','num_pics']



#new_feat = ['price','latitude','longitude','num_pics',

 #           'num_features','created_year','created_month','len_description']

X = df[new_feat].fillna(0)

y = df['interest_level'].astype('category').cat.codes

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=9)

X.tail()
clf1 = GradientBoostingClassifier(n_estimators=200, max_depth=9)

clf2 = AdaBoostClassifier(n_estimators=200)

clf3 = RandomForestClassifier(n_estimators=300)



estimators = [('gb', clf1), ('ab', clf2), ('rf', clf3)]

vclf = VotingClassifier(estimators=estimators, voting='soft', n_jobs= -1)



vclf.fit(X_train, y_train)

y_val_pred = vclf.predict_proba(X_val)

log_loss(y_val, y_val_pred)
X_train = df[new_feat].fillna(0)

y_train = df['interest_level']

vclf.fit(X_train, y_train)





df2 = pd.read_json('../input/test.json')

df2['address'] = df2['display_address'].astype('category').cat.codes

df2['street_address'] = df2['street_address'].astype('category').cat.codes

df2['building_id'] = df2['building_id'].astype('category').cat.codes

df2['manager_id'] = df2['manager_id'].astype('category').cat.codes

df2['num_features'] = df2['features'].apply(len)

df2['created'] = pd.to_datetime(df2['created'])

df2['created_year'] = df2['created'].dt.year.astype('category').cat.codes

df2['created_month'] = df2['created'].dt.month.astype('category').cat.codes

df2['len_description'] = df2['description'].apply(lambda x: len(x.split(' ')))

df2['num_pics'] = df2['photos'].apply(len)



X = df2[new_feat].fillna(0)

y = vclf.predict_proba(X)
labels2idx = {label: i for i, label in enumerate(vclf.classes_)}

labels2idx
sub = pd.DataFrame()

sub['listing_id'] = df2['listing_id']

for l in ['high', 'medium', 'low']:

    sub[l] = y[:, labels2idx[l]]

sub.to_csv('submissionVoting.csv', index=False)
sub = pd.read_csv('submissionVoting.csv')

sub.head()