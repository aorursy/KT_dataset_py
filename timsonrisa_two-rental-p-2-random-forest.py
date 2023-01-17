import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
def PrepareFeatures(TestOrTrain):
    source = f'../input/{TestOrTrain}.json'
    data = pd.read_json(source)
    # Some noise in price feature we saw in Part 1:
    ulimit = np.percentile(data.price.values, 99)
    data['price'][data['price']>ulimit] = ulimit
    # Get Different features as in Part 1:
    data['hasDesc'] = data['description'].apply(lambda x: len(x.strip())!=0)
    data["nFeatures"] = data["features"].apply(len)
    data["nDescWords"] = data["description"].apply(lambda x: len(x.split(" ")))
    data['nPhotos'] = data['photos'].apply(lambda x: min(10, len(x)))
    data['created'] = pd.to_datetime(data['created'])
    data['month'] = data['created'].dt.month
    data['weekday'] = data['created'].apply(lambda x: x.weekday())
    return data

# Using categorical (more sparse) data, we ispected in Part 1:
def CreateCategFeat(data, features_list):
    f_dict = {'hasParking':['parking', 'garage'], 'hasGym':['gym', 'fitness', 'health club'],
              'hasPool':['swimming pool', 'pool'], 'noFee':['no fee', "no broker's fees"],
              'hasElevator':['elevator'], 'hasGarden':['garden', 'patio', 'outdoor space'],
              'isFurnished': ['furnished', 'fully  equipped'], 
              'reducedFee':['reduced fee', 'low fee'],
              'hasAC':['air conditioning', 'central a/c', 'a/c', 'central air', 'central ac'],
              'hasRoof':['roof', 'sundeck', 'private deck', 'deck'],
              'petFriendly':['pets allowed', 'pet friendly', 'dogs allowed', 'cats allowed'],
              'shareable':['shares ok'], 'freeMonth':['month free'],
              'utilIncluded':['utilities included']}
    for feature in features_list:
        data[feature] = False
        for ind, row in data.iterrows():
            for f in row['features']:
                f = f.lower().replace('-', '')
                if any(e in f for e in f_dict[feature]):
                    data.at[ind, feature]= True
data = PrepareFeatures('train')
cat_features = ['hasParking', 'hasGym', 'hasPool', 'noFee', 'hasElevator',
                'hasGarden', 'isFurnished', 'reducedFee', 'hasAC', 'hasRoof',
                'petFriendly', 'shareable', 'freeMonth', 'utilIncluded']
CreateCategFeat(data, cat_features)
features = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "nPhotos", "hasDesc", 'nFeatures', 'nDescWords', "month", 'weekday']
features.extend(cat_features)
X = data[features]
y = data["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05)
clf = RandomForestClassifier(n_estimators=2000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)
for i, j in sorted(zip(X_train.columns, clf.feature_importances_)):
    print(i, j)
test = PrepareFeatures('test')
CreateCategFeat(test, cat_features)
X = test[features]
y = clf.predict_proba(X)
labels2idx = {label: i for i, label in enumerate(clf.classes_)}
sub = pd.DataFrame()
sub["listing_id"] = test["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)