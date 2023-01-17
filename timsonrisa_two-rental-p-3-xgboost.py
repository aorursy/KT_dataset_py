import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
def PrepareFeatures(TestOrTrain):
    source = f'../input/{TestOrTrain}.json'
    data = pd.read_json(source)
    ulimit = np.percentile(data.price.values, 99)
    data['price'][data['price']>ulimit] = ulimit
    data['hasDesc'] = data['description'].apply(lambda x: len(x.strip())!=0)
    data["nFeatures"] = data["features"].apply(len)
    data["nDescWords"] = data["description"].apply(lambda x: len(x.split(" ")))
    data['nPhotos'] = data['photos'].apply(lambda x: min(10, len(x)))
    data['created'] = pd.to_datetime(data['created'])
    data['month'] = data['created'].dt.month
    data['weekday'] = data['created'].apply(lambda x: x.weekday())
    
    if TestOrTrain == 'train':
        interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
        data['interest_level'] = data['interest_level'].apply(lambda x: interest_level_map[x])
    return data

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
param = {'eval_metric':"mlogloss", 'eta':0.02, 'objective':'multi:softprob', 'silent':1,
        'max_depth':10, 'num_class':3, 'subsample':0.7, 'colsample_bytree':0.7}
num_rounds = 500
xgtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(param, xgtrain, num_rounds)
xgval = xgb.DMatrix(X_val)
y_val_pred = model.predict(xgval) 
log_loss(y_val, y_val_pred)
test = PrepareFeatures('test')
CreateCategFeat(test, cat_features)
X_test = test[features]
xgtest = xgb.DMatrix(X_test)
y = model.predict(xgtest)
labels2idx = {'low': 0, 'medium': 1, 'high': 2}
sub = pd.DataFrame()
sub["listing_id"] = test["listing_id"]
for label in labels2idx.keys():
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)