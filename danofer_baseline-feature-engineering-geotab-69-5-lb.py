import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing



from sklearn.linear_model import LinearRegression, LassoLarsCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

# from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load Data



train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv").sample(frac=0.15,random_state=42)#,nrows=123456)

test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")

train.nunique()
print(train["City"].unique())

print(test["City"].unique())
# test.groupby(["City"]).apply(np.unique)

test.groupby(["City"]).nunique()
train.isna().sum(axis=0)
test.isna().sum(axis=0)
directions = {

    'N': 0,

    'NE': 1/4,

    'E': 1/2,

    'SE': 3/4,

    'S': 1,

    'SW': 5/4,

    'W': 3/2,

    'NW': 7/4

}
train['EntryHeading'] = train['EntryHeading'].map(directions)

train['ExitHeading'] = train['ExitHeading'].map(directions)



test['EntryHeading'] = test['EntryHeading'].map(directions)

test['ExitHeading'] = test['ExitHeading'].map(directions)
train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration



test['diffHeading'] = test['EntryHeading']-test['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration



train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(10)
### code if we wanted the diffs, without changing the raw variables:



# train['diffHeading'] = train['ExitHeading'].map(directions) - train['EntryHeading'].map(directions)

# test['diffHeading'] = test['ExitHeading'].map(directions) - test['EntryHeading'].map(directions)
train.head()
train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)

test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)
le = preprocessing.LabelEncoder()

# le = preprocessing.OneHotEncoder(handle_unknown="ignore") # will have all zeros for novel categoricals, [can't do drop first due to nans issue , otherwise we'd  drop first value to avoid colinearity
train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]

test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]



print(train["Intersection"].sample(6).values)
train.head()
test.head()
# pd.concat([train,le.transform(train["Intersection"].values.reshape(-1,1)).toarray()],axis=1).head()
pd.concat([train["Intersection"],test["Intersection"]],axis=0).drop_duplicates().values
le.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)

train["Intersection"] = le.transform(train["Intersection"])

test["Intersection"] = le.transform(test["Intersection"])
train.head()
pd.get_dummies(train["City"],dummy_na=False, drop_first=False).head()
# pd.get_dummies(train[["EntryHeading","ExitHeading","City"]].head(),prefix = {"EntryHeading":'en',"ExitHeading":"ex","City":"city"})
train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)

test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
train.shape,test.shape
test.head()
train.columns
FEAT_COLS = ["IntersectionId",

             'Intersection',

           'diffHeading',  'same_street_exact',

           "Hour","Weekend","Month",

          'Latitude', 'Longitude',

          'EntryHeading', 'ExitHeading',

            'Atlanta', 'Boston', 'Chicago',

       'Philadelphia']
train.head()
train.columns
X = train[FEAT_COLS]

y1 = train["TotalTimeStopped_p20"]

y2 = train["TotalTimeStopped_p50"]

y3 = train["TotalTimeStopped_p80"]

y4 = train["DistanceToFirstStop_p20"]

y5 = train["DistanceToFirstStop_p50"]

y6 = train["DistanceToFirstStop_p80"]
y = train[['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',

        'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]
testX = test[FEAT_COLS]
## kaggle kernel performance can be very unstable when trying to use miltuiprocessing

# lr = LinearRegression()

lr = RandomForestRegressor(n_estimators=100,min_samples_split=3)#,n_jobs=3) #different default hyperparams, not necessarily any better
## Original: model + prediction per target

#############



lr.fit(X,y1)

pred1 = lr.predict(testX)

lr.fit(X,y2)

pred2 = lr.predict(testX)

lr.fit(X,y3)

pred3 = lr.predict(testX)

lr.fit(X,y4)

pred4 = lr.predict(testX)

lr.fit(X,y5)

pred5 = lr.predict(testX)

lr.fit(X,y6)

pred6 = lr.predict(testX)





# Appending all predictions

all_preds = []

for i in range(len(pred1)):

    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:

        all_preds.append(j[i])   



sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

sub["Target"] = all_preds

sub.to_csv("benchmark_beat_rfr_multimodels.csv",index = False)



print(len(all_preds))
## New/Alt: multitask -  model for all targets



lr.fit(X,y)

print("fitted")



all_preds = lr.predict(testX)
## convert list of lists to format required for submissions

print(all_preds[0])



s = pd.Series(list(all_preds) )

all_preds = pd.Series.explode(s)



print(len(all_preds))

print(all_preds[0])
sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

print(sub.shape)

sub.head()
sub["Target"] = all_preds.values

sub.sample(5)
sub.to_csv("benchmark_beat_rfr_multitask.csv",index = False)
train.drop("Path",axis=1).to_csv("train_danFeatsV1.csv.gz",index = False,compression="gzip")

test.drop("Path",axis=1).to_csv("test_danFeatsV1.csv.gz",index = False,compression="gzip")