import pandas as pd
df_feats = pd.read_csv('../input/ds1-kaggle-challenge/train_features.csv')

df_label = pd.read_csv('../input/ds1-kaggle-challenge/train_labels.csv') 
df_label.describe(include='object')
#let us see the whole column profile of the data frame 

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

df_feats.head()
df_feats.shape, df_label.shape
df_label.status_group.value_counts(normalize = True)
full = pd.DataFrame.merge(df_label,df_feats)
full.head()
full.isnull().sum()
clean = full.dropna(axis = 1)
clean.isna().sum()
from sklearn.model_selection import train_test_split

X1 = clean.drop(columns = ['status_group',], axis = 1)

y = clean['status_group']

X_train, X_test, y_train, y_test = train_test_split(X1, y,test_size = .5, random_state=42)
X_train.head()
X_train.isna().sum().sum()
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import category_encoders as ce

import numpy as np 

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):

        columnsToEncode = list(df.select_dtypes(include=['category','object']))

        le = LabelEncoder()

        for feature in columnsToEncode:

            try:

                df[feature] = le.fit_transform(df[feature])

            except:

                print('Error encoding '+feature)

        return df

      
X_train_DC = dummyEncode(X_train)

X_train_DC.head()

X_test_DC = dummyEncode(X_test)

X_test_DC.head()

X = dummyEncode(X1)
X_train_DC.isna().sum().sum()

X_train_DC.shape
model= LogisticRegression()

model.fit(X_train_DC, y_train)

y_pred = model.predict(X_test_DC)

accuracy_score(y_test, y_pred)

pipeline = make_pipeline(ce.OneHotEncoder(use_cat_names=True),

                         StandardScaler(), LogisticRegression(solver ='lbfgs',n_jobs=-1, multi_class = 'auto',C=2))

pipeline.fit(X_train_DC, y_train)
y_pred = pipeline.predict(X_train)

pred = pd.DataFrame(y_pred, X_train_DC['id'])
pred.columns = ['status_group']
pred.head()

pred.shape

pred.head()
newsub = pd.DataFrame(pred)

newsub.shape

sub_2 = newsub.index

subm = pd.DataFrame( newsub['status_group'],sub_2)

subm.head()

subm.reset_index(inplace = True)
#subm.to_csv('C:/Users/dakot/Documents/GitHub/sumbission1.csv',columns = ['id','status_group'], index = False )
subm.shape
df_test = pd.read_csv('../input/ds1-kaggle-challenge/test_features.csv') 
df_test.head()
df_test.isna().sum()

nona = df_test.dropna(axis = 1)
nona.shape
X = dummyEncode(nona)
X.head()
pipeline.fit(X_test, y_test)

y_preds = pipeline.predict(X)



y_preds.shape
preds = pd.DataFrame(y_preds, X['id'])

preds.columns = ['status_group']

preds.head()
newsubs = pd.DataFrame(preds)

newsubs.shape

sub_2s = newsubs.index

subms = pd.DataFrame( newsubs['status_group'],sub_2s)

subms.head()

subms.reset_index(inplace = True)
subms.head()
from sklearn import tree

from sklearn.metrics import classification_report

 

clf = tree.DecisionTreeClassifier(random_state=42)

clf = clf.fit(X_train, y_train)

 

y_pred2 = clf.predict(X)

#print(classification_report(y_test, y_pred2))

#print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred2)))
y_pred2.shape
def format(predictions):

    pre = pd.DataFrame(predictions, X['id'])

    pre.columns = ['status_group']

    new = pd.DataFrame(pre)

    sub_2s = new.index

    subs = pd.DataFrame( new['status_group'],sub_2s)

    subs.reset_index(inplace = True)

    print(subs.head(),subs.shape)

    subs.to_csv('C:/Users/dakot/Documents/GitHub/sumbission1.csv',columns = ['id','status_group'], index = False )

    return 'YAY!'

pipeline = make_pipeline(ce.OneHotEncoder(use_cat_names=True),

                         StandardScaler(), LogisticRegression(solver ='lbfgs',n_jobs=-1, multi_class = 'auto',C=2))

pipeline.fit(X_train, y_train)
pred3 = pipeline.predict(X)
pred3
treepipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True),

                         StandardScaler(),tree.DecisionTreeClassifier(random_state=42) )

treepipe.fit(X_train, y_train)
tpred = treepipe.predict(X_test)

print(accuracy_score(y_test,tpred))

pred4 = treepipe.predict(X)
from sklearn.preprocessing import RobustScaler

treepipe2 = make_pipeline(ce.OneHotEncoder(use_cat_names=True),

                         RobustScaler(),tree.DecisionTreeClassifier(random_state=42) )

treepipe2.fit(X_train, y_train)

pred = treepipe.predict(X_test)
accuracy_score(y_test,pred)
pred5 = treepipe2.predict(X)
pred5

# the training data set 

full.isna().sum()
full.funder.fillna(full.funder.describe().top,inplace = True)

full.installer.fillna(full.installer.describe().top,inplace = True)

full.subvillage.fillna(full.subvillage.describe().top, inplace = True)

full.public_meeting.fillna(full.public_meeting.describe().top,inplace = True)

full.scheme_management.fillna(full.scheme_management.describe().top, inplace = True)

full.scheme_name.fillna(full.scheme_name.describe().top, inplace = True)

full.permit.fillna(full.permit.describe().top,inplace = True)
full.isna().sum().sum()
full.columns
Xi = full.drop(columns= ['status_group','date_recorded'], axis = 1)

yi = full['status_group']
Xi.shape, yi.shape
# DJ split that S*&&%

X_train, X_test, y_train, y_test = train_test_split(Xi, yi,test_size = .5, random_state=42)

#now encode it

X_trains = dummyEncode(X_train)

X_tests = dummyEncode(X_test)
#how does it like the trees

cl = tree.DecisionTreeClassifier(random_state=42)

cl = clf.fit(X_trains, y_train)

 

y_predictor = clf.predict(X_tests)

print(classification_report(y_test, y_predictor))

print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_predictor)))

#accuracy of .699 for the train data when split how about the test data
test = df_test

print(test.shape)

test.funder.fillna(test.funder.describe().top,inplace = True)

test.installer.fillna(test.installer.describe().top,inplace = True)

test.subvillage.fillna(test.subvillage.describe().top, inplace = True)

test.public_meeting.fillna(test.public_meeting.describe().top,inplace = True)

test.scheme_management.fillna(test.scheme_management.describe().top, inplace = True)

test.scheme_name.fillna(test.scheme_name.describe().top, inplace = True)

test.permit.fillna(test.permit.describe().top,inplace = True)
test.head()
Xt = test.drop(columns = ['date_recorded'], axis = 1)

XT = dummyEncode(Xt)

#TREE ME!!!

cl = tree.DecisionTreeClassifier(random_state=42)

cl = clf.fit(X_trains, y_train)

 

y_predictors = clf.predict(XT)

# print(classification_report(y_test, y_predictor))

# print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_predictor)))
#lets hit the pipe testing with standard scale then robust, log_reg and tree

logpipe = make_pipeline(RobustScaler(),

                        tree.DecisionTreeClassifier(random_state=42) )

logpipe.fit(X_trains, y_train)

predlog = logpipe.predict(X_test)

accuracy_score(y_test,predlog)

# yeah .64 is no bueno with standard scaler logistic regression

#robust scale log_regression is.63 which doesnt tickle my fancy 

# Standard scale D tree gives .69 but im not impressed

#robust scale Dtree gives a slightly higher .699
# What about different encoding?

import category_encoders as ce

encoder = ce.HashingEncoder()

hashingpipe = make_pipeline(ce.HashingEncoder(),RobustScaler(),

                        tree.DecisionTreeClassifier(random_state=42) )

hashingpipe.fit(X_train, y_train)

predlogs = hashingpipe.predict(X_test)

accuracy_score(y_test,predlogs)

df_test1 = pd.read_csv('../input/ds1-kaggle-challenge/test_features.csv') 
from sklearn.preprocessing import MinMaxScaler

df_test1.isna().sum()

df_test1['gps_height'].replace(0.0, np.nan, inplace=True)

df_test1['population'].replace(0.0, np.nan, inplace=True)

df_test1['amount_tsh'].replace(0.0, np.nan, inplace=True)

df_test1.isnull().sum()
df_test1['gps_height'].fillna(df_test1.groupby(['region', 'district_code'])['gps_height'].transform('mean'), inplace=True)

df_test1['gps_height'].fillna(df_test1.groupby(['region'])['gps_height'].transform('mean'), inplace=True)

df_test1['gps_height'].fillna(df_test1['gps_height'].mean(), inplace=True)

df_test1['population'].fillna(df_test1.groupby(['region', 'district_code'])['population'].transform('median'), inplace=True)

df_test1['population'].fillna(df_test1.groupby(['region'])['population'].transform('median'), inplace=True)

df_test1['population'].fillna(df_test1['population'].median(), inplace=True)

df_test1['amount_tsh'].fillna(df_test1.groupby(['region', 'district_code'])['amount_tsh'].transform('median'), inplace=True)

df_test1['amount_tsh'].fillna(df_test1.groupby(['region'])['amount_tsh'].transform('median'), inplace=True)

df_test1['amount_tsh'].fillna(df_test1['amount_tsh'].median(), inplace=True)

df_test1.isnull().sum()

features=['amount_tsh', 'gps_height', 'population']

scaler = MinMaxScaler(feature_range=(0,20))

df_test1[features] = scaler.fit_transform(df_test1[features])

df_test1[features].head(20)

df_test1.isna().sum()

df_test1['longitude'].replace(0.0, np.nan, inplace=True)

df_test1['latitude'].replace(0.0, np.nan, inplace=True)

df_test1['construction_year'].replace(0.0, np.nan, inplace=True)

df_test1['latitude'].fillna(df_test1.groupby(['region', 'district_code'])['latitude'].transform('mean'), inplace=True)

df_test1['longitude'].fillna(df_test1.groupby(['region', 'district_code'])['longitude'].transform('mean'), inplace=True)

df_test1['longitude'].fillna(df_test1.groupby(['region'])['longitude'].transform('mean'), inplace=True)

df_test1['construction_year'].fillna(df_test1.groupby(['region', 'district_code'])['construction_year'].transform('median'), inplace=True)

df_test1['construction_year'].fillna(df_test1.groupby(['region'])['construction_year'].transform('median'), inplace=True)

df_test1['construction_year'].fillna(df_test1.groupby(['district_code'])['construction_year'].transform('median'), inplace=True)

df_test1['construction_year'].fillna(df_test1['construction_year'].median(), inplace=True)

df_test1['date_recorded'] = pd.to_datetime(df_test1['date_recorded'])

df_test1['years_service'] = df_test1.date_recorded.dt.year - df_test1.construction_year

print(df_test1.isnull().sum())

garbage=['wpt_name','num_private','subvillage','region_code','recorded_by','management_group',

         'extraction_type_group','extraction_type_class','scheme_name','payment',

        'quality_group','quantity_group','source_type','source_class','waterpoint_type_group',

        'ward','public_meeting','permit','date_recorded','construction_year']

df_test1.drop(garbage,axis=1, inplace=True)
#take out any random capital letters in the entries

df_test1.waterpoint_type = df_test1.waterpoint_type.str.lower()

df_test1.funder = df_test1.funder.str.lower()

df_test1.basin = df_test1.basin.str.lower()

df_test1.region = df_test1.region.str.lower()

df_test1.source = df_test1.source.str.lower()

df_test1.lga = df_test1.lga.str.lower()

df_test1.management = df_test1.management.str.lower()

df_test1.quantity = df_test1.quantity.str.lower()

df_test1.water_quality = df_test1.water_quality.str.lower()

df_test1.payment_type=df_test1.payment_type.str.lower()

df_test1.extraction_type=df_test1.extraction_type.str.lower()
df_test1.columns

df_test1["funder"].fillna("other", inplace=True)

df_test1["scheme_management"].fillna("other", inplace=True)

df_test1["installer"].fillna("other", inplace=True)

df_test1.isna().sum()
df_test1.head()

df_test1.shape
#AUTOMATE ALL THE THINGS!!!

def MrClean(df):

    df_t= df

    df_t['gps_height'].replace(0.0, np.nan, inplace=True)

    df_t['population'].replace(0.0, np.nan, inplace=True)

    df_t['amount_tsh'].replace(0.0, np.nan, inplace=True)

    df_t['gps_height'].fillna(df_t.groupby(['region', 'district_code'])['gps_height'].transform('mean'), inplace=True)

    df_t['gps_height'].fillna(df_t.groupby(['region'])['gps_height'].transform('mean'), inplace=True)

    df_t['gps_height'].fillna(df_t['gps_height'].mean(), inplace=True)

    df_t['population'].fillna(df_t.groupby(['region', 'district_code'])['population'].transform('median'), inplace=True)

    df_t['population'].fillna(df_t.groupby(['region'])['population'].transform('median'), inplace=True)

    df_t['population'].fillna(df_t['population'].median(), inplace=True)

    df_t['amount_tsh'].fillna(df_t.groupby(['region', 'district_code'])['amount_tsh'].transform('median'), inplace=True)

    df_t['amount_tsh'].fillna(df_t.groupby(['region'])['amount_tsh'].transform('median'), inplace=True)

    df_t['amount_tsh'].fillna(df_t['amount_tsh'].median(), inplace=True)

    features=['amount_tsh', 'gps_height', 'population']

    scaler = MinMaxScaler(feature_range=(0,20))

    df_t[features] = scaler.fit_transform(df_t[features])

    df_t['longitude'].replace(0.0, np.nan, inplace=True)

    df_t['latitude'].replace(0.0, np.nan, inplace=True)

    df_t['construction_year'].replace(0.0, np.nan, inplace=True)

    df_t['latitude'].fillna(df_t.groupby(['region', 'district_code'])['latitude'].transform('mean'), inplace=True)

    df_t['longitude'].fillna(df_t.groupby(['region', 'district_code'])['longitude'].transform('mean'), inplace=True)

    df_t['longitude'].fillna(df_t.groupby(['region'])['longitude'].transform('mean'), inplace=True)

    df_t['construction_year'].fillna(df_t.groupby(['region', 'district_code'])['construction_year'].transform('median'), inplace=True)

    df_t['construction_year'].fillna(df_t.groupby(['region'])['construction_year'].transform('median'), inplace=True)

    df_t['construction_year'].fillna(df_t.groupby(['district_code'])['construction_year'].transform('median'), inplace=True)

    df_t['construction_year'].fillna(df_t['construction_year'].median(), inplace=True)

    df_t['date_recorded'] = pd.to_datetime(df_t['date_recorded'])

    df_t['years_service'] = df_t.date_recorded.dt.year - df_t.construction_year

   

    garbage=['wpt_name','num_private','subvillage','region_code','recorded_by','management_group',

         'extraction_type_group','extraction_type_class','scheme_name','payment',

        'quality_group','quantity_group','source_type','source_class','waterpoint_type_group',

        'ward','public_meeting','permit','date_recorded','construction_year']

    df_t.drop(garbage,axis=1, inplace=True)

    df_t.waterpoint_type = df_t.waterpoint_type.str.lower()

    df_t.funder = df_t.funder.str.lower()

    df_t.basin = df_t.basin.str.lower()

    df_t.region = df_t.region.str.lower()

    df_t.source = df_t.source.str.lower()

    df_t.lga = df_t.lga.str.lower()

    df_t.management = df_t.management.str.lower()

    df_t.quantity = df_t.quantity.str.lower()

    df_t.water_quality = df_t.water_quality.str.lower()

    df_t.payment_type=df_t.payment_type.str.lower()

    df_t.extraction_type=df_t.extraction_type.str.lower()

    df_t["funder"].fillna("other", inplace=True)

    df_t["scheme_management"].fillna("other", inplace=True)

    df_t["installer"].fillna("other", inplace=True)

    return df_t
#Full is the df of both the train_features csv and train_labels merged 

full = pd.DataFrame.merge(df_label,df_feats)

full.shape

print(full.columns)
#Call out mrclean!

soclean =  MrClean(full)
soclean.head()

soclean.isna().sum()
yc = soclean['status_group']

Xc = soclean
Xc.head()
Xc.drop(columns = ['status_group'], axis = 1, inplace = True)
Xc.columns
#split this ish 

X_train, X_test, y_train, y_test = train_test_split(Xc, yc,test_size = .2, random_state=42)
X_train.head()
# TREES!!!

cleanpipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True),StandardScaler(),

                        tree.DecisionTreeClassifier(random_state=42) )

cleanpipe.fit(X_train, y_train)

preds = cleanpipe.predict(X_test)

accuracy_score(y_test,preds)
preddi = cleanpipe.predict(df_test1)
preddi.shape
full = pd.DataFrame.merge(df_label,df_feats)

full.shape

print(full.columns)

soclean =  MrClean(full)

train = soclean

test = df_test1
train.shape,test.shape
target = train.pop('status_group')

train['train']=1

test['train']=0
combo = pd.concat([train, test])

combo.info()
combo['funder'] = pd.factorize(combo['funder'])[0]

combo['installer'] = pd.factorize(combo['installer'])[0]

combo['scheme_management'] = pd.factorize(combo['scheme_management'])[0]

combo['extraction_type'] = pd.factorize(combo['extraction_type'])[0]

combo['management'] = pd.factorize(combo['management'])[0]

combo['payment_type'] = pd.factorize(combo['payment_type'])[0]

combo['water_quality'] = pd.factorize(combo['water_quality'])[0]

combo['quantity'] = pd.factorize(combo['quantity'])[0]

combo['source'] = pd.factorize(combo['source'])[0]

combo['waterpoint_type'] = pd.factorize(combo['waterpoint_type'])[0]

combo['basin'] = pd.factorize(combo['basin'])[0]

combo['region'] = pd.factorize(combo['region'])[0]

combo['lga'] = pd.factorize(combo['lga'])[0]

combo['district_code'] = pd.factorize(combo['district_code'])[0]

combo['years_service'] = pd.factorize(combo['years_service'])[0]

combo.head()
train_df = combo[combo["train"] == 1]

test_df = combo[combo["train"] == 0]

train_df.drop(["train"], axis=1, inplace=True)

train_df.drop(['id'],axis=1, inplace=True)

test_df.drop(["train"], axis=1, inplace=True)
X = train_df

y = target
X.shape,y.shape
y.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

model_rfc = RandomForestClassifier(n_estimators=1000, n_jobs = -1)
score = cross_val_score(model_rfc, X, y, cv=3, n_jobs = -1)
score.mean()
X_test=test_df

X_test.shape
model_rfc.fit(X,y)
X.info()

importances = model_rfc.feature_importances_

importances
X_test.shape, X.shape
a=X_test['id']

X_test.drop(['id'],axis=1, inplace=True)

y_pred = model_rfc.predict(X_test)
y_pred
a.head()
y_pred.shape,a.shape
y_pred=pd.DataFrame(y_pred)

y_pred['id']=a

y_pred.columns=['status_group','id']

y_pred=y_pred[['id','status_group']]
y_pred.head()
from xgboost import XGBClassifier

modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 

                      num_class = 4, maximize = False, eval_metric = 'merror', eta = .2,

                      max_depth = 14, colsample_bytree = .4)
#print(cross_val_score(modelxgb, X, y, cv=3,n_jobs = -1))

modelxgb.fit(X,y)
y_preds = modelxgb.predict(X_test)
y_preds=pd.DataFrame(y_preds)

y_preds['id']=a

y_preds.columns=['status_group','id']

y_preds=y_preds[['id','status_group']]
y_preds.shape
y_preds.head()
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1000)
scores = (cross_val_score(knn, X, y, cv=3,n_jobs = -1))

scores.mean()
cl = tree.DecisionTreeClassifier(random_state=42)

cl.fit(X,y)
y_predcl = cl.predict(X_test)
y_predcl=pd.DataFrame(y_predcl)

y_predcl['id']=a

y_predcl.columns=['status_group','id']

y_predcl=y_predcl[['id','status_group']]
y_predcl.head()
y_predcl.shape
log = LogisticRegression(solver ='saga',n_jobs=-1, multi_class = 'auto',C=1.0)
print(cross_val_score(log, X, y, cv=3,n_jobs = -1))

log.fit(X,y)
y_predlog =log.predict(X_test)
y_predlog=pd.DataFrame(y_predlog)

y_predlog['id']=a

y_predlog.columns=['status_group','id']

y_predlog=y_predlog[['id','status_group']]
y_predlog.head()
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score
clf = ExtraTreesClassifier(n_estimators=500, max_depth=None,

                           min_samples_split=10, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)
scores.mean()
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,

                             min_samples_split=8, random_state=0 , 

                             n_jobs=-1)

scores = cross_val_score(clf, X, y, cv=5)

scores.mean()                               
clf.fit(X,y)
pred_rfc =clf.predict(X_test)
pred_rfc=pd.DataFrame(pred_rfc)

pred_rfc ['id']=a

pred_rfc .columns=['status_group','id']

pred_rfc = pred_rfc[['id','status_group']]
pred_rfc.head()
from xgboost import XGBClassifier

modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 

                      num_class = 3, maximize = False, eval_metric = 'merror', eta = .1,

                      max_depth = 14, colsample_bytree = .4)
score = (cross_val_score(modelxgb, X, y, cv=5,n_jobs = -1))

score.mean()
modelxgb.fit(X,y)
predict = modelxgb.predict(X_test)
predict=pd.DataFrame(predict)

predict ['id']=a

predict .columns=['status_group','id']

predict = predict[['id','status_group']]

predict.head()