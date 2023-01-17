import numpy as np 

import pandas as pd 

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import xgboost

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings('ignore')
train= pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')

test= pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')

pet_id = test['pet_id']
print("Train Shape: ",train.shape)

print("Test Shape: ", test.shape)

print(train.columns)

print(test.columns)
train.head()
train.dtypes
test.head()
test.dtypes
print('Breed Category :')

print(train['breed_category'].value_counts())

print()

print('Pet Category :')

print(train['pet_category'].value_counts())
train.isnull().sum()
test.isnull().sum()
train['pet_id_new'] = train['pet_id'].str[:7]

test['pet_id_new'] = test['pet_id'].str[:7]
print(train.pet_id_new.nunique())

print(train.pet_id_new.value_counts())
train['issue_date']= pd.to_datetime(train['issue_date'])

train['listing_date']= pd.to_datetime(train['listing_date'])

train.loc[train['listing_date'] < train['issue_date']]
train.drop([1504, 5301],inplace=True)
test['issue_date']= pd.to_datetime(test['issue_date'])

test['listing_date']= pd.to_datetime(test['listing_date'])

test.loc[test['listing_date'] < test['issue_date']]
train['issue_year'] = train['issue_date'].dt.year

train['issue_month'] = train['issue_date'].dt.month

train['issue_day'] = train['issue_date'].dt.day



train['listing_year'] = train['listing_date'].dt.year

train['listing_month'] = train['listing_date'].dt.month

train['listing_day'] = train['listing_date'].dt.day

train['listing_hour'] = train['listing_date'].dt.hour

train['listing_minute'] = train['listing_date'].dt.minute



test['issue_year'] = test['issue_date'].dt.year

test['issue_month'] = test['issue_date'].dt.month

test['issue_day'] = test['issue_date'].dt.day



test['listing_year'] = test['listing_date'].dt.year

test['listing_month'] = test['listing_date'].dt.month

test['listing_day'] = test['listing_date'].dt.day

test['listing_hour'] = test['listing_date'].dt.hour

test['listing_minute'] = test['listing_date'].dt.minute
train = train.fillna(-99)

test = test.fillna(-99)

print(train['condition'].value_counts())

print()

print(test['condition'].value_counts())
train.groupby(['condition','pet_category']).size()
train.groupby(['condition', 'breed_category']).size()
cd = pd.DataFrame(train['condition'])

cd['condition99'] = cd[cd['condition']==-99]

cd['condition99'] = cd['condition99'].fillna(0)

cd.condition99[cd.condition99 == -99 ] = 1

m1 = pd.DataFrame(cd['condition99'])



tt = pd.DataFrame(test['condition'])

tt['condition99'] = tt[tt['condition']==-99]

tt['condition99'] = tt['condition99'].fillna(0)

tt.condition99[tt.condition99 == -99 ] = 1

t1 = pd.DataFrame(tt['condition99'])



cd = pd.DataFrame(train['condition'])

cd['condition00'] = cd[cd['condition']==0]

cd.condition00[cd.condition00 == 0 ] = 1

cd['condition00'] = cd['condition00'].fillna(0)

m2 = pd.DataFrame(cd['condition00'])



tt = pd.DataFrame(test['condition'])

tt['condition00'] = tt[tt['condition']==0]

tt.condition00[tt.condition00 == 0 ] = 1

tt['condition00'] = tt['condition00'].fillna(0)

t2 = pd.DataFrame(tt['condition00'])



cd = pd.DataFrame(train['condition'])

cd['condition1'] = cd[cd['condition']==1]

cd['condition1'] = cd['condition1'].fillna(0)

m3 = pd.DataFrame(cd['condition1'])



tt = pd.DataFrame(test['condition'])

tt['condition1'] = tt[tt['condition']==1]

tt['condition1'] = tt['condition1'].fillna(0)

t3 = pd.DataFrame(tt['condition1'])
train = pd.concat([train,m1,m2,m3], axis=1, sort=False)



test = pd.concat([test,t1,t2,t3], axis=1, sort=False)
test.condition[test.condition == 1 ] = 0

test.condition[test.condition == 0 ] = 0

test.condition[test.condition == -99 ] = 0

test.condition[test.condition == 2 ] = 1



train.condition[train.condition == 1 ] = 0

train.condition[train.condition == 0 ] = 0

train.condition[train.condition == -99 ] = 0

train.condition[train.condition == 2 ] = 1
train.rename(columns = {'condition':'condition2'}, inplace = True)
print(len(train[train['length(m)'] == 0]))

print(len(test[test['length(m)']==0]))
train['length(cm)'] = train['length(m)'].apply(lambda x: x*100)

test['length(cm)'] = test['length(m)'].apply(lambda x: x*100)



train.drop('length(m)', axis=1, inplace=True)

test.drop('length(m)', axis=1, inplace=True)
# replace all 0 length with mean of lengths

val = train['length(cm)'].mean()

train['length(cm)'] = train['length(cm)'].replace(to_replace=0, value=val)

test['length(cm)'] = test['length(cm)'].replace(to_replace=0, value=val)
train[['length(cm)','height(cm)']].describe()
train['ratio_len_height'] = train['length(cm)']/train['height(cm)']

test['ratio_len_height'] = test['length(cm)']/test['height(cm)']
train['X1'].value_counts()
test['X1'].value_counts()
train['X2'].value_counts()
test['X2'].value_counts()
train.drop(['pet_id','issue_date','listing_date'], axis = 1,inplace=True) 



test.rename(columns = {'condition':'condition2'}, inplace = True) 



test.drop(['pet_id','issue_date','listing_date'], axis = 1,inplace=True) 
print(train.columns.shape)

print(test.columns.shape)
print(train.shape)

print(test.shape)
df = train.append(test) 

df.shape
one_hot1 = pd.get_dummies(df['color_type'])

one_hot1.shape
one_hot2 = pd.get_dummies(df['pet_id_new'])

one_hot2.shape
df = pd.concat([df,one_hot1,one_hot2], axis=1)
df = df.drop(['color_type','pet_id_new'],axis = 1)
df.shape
final_data=df.iloc[:18832]
final_test=df.iloc[18832:]
final_test = final_test.drop(['breed_category','pet_category'],axis = 1)
Y_breed = final_data['breed_category']

Y_pet = final_data['pet_category']

X = final_data.drop(['breed_category','pet_category'],axis = 1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y_breed, test_size=0.3, random_state=0)

X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y_pet, test_size=0.3, random_state=0)
# Balancing Dataset

sm = SMOTE(random_state=2)

X1_train_res, y1_train_res = sm.fit_sample(X1_train, y1_train)

X2_train_res, y2_train_res = sm.fit_sample(X2_train, y2_train)
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)

clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

clf3 = GaussianNB()

clf4 = xgboost.XGBClassifier()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('xgb',clf4)], voting='hard')
eclf1 = eclf.fit(X1_train_res, y1_train_res)

y1_pred= eclf1.predict(X1_test)

breed_pred = eclf1.predict(final_test)

f1_score(y1_test, y1_pred, average='weighted')
eclf2 = eclf.fit(X2_train_res, y2_train_res)

y2_pred= eclf2.predict(X2_test)

pet_pred = eclf2.predict(final_test)

f1_score(y2_test, y2_pred, average='weighted')
sub = pd.DataFrame([pet_id,breed_pred,pet_pred])

sub = sub.transpose().set_index('pet_id')

sub.rename(columns = {'Unnamed 0':'breed_category','Unnamed 1':'pet_category'}, inplace = True) 

sub.to_csv('submission.csv')