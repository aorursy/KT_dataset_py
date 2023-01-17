

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test["loss"] = 0
print(train.shape)

print(test.shape)
full = train.append(test)
list(full.columns)
cats_cols = [name for name in list(full.columns) if "cat" in name]

cats_cols
data_cat = full[cats_cols]

data_cat.head()
from  sklearn.preprocessing import LabelEncoder

def encode_cats(cat_array):

    encoding = LabelEncoder()

    return(encoding.fit_transform(cat_array))

    

    

data_cat = data_cat.apply(encode_cats)
full[cats_cols] = data_cat

full.head()
count_cols = [cont for cont in full.columns if "cont" in cont]

count_data = full[count_cols]

count_data.head()
count_data.describe()
def new_cont(cont_feature):

    cont_squared = cont_feature**2

    cont_root = np.sqrt(cont_feature)

    return cont_squared,cont_root



columns = count_data.columns

for column in columns:

    col_sqr,col_root = new_cont(count_data[column])

    count_data[column + "_sqr"] = col_sqr

    count_data[column + "_root"] = col_root

   

count_data.head()
full[count_data.columns] = count_data
full.head()
train = full.iloc[:len(train)]

train.shape
test = full.iloc[len(train):len(full)]

test.shape
test = test.drop("loss",axis=1)

ids = test.id

test = test.drop("id",axis=1)

test.shape
train_loss = train.loss

train =train.drop("loss",axis=1)

train = train.drop("id",axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train,train_loss, test_size=0.3, random_state=42)
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=200)

xgb.fit(X_train, y_train)

train_acc = xgb.score(X_train,y_train)

print("Train Accuracy")

print(train_acc.mean())



print("Cv Accuracy")

scores = cross_val_score(xgb,X_test,y_test,scoring = 'r2')

scores.mean()
feature_df = pd.DataFrame({"Feature":list(train.columns),"Importance":xgb.feature_importances_})

feature_df=feature_df.sort_values(by="Importance",ascending=False)

feature_df.head()
feature_df = feature_df.loc[feature_df.Importance > 0]

feature_df.shape
train2 = train[feature_df.Feature]

train2.shape
X_train, X_test, y_train, y_test = train_test_split(train2,train_loss, test_size=0.3, random_state=42)
xgb = XGBRegressor(n_estimators=200)

xgb.fit(X_train, y_train)

train_acc = xgb.score(X_train,y_train)

print("Train Accuracy")

print(train_acc.mean())



print("Cv Accuracy")

scores = cross_val_score(xgb,X_test,y_test,scoring = 'r2')

scores.mean()
len(xgb.feature_importances_)
feature_df = pd.DataFrame({"Feature":list(train2.columns),"Importance":xgb.feature_importances_})

feature_df=feature_df.sort_values(by="Importance",ascending=False)

feature_df.head()
feature_df.Importance.describe()
feature_df = feature_df.loc[feature_df.Importance >= 0.01]

feature_df.shape
train3 = train[feature_df.Feature]

train3.shape

X_train, X_test, y_train, y_test = train_test_split(train3,train_loss, test_size=0.3, random_state=42)
xgb = XGBRegressor(n_estimators=1000)

xgb.fit(X_train, y_train)

train_acc = xgb.score(X_train,y_train)

print("Train Accuracy")

print(train_acc.mean())



print("Cv Accuracy")

scores = cross_val_score(xgb,X_test,y_test,scoring = 'r2')

scores.mean()
feature_df = pd.DataFrame({"Feature":list(train3.columns),"Importance":xgb.feature_importances_})

feature_df=feature_df.sort_values(by="Importance",ascending=False)

feature_df
test = test[train3.columns]
list(feature_df.Feature)
test.columns
predictions = xgb.predict(test)
results = {"id":list(ids),"loss":predictions}

result_df = pd.DataFrame(results)

result_df.head()
result_df.to_csv('submission.csv',header=True, index_label='id')
from sklearn.ensemble import RandomForestRegressor



RFR = RandomForestRegressor()

RFR.fit(X_train,y_train)

train_acc = RFR.score(X_train,y_train)

print("Train Accuracy")

print(train_acc.mean())

scores = cross_val_score(RFR,X_test,y_test,scoring = 'r2')

print("Cv Accuracy")

print(scores.mean())
from sklearn.ensemble import GradientBoostingRegressor



GB = GradientBoostingRegressor()

GB.fit(X_train,y_train)

train_acc = GB.score(X_train,y_train)

print("Train Accuracy")

print(train_acc.mean())

scores = cross_val_score(GB,X_test,y_test,scoring = 'r2')

print("Cv Accuracy")

print(scores.mean())