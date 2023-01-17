import os

import numpy as np

import pandas as pd

from datetime import date



from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve

from sklearn.preprocessing import MinMaxScaler



DATA_ROOT = "../input/"
dfoff = pd.read_csv(os.path.join(DATA_ROOT,'train_offline.csv'))

dftest = pd.read_csv(os.path.join(DATA_ROOT,'test_offline.csv'))

dftest = dftest[~dftest.Coupon_id.isna()]

dftest.reset_index(drop=True, inplace=True)

print(dfoff.shape)

print(dftest.shape)

dfoff.head(20)
## Creat target label 

"""

According to the definition, 

1) buy with coupon within (include) 15 days ==> 1

2) buy with coupon but out of 15 days ==> 0

3) buy without coupon ==> -1 (we don't care)

"""

def label(row):

    if np.isnan(row['Date_received']):

        return -1

    if not np.isnan(row['Date']):

        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')

        if td <= pd.Timedelta(15, 'D'):

            return 1

    return 0



dfoff["label"] = dfoff.apply(label, axis=1)

dfoff["label"].value_counts()
# Generate features - weekday acquired coupon

def getWeekday(row):

    if (np.isnan(row)) or (row==-1):

        return row

    else:

        return pd.to_datetime(row, format = "%Y%m%d").dayofweek+1 # add one to make it from 0~6 -> 1~7



dfoff['weekday'] = dfoff['Date_received'].apply(getWeekday)

dftest['weekday'] = dftest['Date_received'].apply(getWeekday)



# weekday_type (weekend = 1)

dfoff['weekday_type'] = dfoff['weekday'].astype('str').apply(lambda x : 1 if x in [6,7] else 0 ) # apply to trainset

dftest['weekday_type'] = dftest['weekday'].astype('str').apply(lambda x : 1 if x in [6,7] else 0 ) # apply to testset
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]

print(weekdaycols)



tmpdf = pd.get_dummies(dfoff['weekday'].replace(-1, np.nan))

tmpdf.columns = weekdaycols

dfoff[weekdaycols] = tmpdf



tmpdf = pd.get_dummies(dftest['weekday'].replace(-1, np.nan))

tmpdf.columns = weekdaycols

dftest[weekdaycols] = tmpdf
# Generate features - coupon discount and distance

def getDiscountType(row):

    if row == 'null':

        return 'null'

    elif ':' in row:

        return 1

    else:

        return 0



def convertRate(row):

    """Convert discount to rate"""

    if row == 'null':

        return 1.0

    elif ':' in row:

        rows = row.split(':')

        return 1.0 - float(rows[1])/float(rows[0])

    else:

        return float(row)



def getDiscountMan(row):

    if ':' in row:

        rows = row.split(':')

        return int(rows[0])

    else:

        return 0



def getDiscountJian(row):

    if ':' in row:

        rows = row.split(':')

        return int(rows[1])

    else:

        return 0



def processData(df):

    

    # convert discunt_rate

    df['discount_rate'] = df['Discount_rate'].astype('str').apply(convertRate)

    df['discount_man'] = df['Discount_rate'].astype('str').apply(getDiscountMan)

    df['discount_jian'] = df['Discount_rate'].astype('str').apply(getDiscountJian)

    df['discount_type'] = df['Discount_rate'].astype('str').apply(getDiscountType)

    

    # convert distance

    df.loc[df.Distance.isna(), "Distance"] = 99

    return df



dfoff = processData(dfoff)

dftest = processData(dftest)
## Naive model

def split_train_valid(row, date_cut="20160416"):

    is_train = True if pd.to_datetime(row, format="%Y%m%d") < pd.to_datetime(date_cut, format="%Y%m%d") else False

    return is_train

    

df = dfoff[dfoff['label'] != -1].copy()

df["is_train"] = df["Date_received"].apply(split_train_valid)

train = df[df["is_train"]]

valid = df[~df["is_train"]]

train.reset_index(drop=True, inplace=True)

valid.reset_index(drop=True, inplace=True)

print("Train size: {}, #positive: {}".format(len(train), train["label"].sum()))

print("Valid size: {}, #positive: {}".format(len(valid), valid["label"].sum()))
original_feature = ['discount_rate',

                    'discount_type',

                    'discount_man', 

                    'discount_jian',

                    'Distance', 

                    'weekday', 

                    'weekday_type'] + weekdaycols

print(len(original_feature),original_feature)
predictors = original_feature

print(predictors)



def check_model(data, predictors):

    

    classifier = lambda: SGDClassifier(

        loss='log', 

        penalty='elasticnet', 

        fit_intercept=True, 

        max_iter=100, 

        shuffle=True, 

        n_jobs=1,

        class_weight=None)



    model = Pipeline(steps=[

        ('ss', StandardScaler()),

        ('en', classifier())

    ])



    parameters = {

        'en__alpha': [ 0.001, 0.01, 0.1],

        'en__l1_ratio': [ 0.001, 0.01, 0.1]

    }



    folder = StratifiedKFold(n_splits=3, shuffle=True)

    

    grid_search = GridSearchCV(

        model, 

        parameters, 

        cv=folder, 

        n_jobs=-1, 

        verbose=1)

    grid_search = grid_search.fit(data[predictors], 

                                  data['label'])

    

    return grid_search
model = check_model(train, predictors)
y_valid_pred = model.predict_proba(valid[predictors])

valid1 = valid.copy()

valid1['pred_prob'] = y_valid_pred[:, 1]
from sklearn.metrics import roc_auc_score, accuracy_score

auc_score = roc_auc_score(y_true=valid.label, y_score=y_valid_pred[:,1])

acc = accuracy_score(y_true=valid.label, y_pred=y_valid_pred.argmax(axis=1))

print("Validation AUC: {:.3f}, Accuracy: {:.3f}".format(auc_score, acc))
targetset = dftest.copy()

print(targetset.shape)

targetset = targetset[~targetset.Coupon_id.isna()]

targetset.reset_index(drop=True, inplace=True)

testset = targetset[predictors].copy()



y_test_pred = model.predict_proba(testset[predictors])

test1 = testset.copy()

test1['pred_prob'] = y_test_pred[:, 1]

print(test1.shape)
output = pd.concat((targetset[["User_id", "Coupon_id", "Date_received"]], test1["pred_prob"]), axis=1)

print(output.shape)



output.loc[:, "User_id"] = output["User_id"].apply(lambda x:str(int(x)))

output.loc[:, "Coupon_id"] = output["Coupon_id"].apply(lambda x:str(int(x)))

output.loc[:, "Date_received"] = output["Date_received"].apply(lambda x:str(int(x)))

output["uid"] = output[["User_id", "Coupon_id", "Date_received"]].apply(lambda x: '_'.join(x.values), axis=1)

output.reset_index(drop=True, inplace=True)
### NOTE: YOUR SUBMITION FILE SHOULD HAVE COLUMN NAME: uid, label

out = output.groupby("uid", as_index=False).mean()

out = out[["uid", "pred_prob"]]

out.columns = ["uid", "label"]

# out.to_csv("baseline_example.csv", header=["uid", "label"], index=False) # submission format

out.head()