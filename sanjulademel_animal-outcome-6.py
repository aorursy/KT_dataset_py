# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
seed = 1024
np.random.seed(seed)

train = pd.read_csv('../input/train.csv',parse_dates=['DateTime']).fillna('')
test = pd.read_csv('../input/test.csv',parse_dates=['DateTime']).fillna('')

def has_name(name):
    if type(name) is str:
        return 1
    else:
        return 0
def convert_AgeuponOutcome_to_days(df):
    result = {}
    for k in df['AgeuponOutcome'].unique():
        if type(k) != type(""):
            result[k] = -1
        else:
            v1, v2 = k.split()
            if v2 in ["year", "years"]:
                result[k] = int(v1) * 365
            elif v2 in ["month", "months"]:
                result[k] = int(v1) * 30
            elif v2 in ["week", "weeks"]:
                result[k] = int(v1)*30
            elif v2 in ["day", "days"]:
                result[k] = int(v1) 
                
    df['_AgeuponOutcome'] = df['AgeuponOutcome'].map(result).astype(float)
    df = df.drop('AgeuponOutcome', axis = 1)
                
    return df
def fix_date_time(df):
    def extract_field(_df, start, stop):
        return _df['DateTime'].map(lambda dt: int(dt[start:stop]))
    df['Year'] = extract_field(df,0,4)
    df['Month'] = extract_field(df,5,7)
    df['Day'] = extract_field(df,8,10)
    df['Hour'] = extract_field(df,11,13)
    df['Minute'] = extract_field(df,14,16)
    
    return df.drop(['DateTime'], axis = 1)
def Bread_cat(df):
    result = {}
    for k in df['Breed'].unique():
        if type(k) != type(""):
            result[k] = "Unknown"
        else:
            result[k] = "Pure"
            v = k.split(" ")
            for w in v:
                if w == "Mix":
                    result[k] = "Mix"         
                
    df['_Breed'] = df['Breed'].map(result).astype(str)
    df = df.drop('Breed', axis = 1)
    return df
def Sexupon_cat(df):
    result1 = {}
    result2 = {}
    for k in df['SexuponOutcome'].unique():
        if type(k) != type(""):
            result1[k] = "Unknown"
            result2[k] = "Unknown"
        elif  k == "Unknown":
            result1[k] = "Unknown"
            result2[k] = "Unknown"          
        else:
            v1,v2 = k.split(" ")
            if v1 == "":
                result1[k] = "Unknown"
            else:
                    result1[k] = v1
            if v2 == "":
                result2[k] = "Unknown"
            else:
                result2[k] = v2
                
    df['_SexuponOutcome_1'] = df['SexuponOutcome'].map(result1).astype(str)
    df['_SexuponOutcome_2'] = df['SexuponOutcome'].map(result2).astype(str)
    df = df.drop('SexuponOutcome', axis = 1)
    return df
def colour_cat(df):
    numberofterms = {}
    BWO = {}
    x = 0
    y = 0
    result2 = {}
    for k in df['Color'].unique():
        if type(k) != type(""):
            x = 0
            y = 4
        elif  k == "Unknown":
            x = 0
            y = 4
        else:
            x = 0
            y = 0
            v1 = k.split(" ")
            for v in v1:
                v2 = v.split("/")
                for colours in v2:
                    x = x + 1
                    if colours == "White":
                        y = y + 1
                    if colours == "Black":
                        y = y + 2
        numberofterms[k] = x
        if y == 0 :
            BWO[k] = "Other"
        elif y == 1:
            BWO[k] = "White"
        elif y == 2:
            BWO[k] = "Black"
        elif y == 3:
            BWO[k] = "Black and White"
        else:          
            BWO[k] = "Unkown"    
                
    df['_Color_1'] = df['Color'].map(numberofterms).astype(int)
    df['_Color_2'] = df['Color'].map(BWO).astype(str)
    df = df.drop('Color', axis = 1)
    return df
train.drop("OutcomeSubtype", axis = 1 , inplace = True)

train["Has name"] = train["Name"].apply(has_name)
test["Has name"] = test["Name"].apply(has_name)

train.drop("Name", axis = 1 , inplace = True)
test.drop("Name",axis = 1, inplace = True)


train = Bread_cat(train)
test = Bread_cat(test)


train = colour_cat(train)
test = colour_cat(test)



train['SexPrefix'] = train['SexuponOutcome'].apply(lambda x: x.split(' ')[0])
test['SexPrefix'] = test['SexuponOutcome'].apply(lambda x: x.split(' ')[0])


train['DateTime_dayofweek'] = train['DateTime'].dt.dayofweek
train['DateTime_dayofyear'] = train['DateTime'].dt.dayofyear
train['DateTime_days_in_month'] = train['DateTime'].dt.days_in_month


test['DateTime_dayofweek'] = test['DateTime'].dt.dayofweek
test['DateTime_dayofyear'] = test['DateTime'].dt.dayofyear
test['DateTime_days_in_month'] = test['DateTime'].dt.days_in_month

data_all = pd.concat([train,test])
data_all = data_all.drop('AnimalID',axis=1)
# data_all = data_all.drop('Name',axis=1)
data_all = data_all.drop('ID',axis=1)
X = []
X_t = []
for c in data_all.columns:
    le = LabelEncoder()
    le.fit(data_all[c].values)
    X.append(le.transform(train[c].values))
    X_t.append(le.transform(test[c].values))

X = np.vstack(X).T
X_t = np.vstack(X_t).T

def make_mf_classifier(X ,y, clf, X_test,n_folds=2, n_round=5):
    n = X.shape[0]
    len_y = len(np.unique(y))
    mf_tr = np.zeros((X.shape[0],len_y))

    mf_te = np.zeros((X_test.shape[0],len_y))

    for i in range(n_round):
        skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            clf.fit(X_tr, y_tr)

            mf_tr[ind_te] += clf.predict_proba(X_te)
            mf_te += clf.predict_proba(X_test)*0.5
            y_pred = clf.predict_proba(X_te)
            score = log_loss(y_te, y_pred)
            print('pred[{}],score[{}]'.format(i,score))
    return (mf_tr / n_round, mf_te / n_round)

skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]
    
    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

print(X_train.shape,X_test.shape)



xgboost = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate = 0.03, 
        max_depth=6, 
        subsample=0.7, 
        colsample_bytree = 0.7, 
        # gamma = 0.7, 
        # max_delta_step=0.1, 
        reg_lambda = 4, 
        # min_child_weight=50, 
        seed = seed, 
        ) 

xgboost.fit(
    X_train,
    y_train,
    eval_metric='mlogloss',
    eval_set=[(X_train,y_train),(X_test,y_test)],
    early_stopping_rounds=100,
    )
y_preds = xgboost.predict_proba(X_test)


res = xgboost.predict_proba(X_t)

submission = pd.DataFrame()
submission["ID"] = np.arange(res.shape[0])+1
submission["Adoption"]= res[:,0]
submission["Died"]= res[:,1]
submission["Euthanasia"]= res[:,2]
submission["Return_to_owner"]= res[:,3]
submission["Transfer"]= res[:,4]

submission.to_csv("sub10.csv",index=False)