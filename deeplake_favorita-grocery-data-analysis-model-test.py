import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn

import gc; gc.enable()

from sklearn import preprocessing



from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV



dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

input = {

    'train'  : pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),

    'test'   : pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),

    'items'  : pd.read_csv('../input/items.csv'),

    'stores' : pd.read_csv('../input/stores.csv'),

    'txns'   : pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'holevts': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oil'    : pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }

input['train'].head()
test = input['test']

#P0) test.csv - predict for the period 2017.08.16 ~ 2017.08.20 

# => hence pick only for AUG data from train.csv, concerning the seasoning impact etc.

train = input['train'][(input['train']['date'].dt.month == 8) & (input['train']['date'].dt.day > 15)]

#P1) Negative values of unit_sales represent returns of that particular item

# => in prediction, returns means same as zero_sales - so convert to 0                       

unit_sales = train['unit_sales'].values

unit_sales[unit_sales < 0.] = 0.

#check histogram and see if better to log-transform

seaborn.distplot(train['unit_sales']);
train['unit_sales'] = np.log1p(unit_sales)

seaborn.distplot(np.log1p(unit_sales));
def proc_object_data(df):

    col = [c for c in df.columns if df[c].dtype == 'object']

    df[col] = df[col].apply(preprocessing.LabelEncoder().fit_transform)

    return df



#P5) stores - city, state, type, and cluster. cluster is a grouping of similar stores.

input['stores'] = proc_object_data(input['stores'])

train = pd.merge(train, input['stores'], how='left', on=['store_nbr'])

test = pd.merge(test, input['stores'], how='left', on=['store_nbr'])



#P6) items - family, class, and perishable.

#P7) Items marked as perishable have a score weight of 1.25; otherwise, the weight is 1.0.

input['items'] = proc_object_data(input['items'])

train = pd.merge(train, input['items'], how='left', on=['item_nbr'])

test = pd.merge(test, input['items'], how='left', on=['item_nbr'])



#P9) Daily oil price. Includes values during both the train and test data timeframe. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)

train = pd.merge(train, input['oil'], how='left', on=['date'])

test = pd.merge(test, input['oil'], how='left', on=['date'])



#P8) The count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe.

# => consider that transaction volume may not have relation with sales qty for each product - so ignore.

train.head(n=10)
#P10) A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. 

# => filter out : transferred == 'TRUE'

holevts = input['holevts']

holevts = holevts[(holevts.transferred != 'TRUE') & (holevts.transferred != 'True')]

# Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). 

# These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.

# => filter out : type = 'Work Day'

holevts = holevts[holevts.type != 'Work Day']

#Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday)

#type (Holiday, Transfer, Bridge, Additional) => Holiday // type (Event) => Event

holevts['on_hol'] = holevts['type'].map({"Holiday":"Holiday", "Transfer":"Holiday", "Bridge":"Holiday", "Additional":"Holiday"})

holevts['on_evt'] = holevts['type'].map({"Event":"Event"})

col = [c for c in holevts if c in ['date', 'locale_name','on_hol','on_evt']]

holevts_L = holevts[holevts.locale == 'Local'][col].rename(columns={'locale_name':'city'})

holevts_R = holevts[holevts.locale == 'Regional'][col].rename(columns={'locale_name':'state'})

holevts_N = holevts[holevts.locale == 'National'][col]



# Actually our test data is only for 2017.08.16~20, at which there's no holiday - hene it won't impact this case. 

# But still proceed to prepare factors (on_hol, on_evt) as these might be one of key factor in general.

train = pd.merge(train, holevts_L, how='left', on=['date','city'])

train = pd.merge(train, holevts_R, how='left', on=['date','state'])

train = pd.merge(train, holevts_N, how='left', on=['date'])

test = pd.merge(test, holevts_L, how='left', on=['date','city'])

test = pd.merge(test, holevts_R, how='left', on=['date','state'])

test = pd.merge(test, holevts_N, how='left', on=['date'])

train.head(n=10)
data = pd.concat([train,test],ignore_index=True)



def proc_cvt_data(df):

    df['year'] = df['date'].dt.year

    df['month'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df['date'] = df['date'].dt.dayofweek    

    df['wage'] = df['day'].map({15:1, 31:1})

    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})

    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})

    df['on_hol'] = np.where(df[["on_hol_x","on_hol_y","on_hol"]].apply(lambda x: x.str.contains('Holiday')).any(1), 1,0)

    df['on_evt'] = np.where(df[["on_evt_x","on_evt_y","on_evt"]].apply(lambda x: x.str.contains('Event')).any(1), 1,0)

    df = df.drop(["on_hol_x","on_hol_y","on_evt_x","on_evt_y","locale_name"], axis=1)

    df = df.fillna(-1)

    return df

data = proc_cvt_data(data)



train = data[data.id < 125497040]

test = data[data.id >= 125497040]

labels = train["unit_sales"]

ids = test["id"]

train.head(n=20)
del input['train']; gc.collect();

del input['test']; gc.collect();

del input['items']; gc.collect();

del input['stores']; gc.collect();

del input['txns']; gc.collect();

del input['holevts']; gc.collect();

del input['oil']; gc.collect();



# Count the number of NaNs each column has. Display columns having more than 30% NAN => skip it as we've done preprocessing

#DROP_NAN_PCT = 0.3

#nans=pd.isnull(data).sum()

#nans[nans > data.shape[0] * DROP_NAN_PCT]
# CHECK_CORRELATION_AND_KEY_FACTORS

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 12))

k = 30  #number of variables for heatmap

cols = corrmat.nlargest(k, 'unit_sales')['unit_sales'].index

cm = np.corrcoef(train[cols].values.T)

seaborn.set(font_scale=1)

hm = seaborn.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# heatmap check shows no strong relationships between the factors so skip the PCA analysis

#data = data.drop("unit_sales", 1)

#pca = PCA(whiten=True).fit(data)

#np.cumsum(pca.explained_variance_ratio_)

#shows the variance is explained by N factors 
def apply_models(train,labels):

    results={}

    def train_get_score(clf):        

        cv = KFold(n_splits=2,shuffle=True,random_state=45)

        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=make_scorer(r2_score))

        return [r2_val_score.mean()]



    results["Linear"]=train_get_score(linear_model.LinearRegression())

    results["Ridge"]=train_get_score(linear_model.Ridge())

    #results["Bayesian Ridge"]=train_get_score(linear_model.BayesianRidge())

    results["Hubber"]=train_get_score(linear_model.HuberRegressor())

    results["Lasso"]=train_get_score(linear_model.Lasso(alpha=1e-4))

    results["RandomForest"]=train_get_score(RandomForestRegressor())

    #results["SVM RBF"]=train_get_score(svm.SVR())

    #results["SVM Linear"]=train_get_score(svm.SVR(kernel="linear"))

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["R Square Score"] 

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,1])

    return results



apply_models(train,labels)
def get_predict_model(clf, train, labels):

    cv = KFold(n_splits=2,shuffle=True,random_state=45)

    parameters = {'alpha': [1000,100,10],'epsilon' : [1.2,1.25,1.50],'tol' : [1e-10]}

    grid_obj = GridSearchCV(clf, parameters, cv=cv,scoring=make_scorer(r2_score))

    predict_model = grid_obj.fit(train, labels).best_estimator_

    #predict_model = clf.fit(train,labels)

    return predict_model



predict_model = get_predict_model(linear_model.HuberRegressor(),train,labels)
predictions = (np.exp(predict_model.predict(test)) - 1) # restoring unit values 

sub = pd.DataFrame({"id": ids, "unit_sales": predictions})

print(sub)
sub.to_csv("favorita_submission.csv", index=False)