from IPython.display import display

from itertools import combinations

import matplotlib.pyplot as plt

import numpy as np

import operator

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook, tqdm



from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, roc_curve, auc, classification_report

from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

from sklearn.utils.multiclass import type_of_target



from tqdm import tqdm_notebook, tqdm

from IPython.display import display

from itertools import combinations



import xgboost as xgb



np.set_printoptions(precision=2)

pd.set_option("max_columns", 99)
df = pd.read_csv('../input/DfTRoadSafety_Accidents_2014.csv')

df.head()
# remove index

df.drop(["Accident_Index"], axis=1, inplace=True)
# rename target columns

df.rename(columns={"Did_Police_Officer_Attend_Scene_of_Accident": "target"}, 

                   inplace=True)

df.target.value_counts()
yes = (df[df['target']==1].shape[0] / df.shape[0]) * 100

no = (df[df['target']==2].shape[0] / df.shape[0]) * 100



print("There are {:.2f}% yes answers encoded 1. There are {:.2f}% no answers encoded 2.".format(yes, no))
numerical_features = set(["Longitude", "Latitude", "Number_of_Vehicles"])
categorical_features = set(df.columns) - numerical_features - set(["target"])
df.shape
# This simply shows how many missing values there are per column in percentage

missing_val = (df.isin([-1]).sum() / df.shape[0]) * 100
missing_col = list((missing_val[missing_val != 0]).index)

missing_col
mode = df[missing_col].mode().T

mode
for col in missing_col:

    val = mode.loc[col].values[0]

    df[col] = df[col].replace(to_replace=-1, value=val)
df.select_dtypes("object").head()
geo_features = ["Police_Force", "LSOA_of_Accident_Location", "Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude", 

        "Local_Authority_(District)", "Local_Authority_(Highway)", "Urban_or_Rural_Area", "1st_Road_Number", "2nd_Road_Number"]
df.drop(["LSOA_of_Accident_Location", "Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude", 

        "Local_Authority_(District)", "Local_Authority_(Highway)", "1st_Road_Number", "2nd_Road_Number"],

        axis=1, inplace=True)
numerical_features = list(set(df.columns).intersection(numerical_features))

numerical_features
sorted(df[numerical_features].skew().items(), key=operator.itemgetter(1), reverse=True)
def smooth(x):

    if x > 0:

        return np.log(x)

    if x <= 0:

        return np.log(1-x)
df["Number_of_Vehicles"] = df["Number_of_Vehicles"].apply(lambda x: smooth(x))
pd.crosstab(df["Number_of_Casualties"], df.target, normalize="columns")
def create_date_features(df, column):

    df[column] = pd.to_datetime(df[column])

    df['day_{}'.format(column)] = df[column].dt.day

    df['week_{}'.format(column)] = df[column].dt.week

    df['month_{}'.format(column)] = df[column].dt.month

    df['year_{}'.format(column)] = df[column].dt.year

    df['weekday_{}'.format(column)] = df[column].dt.weekday

    #df['numeric_{}'.format(column)] = df[column].astype(np.int64) * 1e-9

    return df
def create_time_features(df, column):

    df[column] = pd.to_datetime(df[column])

    df['hour_{}'.format(column)] = df[column].dt.hour

    df['minute_{}'.format(column)] = df[column].dt.minute

    return df
date_col = "Date"

time_col = "Time"



df[date_col] = create_date_features(df, date_col)

df.drop(date_col, axis=1, inplace=True)



df[time_col] = create_time_features(df, time_col)

df.drop(time_col, axis=1, inplace=True)
df.head()
def is_week_end(val):

    if val == 6 or val==7:

        return 1

    else:

        return 0
def is_night(val):

    if val in [21, 22, 23] + list(range(0, 7)):

        return 1

    else:

        return 0
df["is_week_end"] = df["Day_of_Week"].apply(is_week_end)
df["is_night"] = df["Day_of_Week"].apply(is_night)
categorical_features = set(df.columns).intersection(categorical_features)
categorical_features
cardinality_dict = {}

for col in categorical_features:

    cardinality_dict[col] = len((df[col].unique()))

        

top_cardinality = sorted(cardinality_dict.items(), key=operator.itemgetter(1), reverse=True)



top_cardinality[:10]
top_card_col = [f_ for f_, card in top_cardinality]
# value counts

for var in top_card_col:

    mapping_vc = df[var].value_counts()

    df['vc_{}'.format(var)] = df[var].map(mapping_vc)

    # remove ordinal features

    df.drop([var], axis=1, inplace=True)
X = df.drop(["target"], axis=1, inplace=False)

y = df["target"].astype(str)
X.head()
# remove single value column

single_val_col = [c for c in df.columns if len(set(df[c])) == 1]

X.drop(single_val_col, axis=1, inplace=True)
type_of_target(y)
# We stratified our data with balanced fold and we shuffle the observations

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
cv_score = []

i=1

for train_index, test_index in sss.split(X, y):

    print('{} of KFold {}'.format(i, sss.n_splits))

    #print('Train {} indexes, Test {} indexes'.format(train_index, test_index))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # model

    clf = RandomForestClassifier(n_estimators=10, max_depth=None,

                             min_samples_split=2, random_state=0)

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)

    score = log_loss(y_test, y_pred_proba)

    cv_score.append(score) 

    print("RF Validation LogLoss: {}".format(score))

    print('Confusion matrix\n', confusion_matrix(y_test, y_pred))

    i+=1
print(classification_report(y_test, y_pred, target_names=["yes", "no"]))
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')
X_top_rf_features = X[list(feat_importances.nlargest(25).index)]
cv_score = []

i=1

for train_index, test_index in sss.split(X_top_rf_features, y):

    print('{} of KFold {}'.format(i, sss.n_splits))

    X_train, X_test = X_top_rf_features.iloc[train_index], X_top_rf_features.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # model

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=0.8)

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    y_pred_proba = xgb_model.predict_proba(X_test)

    score = log_loss(y_test, y_pred_proba)

    cv_score.append(score) 

    print("XGBoost Validation LogLoss: {}".format(score))

    print(confusion_matrix(y_test, y_pred))

    xgb.plot_importance(xgb_model)

    i+=1

        
cv_score = []

i=1

for train_index, test_index in sss.split(X, y):

    print('{} of KFold {}'.format(i, sss.n_splits))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # model

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=0.8)

    # fit model on all training data

    xgb_model.fit(X_train, y_train)

    # make predictions for test data and evaluate

    y_pred = xgb_model.predict(X_test)

    y_pred_proba = xgb_model.predict_proba(X_test)

    score = log_loss(y_test, y_pred_proba)

    cv_score.append(score) 

    print("XGBoost Validation LogLoss: {}".format(score))

    print(confusion_matrix(y_test, y_pred))

    xgb.plot_importance(xgb_model)

    

    # Fit model using each importance as a threshold

    thresholds = np.sort(xgb_model.feature_importances_)

    

    for thresh in sorted(list(set(thresholds))):

        # select features using threshold

        selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)

        select_X_train = selection.transform(X_train)

        # train model

        selection_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=0.8)

        selection_model.fit(select_X_train, y_train)

        # eval model

        select_X_test = selection.transform(X_test)

        y_pred_proba = selection_model.predict_proba(select_X_test)

        score = log_loss(y_test, y_pred_proba)

        print("Thresh=%.3f, n=%d, LogLoss: %.4f" % (thresh, select_X_train.shape[1], score))



    i+=1

        
print('Confusion matrix\n', confusion_matrix(y_test, y_pred))

print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))  