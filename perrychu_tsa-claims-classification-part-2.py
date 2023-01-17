#Data / vizualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Feature engineering
from datetime import datetime
from collections import defaultdict

#Modeling
##Utilities
from sklearn.model_selection import KFold, train_test_split
import sklearn.metrics as skm
import pickle
##Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
df = pd.read_csv("../input/tsa-claims-classification-part-1/tsa_claims_clean.csv", low_memory=False)

print(len(df))
df.head(3)
df["Date_Received"] = pd.to_datetime(df.Date_Received,format="%Y-%m-%d")
df["Month_Received"] = df.Date_Received.dt.month
df["DayMonth_Received"] = df.Date_Received.dt.day
df["DayYear_Received"] = df.Date_Received.dt.dayofyear

df["Incident_Date"] = pd.to_datetime(df.Incident_Date,format="%Y-%m-%d")
df["Incident_Month"] = df.Incident_Date.dt.month
df["Incident_DayMonth"] = df.Incident_Date.dt.day
df["Incident_DayYear"] = df.Incident_Date.dt.dayofyear

df["Report_Delay"] = (df.Date_Received - df.Incident_Date).dt.days

date_var = ["Report_Delay",
        "Month_Received","DayYear_Received","DayMonth_Received",
        "Incident_Month","Incident_DayYear","Incident_DayMonth"]
#Frequency rank
#Default is max rank + 1
def get_count_rank(var_column):
    val_count = var_column.value_counts(dropna=False)
    conversion_dict = defaultdict(lambda: len(val_count)+1, zip(val_count.index, range(len(val_count.values))))
        
    return conversion_dict
    
#Frequency
#Default is zero
def get_count(var_column):
    val_count = var_column.value_counts(dropna=False)
    conversion_dict = defaultdict(lambda: 0, zip(val_count.index, val_count.values))
    
    return conversion_dict

#Apply conversion from text to numeric
def apply_conversion(var_column, conversion_dict):

    return var_column.map(lambda x: conversion_dict[x])

def create_numeric(train_df, test_df, conversion_func, columns, postfix="_"):
    '''
    Applies conversion function over the training set to train and test sets.
    Inputs:
        train_df: training data (input to conversion func, then applied)
        test_df: test data (conversion applied)
        conversion_func: returns a map assigning numeric value to each text category
        columns: columns to apply
        postfix: postfix to add to created columns (may overwrite, e.g. on empty string)
        
    '''
    maps = {}
    new_columns = []
    
    for name in columns:
        new_name = name+postfix
        new_columns.append(new_name)
        
        maps[name] = conversion_func(train_df[name])
        
        train_df[new_name] = apply_conversion(train_df[name], maps[name])
        test_df[new_name] = apply_conversion(test_df[name],maps[name])

    text_count_var = [x + "_Count" for x in string_categories]
    
    return train_df, test_df, new_columns

df, df_holdout = train_test_split(df, test_size = .2, shuffle = True, random_state = 1)

string_categories = ["Claim_Type","Claim_Site","Airport_Code_Group","Airline_Name"]

#Category complaint counts
df, df_holdout, text_count_var = create_numeric(df, df_holdout, get_count, string_categories, "_Count")

#Category complaint rank
df, df_holdout, text_rank_var = create_numeric(df, df_holdout, get_count_rank, string_categories, "_Rank")


print("Training / Validation:", len(df), "\nTest:", len(df_holdout))
df_holdout.head(5)
def print_scores(scores_array, ylabel_strings):
    ''' Prints a table with headings for output of sklearn.metrics.precision_recall_fscore_support
        Inputs: scores_array -> np.array of scores from skm.
                ylabel_strings -> the target labels
    '''
    #Each row is a score in the output, transpose to get features across rows
    array = np.transpose(scores_array) 
    macro_avg = np.average(array,axis=0)
    labels = sorted(ylabel_strings)
    
    max_len = str(np.max([len(s) for s in ylabel_strings]))
        
    print(("\n{:>"+max_len+"} {:>10s} {:>10s} {:>10s} {:>10s}").format("","Precision","Recall","F1","Support"))
    
    for i in range(len(labels)):
        print(("{:>"+max_len+"} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.0f}")
              .format(labels[i],array[i][0],array[i][1],array[i][2],array[i][3]))
    
    print(("{:>"+max_len+"} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.0f}")
          .format("Avg/Tot",macro_avg[0],macro_avg[1],macro_avg[2],macro_avg[3]))
    
def validation_loop(model,X,Y,k=5,rand_state=1):
    ''' Runs k-fold validation loop for input model, X, Y. Prints classification accuracy 
             and the following per-label metrics: precision, recall, f1, support.
        Inputs: 
                ylabel_strings -> the target labels
    '''
    test_accs, test_scores = [], []
    train_accs, train_scores = [], []
    
    i=1

    for train_ind, test_ind in KFold(k,shuffle=True,random_state=rand_state).split(X,Y):
        #print("Starting {} of {} folds".format(i,k))

        model.fit(X[train_ind],Y[train_ind])
        
        #Test metrics
        pred = model.predict(X[test_ind])
        acc = skm.accuracy_score(Y[test_ind],pred)
        test_accs.append(acc)
        score = skm.precision_recall_fscore_support(Y[test_ind],pred)
        test_scores.append(score)
        
        #Train metrics
        pred = model.predict(X[train_ind])
        acc = skm.accuracy_score(Y[train_ind],pred)
        train_accs.append(acc)
        score = skm.precision_recall_fscore_support(Y[train_ind],pred)
        train_scores.append(score)
        
        i+=1
    
    print("\nAvg. Train Metrics")
    print ("Accuracy: {:.5f}".format(np.average(train_accs)))
    print_scores(np.average(train_scores,axis=0),np.unique(Y))
    
    print("\nAvg. Validation Metrics")
    print ("Accuracy: {:.5f}".format(np.average(test_accs)))
    print_scores(np.average(test_scores,axis=0),np.unique(Y))
    
    
features = ["Claim_Value"]
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators=250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
dummies_df = pd.get_dummies(df[["Claim_Type","Claim_Site","Airport_Code_Group","Airline_Name"]],prefix=["Type","Site","Airport","Airline"])

features = ["Claim_Value"] + list(dummies_df.columns)
target = "Status"

model_df=df[["Status","Claim_Value"]].join(dummies_df).dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"] + text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"] + text_rank_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"] + text_count_var + text_rank_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var+text_count_var
target = "Status"
csv_df=df[[target]+features].dropna()

csv_df.to_csv("tsa_model_features.csv",index=False)
features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = model_df[features].reset_index(drop=True)
Y = model_df[target].reset_index(drop=True)

model = xgb.XGBClassifier(n_estimators = 30000,
                          learning_rate = .2,
                          max_depth = 4,
                          objective = "multi:softmax",
                          subsample=1,
                          min_child_weight=1,
                          colsample_bytree=.8,
                          random_state = 1,
                          n_jobs = -1
                         )

test_accs, test_scores = [], []
train_accs, train_scores = [], []

logloss = []
ntrees = []

i=1
k=5
rand_state = 1

for train_ind, test_ind in KFold(k,shuffle=True,random_state=rand_state).split(X,Y):
    #print("Starting {} of {} folds".format(i,k))

    eval_set=[(X.iloc[train_ind],Y.iloc[train_ind]),(X.iloc[test_ind],Y.iloc[test_ind])] 
    fit_model = model.fit( 
                    X.iloc[train_ind], Y.iloc[train_ind], 
                    eval_set=eval_set,
                    eval_metric='mlogloss',
                    early_stopping_rounds=50,
                    verbose=False
                   )
    
    logloss.append(model.best_score)
    ntrees.append(model.best_ntree_limit)
    
    #Test metrics
    pred = model.predict(X.iloc[test_ind],ntree_limit=model.best_ntree_limit)
    acc = skm.accuracy_score(Y.iloc[test_ind],pred)
    test_accs.append(acc)
    score = skm.precision_recall_fscore_support(Y.iloc[test_ind],pred)
    test_scores.append(score)
    #print(acc)
    #print(skm.classification_report(Y[test_ind],pred))

    #Train metrics
    pred = model.predict(X.iloc[train_ind],ntree_limit=model.best_ntree_limit)
    acc = skm.accuracy_score(Y.iloc[train_ind],pred)
    train_accs.append(acc)
    score = skm.precision_recall_fscore_support(Y.iloc[train_ind],pred)
    train_scores.append(score)

    i+=1

print("\nAvg. Train Metrics")
print ("Accuracy: {:.5f}".format(np.average(train_accs)))
print_scores(np.average(train_scores,axis=0),np.unique(Y))

print("\nAvg. Validation Metrics")
print ("Accuracy: {:.5f}".format(np.average(test_accs)))
print_scores(np.average(test_scores,axis=0),np.unique(Y))

print("\nLogloss:", np.average(logloss), "Std Dev:", np.std(logloss))
print("Best number of trees", ntrees)
features = ["Claim_Value"]+date_var+text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = LogisticRegression(C=100)

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var+text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = GaussianNB()

validation_loop(model,X,Y,rand_state=1)
features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = model_df[features].reset_index(drop=True)
Y = model_df[target].reset_index(drop=True)

model_df_holdout = df_holdout[[target]+features].dropna()
X_holdout = model_df_holdout[features]
Y_holdout = model_df_holdout[target]

model =  xgb.XGBClassifier(max_depth = 6,
                           subsamples = 1,
                           min_child_weight=6,
                           colsample_bytree=0.6,
                           n_estimators=107,
                           learning_rate=.1,
                           objective = "multi:softmax",
                           random_state = 1,
                           n_jobs = -1)

fit_model = model.fit(X, Y)

print("Training")
pred = model.predict(X)
print(skm.accuracy_score(Y,pred))
print(skm.classification_report(Y, pred))

print("\nHoldout")
pred = model.predict(X_holdout)
print(skm.accuracy_score(Y_holdout,pred))
print(skm.classification_report(Y_holdout, pred))


xgb.plot_importance(fit_model, importance_type="gain");
