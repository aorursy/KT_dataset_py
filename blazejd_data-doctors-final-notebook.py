import numpy as np

import pandas as pd

import scipy as sp

from sklearn import preprocessing

from sklearn.model_selection import *

from sklearn.metrics import roc_auc_score

import xgboost as xgb

from sklearn.preprocessing import *

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import *

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import mlxtend

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.neural_network import MLPClassifier



%matplotlib inline

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



SEED = 7

NFOLDS = 5
df = pd.read_csv('/kaggle/input/flumadness2020/flu_train.csv')
df_test = pd.read_csv('/kaggle/input/flumadness2020/flu_test.csv')
test_ID = df_test['ID']
label_dist = df.flu.value_counts()/df.shape[0]; label_dist
sns.barplot(y=label_dist.values,x=label_dist.index)

plt.title("Target class distribution",fontsize=15)

plt.xlabel("Classes")

plt.ylabel("Distribution in %")

plt.show()
cat_cols = list(df.select_dtypes(include=['object']).columns)

cont_cols = list(df.select_dtypes(exclude=['object']).columns)

cont_cols.remove('flu')
df.columns
df.describe()
f, ax = plt.subplots(figsize= [15,12])

sns.heatmap(df[cont_cols+['flu']].corr(), annot=True, fmt=".2f", ax=ax, 

            cbar_kws={'label': 'Correlation Coefficient'}, cmap='viridis')

ax.set_title("Correlation Matrix", fontsize=15)

plt.show()
bp_cols = [col for col in df.columns if 'BP' in col and 'Ave' not in col]; bp_cols
cor_cols = ['Weight','Height','HHIncome']+bp_cols; cor_cols
missing_perc = df.isna().sum()/df.shape[0]

missing_perc.sort_values(ascending=False)
plt.xlabel('Numbers of missing data')

plt.ylabel('Numbers of columns')

plt.title('Distribution of missing data')

print("Total missing values: ", (df.isnull().sum().sum()))

print("Mean missing values per column : " , df.isnull().sum().sum()/len(df.columns))

df.isnull().sum().hist(bins=20);
X = ['Missing Values','Poor','Fair','Good', 'Very Good', 'Excellent']

Y = [df["HealthGen"].isnull().sum(),(df["HealthGen"] == "Poor").sum(),(df["HealthGen"] == "Fair").sum(),(df["HealthGen"] == "Good").sum(),

    (df["HealthGen"] == "Vgood").sum(), (df["HealthGen"] == "Excellent").sum()]

plt.bar(X, Y)

plt.ylabel('Numbers of people')

plt.title('Health Ditribution of the People')

plt.show()
# Check the Percentages of people that have the flu and are in poor health:



perc_a = ((df.HealthGen == "Poor") & (df.flu == 1)).sum()/df["HealthGen"].notnull().sum()*100

perc_b = ((df.HealthGen == "Fair") & (df.flu == 1)).sum()/df["HealthGen"].notnull().sum()*100

perc_c = ((df.HealthGen == "Good") & (df.flu == 1)).sum()/df["HealthGen"].notnull().sum()*100

perc_d = ((df.HealthGen == "Vgood") & (df.flu == 1)).sum()/df["HealthGen"].notnull().sum()*100

perc_e = ((df.HealthGen == "Excellent") & (df.flu == 1)).sum()/df["HealthGen"].notnull().sum()*100





X = ['Poor','Fair','Good', 'Very Good', 'Excellent']

Y = [perc_a, perc_b, perc_c, perc_d,perc_e]

plt.bar(X, Y)

plt.ylabel('Percentage of people having the flu')

plt.title('Health Influence of the people on the flu')

plt.show()
pd.read_csv("../input/flu-results/2results_nans_regression.csv",sep=";",index_col=0,keep_default_na=False)
pd.read_csv("../input/flu-results/3results_nans_mean.csv",sep=";",index_col=0,keep_default_na=False)
# continuous variables - impute mean

df[cont_cols] = df[cont_cols].fillna(df[cont_cols].mean())

df_test[cont_cols] = df_test[cont_cols].fillna(df_test[cont_cols].mean())



# categorical variables - make a separate class

df[cat_cols] = df[cat_cols].fillna("Missing")

df_test[cat_cols] = df_test[cat_cols].fillna("Missing")
df.isna().any().any() or df_test.isna().any().any()
for col in cat_cols:

    print("Column {} unique_values {}".format(col, list(set(df[col].unique()) | set(df_test[col].unique()))))
ordered_dict = {

    'Education':['Missing','8th Grade','9 - 11th Grade','High School','Some College','College Grad'],

    'HHIncome':['Missing',' 0-4999',' 5000-9999','10000-14999','15000-19999','20000-24999','25000-34999', 

                '35000-44999', '45000-54999', '55000-64999','65000-74999','75000-99999','more 99999'],

    'BMICatUnder20yrs':['Missing','UnderWeight','NormWeight','OverWeight','Obese'],

    'BMI_WHO':['Missing','12.0_18.5','18.5_to_24.9','25.0_to_29.9','30.0_plus'],

    'HealthGen':['Missing','Poor','Fair','Good','Vgood','Excellent'],

    'LittleInterest':['Missing','None','Several','Most'],

    'Depressed':['Missing','None','Several','Most'],

    'TVHrsDay':['Missing', '0_hrs','0_to_1_hr','1_hr','2_hr','3_hr', '4_hr','More_4_hr'],

    'CompHrsDay':['Missing', '0_hrs','0_to_1_hr','1_hr','2_hr','3_hr', '4_hr','More_4_hr']

}
ordered_cols = [key for key in ordered_dict]
mapping = [ordered_dict[col] for col in ordered_cols]
oe = OrdinalEncoder(categories=mapping)

oe.fit(df[ordered_cols])

df[ordered_cols] = oe.transform(df[ordered_cols])

df_test[ordered_cols] = oe.transform(df_test[ordered_cols])
cat_cols = [col for col in cat_cols if col not in ordered_cols]

oe = OrdinalEncoder()

oe.fit(df[cat_cols])

df[cat_cols] = oe.transform(df[cat_cols])

df_test[cat_cols] = oe.transform(df_test[cat_cols])
pd.read_csv("../input/flu-results/8results_label_encoding.csv",sep=";",index_col=0,keep_default_na=False)
pd.read_csv("../input/flu-results/3results_nans_mean.csv",sep=";",index_col=0,keep_default_na=False)
# get columns with missing values in more than 80% examples

def get_too_many_missing(data):

    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.8]

    return many_null_cols



# get columns in which a single value is in more than 80% examples

def get_too_many_repeated(data):

    big_top_value_cols = [col for col in data.columns if data[col].value_counts(dropna=False, normalize=True).values[0] > 0.8]

    return big_top_value_cols



missing_cols = get_too_many_missing(df)

repeated_cols = get_too_many_repeated(df)
dropped_cols = cor_cols+missing_cols+repeated_cols+['ID']; dropped_cols
pd.read_csv("../input/flu-results/4results_manual_selection.csv",sep=";",index_col=0,keep_default_na=False)
pd.read_csv("../input/flu-results/5results_all_features.csv",sep=";",index_col=0,keep_default_na=False)
pd.read_csv("../input/flu-results/3results_nans_mean.csv",sep=";",index_col=0,keep_default_na=False)
X = df[['Race1',

  'Work',

  'Diabetes',

  'HealthGen',

  'DaysMentHlthBad',

  'LittleInterest',

  'Depressed',

  'SleepTrouble',

  'PhysActive',

  'CompHrsDayChild']]

y = df['flu']

df_test = df_test[['Race1',

  'Work',

  'Diabetes',

  'HealthGen',

  'DaysMentHlthBad',

  'LittleInterest',

  'Depressed',

  'SleepTrouble',

  'PhysActive',

  'CompHrsDayChild']]
pd.read_csv("../input/flu-results/xgb_results.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results.csv",index_col=0)
pd.read_csv("../input/flu-results/mlp_results.csv",index_col=0)
pd.read_csv("../input/flu-results/logistic_results.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results_only_scale_pos_weight.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results_best.csv",index_col=0)
weight_ratio = df.flu.value_counts()[0]/df.flu.value_counts()[1]
def fold_metrics(model, x_train, y_train, x_val, y_val, metrics_dict,debug=False): 

    

    fold_train_overall = model.score(x_train, y_train)

    fold_train_class_0 = model.score(x_train[y_train==0], y_train[y_train==0])

    fold_train_class_1 = model.score(x_train[y_train==1], y_train[y_train==1])

    fold_train_auc = roc_auc_score(y_train,model.predict_proba(x_train)[:,1])

    y_preds = model.predict(x_train)

    fold_train_conf = confusion_matrix(y_train,y_preds)

    fold_train_prec = precision_score(y_train,y_preds)

    fold_train_recall = recall_score(y_train,y_preds)

    



    fold_overall = model.score(x_val, y_val)

    fold_class_0 = model.score(x_val[y_val==0], y_val[y_val==0])

    fold_class_1 = model.score(x_val[y_val==1], y_val[y_val==1])

    fold_auc = roc_auc_score(y_val,model.predict_proba(x_val)[:,1])

    y_preds = model.predict(x_val)

    fold_conf = confusion_matrix(y_val,y_preds)

    fold_prec = precision_score(y_val,y_preds)

    fold_recall = recall_score(y_val,y_preds)



    

    metrics_dict['train']['overall'].append(fold_train_overall)

    metrics_dict['train']['class_0'].append(fold_train_class_0)

    metrics_dict['train']['class_1'].append(fold_train_class_1)

    metrics_dict['train']['auc'].append(fold_train_auc)

    metrics_dict['train']['conf'].append(fold_train_conf)

    metrics_dict['train']['prec'].append(fold_train_prec)

    metrics_dict['train']['rec'].append(fold_train_recall)

    

    

    metrics_dict['test']['overall'].append(fold_overall)

    metrics_dict['test']['class_0'].append(fold_class_0)

    metrics_dict['test']['class_1'].append(fold_class_1)

    metrics_dict['test']['auc'].append(fold_auc)

    metrics_dict['test']['conf'].append(fold_conf)

    metrics_dict['test']['prec'].append(fold_prec)

    metrics_dict['test']['rec'].append(fold_recall)

    if debug:

        print("Fold metrics")

        print('train\n',fold_train_conf)

        print('test\n',fold_conf)

        metrics = pd.DataFrame.from_dict({'train':[fold_train_overall,fold_train_class_0, fold_train_class_1,fold_train_auc,fold_train_prec,fold_train_recall],

                                         'test':[fold_overall,fold_class_0, fold_class_1,fold_auc,fold_prec,fold_recall]})

        metrics.index = ['overall','class_0','class_1','auroc','precision','recall']

        display(metrics)
def mean_fold_metrics(metrics_dict):

    

    train_overall = np.mean(metrics_dict['train']['overall'])

    train_class_0 = np.mean(metrics_dict['train']['class_0'])

    train_class_1 = np.mean(metrics_dict['train']['class_1'])

    train_auc = np.mean(metrics_dict['train']['auc'])

#   element-wise mean

    train_conf = np.mean(metrics_dict['train']['conf'],axis=0)

    train_prec = np.mean(metrics_dict['train']['prec'])

    train_recall = np.mean(metrics_dict['train']['rec'])



    overall = np.mean(metrics_dict['test']['overall'])

    class_0 = np.mean(metrics_dict['test']['class_0'])

    class_1 = np.mean(metrics_dict['test']['class_1'])

    auc = np.mean(metrics_dict['test']['auc'])

#   element-wise mean

    conf = np.mean(metrics_dict['test']['conf'],axis=0)

    prec = np.mean(metrics_dict['test']['prec'])

    recall = np.mean(metrics_dict['test']['rec'])

    metrics = pd.DataFrame.from_dict({'train':[train_overall,train_class_0, train_class_1,train_auc,train_prec,train_recall],

                                     'test':[overall,class_0, class_1,auc,prec,recall]})

    metrics.index = ['overall','class_0','class_1','auroc','precision','recall']

    

    print()

    print("Mean fold metrics")

    print('train\n',train_conf)

    print('test\n',conf)

    display(metrics)

    return metrics
def train_model(X, X_test,y, folds, params, model_type='lgb',n_jobs=-1, n_estimators=None, plot_feature_importance=True, verbose=500,early_stopping_rounds=None):

#     add n_estimators and early_stopping as arguments in lgbm

    print("Model type",model_type)

    metrics_dict = {'train':{'overall':[],'class_0':[],'class_1':[],'auc':[],'conf':[],'prec':[],'rec':[]},

                   'test':{'overall':[],'class_0':[],'class_1':[],'auc':[],'conf':[],'prec':[],'rec':[]}}

    result_dict = {}

    n_splits = folds.n_splits

    columns = X.columns

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), 1))

    # averaged predictions on train data(i think this should be "test" data)

    prediction = np.zeros((len(X_test), 1))

    feature_importance = pd.DataFrame()

    

    

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

        print('Fold nr {}'.format(fold_n))

        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type=='lgb':

            model = lgb.LGBMClassifier(**params, importance_type='gain')

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',

                    verbose=verbose)

            y_pred_valid = model.predict_proba(X_valid)[:,1]

    #         we want 0s and 1s for submission

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        elif model_type=='xgb':

            model = xgb.XGBClassifier(random_state=SEED,scale_pos_weight=weight_ratio)

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)[:,1]

            #         we want 0s and 1s for submission



            y_pred = model.predict(X_test)

        elif model_type=='logistic':

            model = LogisticRegression(C=0.000001)

            model.fit(X_train,y_train)

            y_pred_valid = model.predict_proba(X_valid)[:,1]

            y_pred = model.predict(X_test)

        elif model_type=='mlp':

            mlp = MLPClassifier()

            model.fit(X_train,y_train)

            y_pred_valid = model.predict_proba(X_valid)[:,1]

            y_pred = model.predict(X_test)

        else:

            raise Exception("Invalid model type")

                    

        fold_metrics(model,X_train,y_train,X_valid,y_valid,metrics_dict)

        oof[valid_index] = y_pred_valid.reshape(-1, 1)

        prediction += y_pred.reshape(-1, 1)

        if plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    

    prediction /= n_splits

    

    result_dict['oof'] = oof

    

    result_dict['metrics'] = mean_fold_metrics(metrics_dict)

    

    

    

    result_dict['prediction'] = prediction.flatten()

        

    

    if plot_feature_importance:

        feature_importance["importance"] /= n_splits

        best_features = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

            by="importance", ascending=False)[:50].reset_index(level=['feature'])





        plt.figure(figsize=(16, 12));

        sns.barplot(x="importance", y="feature", data=best_features);

        plt.title('{} Features (avg over folds)'.format(model_type.upper()));



        result_dict['feature_importance'] = feature_importance

        result_dict['top_columns'] = best_features['feature'].unique()

        

    return result_dict
params = {

        'max_depth':3,

        'min_child_weight':1,

        'scale_pos_weight':weight_ratio,

        'objective':'binary'

}
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
results = train_model(X,df_test,y,folds,params,'lgb')
submission = pd.DataFrame(columns=['ID','Prediction'])
submission['Prediction'] = results['prediction'].astype(int)
submission['ID'] = test_ID
submission[['ID','Prediction']].head()
submission.Prediction.value_counts()
import datetime 

  

now = datetime.datetime.now() 

now = now.strftime("%d_%m_%Y_%H_%M")

filename = 'submission{}.csv'.format(now)

submission[['ID','Prediction']].to_csv('../working/{}'.format(filename),index=False)

print(filename)
pd.read_csv("../input/flu-results/lgb_results.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results_only_scale_pos_weight.csv",index_col=0)
pd.read_csv("../input/flu-results/lgb_results_best.csv",index_col=0)
pd.read_csv("../input/flu-results/4results_manual_selection.csv",sep=";",index_col=0,keep_default_na=False)
pd.read_csv("../input/flu-results/3results_nans_mean.csv",sep=";",index_col=0,keep_default_na=False)