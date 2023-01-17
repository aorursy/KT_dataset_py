# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

import gc

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=DeprecationWarning)

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/lending-club-loan-data/loan.csv",low_memory="False")
df.head()
df.shape
df.columns
df_description = pd.read_excel('../input/lending-club-loan-data/LCDataDictionary.xlsx').dropna()

df_description.style.set_properties(subset=['Description'], **{'width': '1000px'})
def null_values(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        return mis_val_table_ren_columns
miss_values = null_values(df)

miss_values.head(20)
target_list = [1 if i=='Default' else 0 for i in df['loan_status']]



df['TARGET'] = target_list

df['TARGET'].value_counts()
df.drop('loan_status',axis=1,inplace=True)
# Number of each type of column

df.dtypes.value_counts().sort_values().plot(kind='barh')

plt.title('Number of columns distributed by Data Types',fontsize=20)

plt.xlabel('Number of columns',fontsize=15)

plt.ylabel('Data type',fontsize=15)
#Let us see how many categorical data do the columns having 'object' data types contain:

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

df['emp_length'].head(3)
df['emp_length'].fillna(value=0,inplace=True)



df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)



df['emp_length'].value_counts().sort_values().plot(kind='barh',figsize=(18,8))

plt.title('Number of loans distributed by Employment Years',fontsize=20)

plt.xlabel('Number of loans',fontsize=15)

plt.ylabel('Years worked',fontsize=15);
fig = plt.figure(figsize=(12,6))

sns.violinplot(x="TARGET",y="loan_amnt",data=df, hue="pymnt_plan", split=True)

plt.title("Payment plan - Loan Amount", fontsize=20)

plt.xlabel("TARGET", fontsize=15)

plt.ylabel("Loan Amount", fontsize=15);
temp = [i for i in df.count()<2260668 *0.30]

df.drop(df.columns[temp],axis=1,inplace=True)
corr = df.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', corr.tail(10))

print('\nMost Negative Correlations:\n', corr.head(10))
df.corr()['dti'].sort_values().tail(6)
fig = plt.figure(figsize=(22,6))

sns.kdeplot(df.loc[df['TARGET'] == 1, 'int_rate'], label = 'target = 1')

sns.kdeplot(df.loc[df['TARGET'] == 0, 'int_rate'], label = 'target = 0');

plt.xlabel('Interest Rate (%)',fontsize=15)

plt.ylabel('Density',fontsize=15)

plt.title('Distribution of Interest Rate',fontsize=20);
#The density of interest rates follow kind of a Gaussian distribution with more density on interest rates between 12%-18%.
fig = plt.figure(figsize=(12,6))

sns.violinplot(x="TARGET",y="loan_amnt",data=df, hue="term", split=True,color='pink')

plt.title("Term - Loan Amount", fontsize=20)

plt.xlabel("TARGET", fontsize=15)

plt.ylabel("Loan Amount", fontsize=15);
fig = plt.figure(figsize=(12,6))

sns.violinplot(x="TARGET",y="loan_amnt",data=df, hue="application_type", split=True,color='green')

plt.title("Application Type - Loan Amount", fontsize=20)

plt.xlabel("TARGET", fontsize=15)

plt.ylabel("Loan Amount", fontsize=15);
df['application_type'].value_counts()
#Violin-plot of TARGET classes with distribution of interest rate differentiated by the loan grades.

fig = plt.figure(figsize=(18,8))

sns.violinplot(x="TARGET",y="int_rate",data=df, hue="grade")

plt.title("Grade - Interest Rate", fontsize=20)

plt.xlabel("TARGET", fontsize=15)

plt.ylabel("Interest Rate", fontsize=15);
#Let us also check the correlation of annual income with loan amount taken.

df.corr()['annual_inc'].sort_values().tail(10)
#Defaulters based on State

fig = plt.figure(figsize=(18,10))

df[df['TARGET']==1].groupby('addr_state')['TARGET'].count().sort_values().plot(kind='barh')

plt.ylabel('State',fontsize=15)

plt.xlabel('Number of loans',fontsize=15)

plt.title('Number of defaulted loans per state',fontsize=20);
#Non Defaulters based on state

fig = plt.figure(figsize=(18,10))

df[df['TARGET']==0].groupby('addr_state')['TARGET'].count().sort_values().plot(kind='barh')

plt.ylabel('State')

plt.xlabel('Number of loans')

plt.title('Number of not-defaulted loans per state');
df['emp_title'].value_counts().head()
df.drop(['emp_title','title','zip_code'],axis=1,inplace=True)
df.shape
df.columns
df['issue_d']= pd.to_datetime(df['issue_d']).apply(lambda x: int(x.strftime('%Y')))

df['last_pymnt_d']= pd.to_datetime(df['last_pymnt_d'].fillna('2016-01-01')).apply(lambda x: int(x.strftime('%m')))

df['last_credit_pull_d']= pd.to_datetime(df['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))

df['earliest_cr_line']= pd.to_datetime(df['earliest_cr_line'].fillna('2001-08-01')).apply(lambda x: int(x.strftime('%m')))

df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'].fillna(value = '2016-02-01')).apply(lambda x:int(x.strftime("%Y")))
df['issue_d']= pd.to_datetime(df['issue_d']).apply(lambda x: int(x.strftime('%Y')))

df['last_pymnt_d']= pd.to_datetime(df['last_pymnt_d'].fillna('2016-01-01')).apply(lambda x: int(x.strftime('%m')))

df['last_credit_pull_d']= pd.to_datetime(df['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))

df['earliest_cr_line']= pd.to_datetime(df['earliest_cr_line'].fillna('2001-08-01')).apply(lambda x: int(x.strftime('%m')))

df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'].fillna(value = '2016-02-01')).apply(lambda x:int(x.strftime("%Y")))
from sklearn import preprocessing

count = 0



for col in df:

    if df[col].dtype == 'object':

        if len(list(df[col].unique())) <= 2:     

            le = preprocessing.LabelEncoder()

            df[col] = le.fit_transform(df[col])

            count += 1

            print (col)

            

print('%d columns were label encoded.' % count)
df = pd.get_dummies(df)

print(df.shape)
df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(df['mths_since_last_delinq'].median())
df.dropna(inplace=True)

df.count().sort_values().head(3)
df['TARGET'].value_counts()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Creating a classification report function,



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        print("Train Result:\n")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        

    elif train==False:

        print("Test Result:\n")        

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test)))) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('TARGET',axis=1),df['TARGET'],test_size=0.15,random_state=101)
#Standardizing features by removing the mean and scaling to unit variance



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test=sc.transform(X_test)
#Oversampling only the training set using Synthetic Minority Oversampling Technique (SMOTE)

import sklearn; print("Scikit-Learn", sklearn.__version__)

from imblearn.pipeline import make_pipeline as imb_make_pipeline

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)

x_train_r, y_train_r = sm.fit_sample(X_train, y_train)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(C = 0.0001,random_state=21)



log_reg.fit(x_train_r, y_train_r)
print_score(log_reg, x_train_r, y_train_r, X_test, y_test, train=False)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=40, random_state=21)

clf_rf.fit(x_train_r, y_train_r)
print_score(clf_rf, x_train_r, y_train_r, X_test, y_test, train=False)
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

from lightgbm import LGBMClassifier
def kfold_lightgbm(train_df, num_folds, stratified = False):

    print("Starting LightGBM. Train shape: {}".format(train_df.shape))

    

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)



    oof_preds = np.zeros(train_df.shape[0])



    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET']]

    

    # Splitting the training set into folds for Cross Validation

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]



        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(

            nthread=4,

            n_estimators=10000,

            learning_rate=0.02,

            num_leaves=32,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.04,

            reg_lambda=0.073,

            min_split_gain=0.0222415,

            min_child_weight=40,

            silent=-1,

            verbose=-1,

            )



        # Fitting the model and evaluating by AUC

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        print_score(clf, train_x, train_y, valid_x, valid_y, train=False)

        # Dataframe holding the different features and their importance

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        

        # Freeing up memory

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    display_importances(feature_importance_df)

    return feature_importance_df
def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(15, 12))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances.png')
feat_importance = kfold_lightgbm(df, num_folds= 3, stratified= False)