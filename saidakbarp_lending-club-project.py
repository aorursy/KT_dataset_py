import numpy as np

import pandas as pd

import os

import gc

import warnings

warnings.filterwarnings("ignore") # ignore python warnings of deprecation



df = pd.read_csv('../input/loan.csv', na_values=['#NAME?']) # '#NAME?' in the datafile will be converted to NaN
df.shape
df.head(5)
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

import plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



%matplotlib inline
plt.subplots(figsize=(20,10))

plt.title("The distribution of loan amounts by status").set_size(40)

sns.boxplot(x="loan_amnt", y="loan_status", data=df)
df['int_round'] = df['int_rate'].round(0).astype(int)



plt.figure(figsize = (10,8))



#Exploring the Int_rate

plt.subplot(211)

g = sns.distplot(np.log(df["int_rate"]))

g.set_xlabel("", fontsize=12)

g.set_ylabel("Distribuition", fontsize=12)

g.set_title("Int Rate Log distribuition", fontsize=20)



plt.subplot(212)

g1 = sns.countplot(x="int_round",data=df, 

                   palette="Set2")

g1.set_xlabel("Int Rate", fontsize=12)

g1.set_ylabel("Count", fontsize=12)

g1.set_title("Int Rate Normal Distribuition", fontsize=20)



plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)



plt.show()
print(df.loan_status.value_counts())



plt.figure(figsize = (12,14))



plt.subplot(311)

g = sns.countplot(x="loan_status", data=df)

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=15)

g.set_title("Loan Status Count", fontsize=20)



plt.subplot(312)

g1 = sns.boxplot(x="loan_status", y="total_acc", data=df)

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

g1.set_xlabel("", fontsize=12)

g1.set_ylabel("Total Acc", fontsize=15)

g1.set_title("Duration Count", fontsize=20)



plt.subplot(313)

g2 = sns.violinplot(x="loan_status", y="loan_amnt", data=df)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

g2.set_xlabel("Duration Distribuition", fontsize=15)

g2.set_ylabel("Count", fontsize=15)

g2.set_title("Loan Amount", fontsize=20)



plt.subplots_adjust(wspace = 0.2, hspace = 0.7,top = 0.9)



plt.show()
df['issue_month'], df['issue_year'] = df['issue_d'].str.split('-', 1).str

plt.figure(figsize=(12,8))

sns.barplot('issue_year', 'loan_amnt', data=df, palette='tab10')

plt.title('Issuance of Loans', fontsize=16)

plt.xlabel('Year', fontsize=14)

plt.ylabel('Average loan amount issued', fontsize=14)
f, ((ax1, ax2)) = plt.subplots(1, 2)

cmap = plt.cm.coolwarm



by_credit_score = df.groupby(['issue_year', 'grade']).loan_amnt.mean()

by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)

ax1.set_title('Loans issued by Credit Score', fontsize=14)

    

    

by_inc = df.groupby(['issue_year', 'grade']).int_rate.mean()

by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)

ax2.set_title('Interest Rates by Credit Score', fontsize=14)



ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size':12},

           ncol=7, mode="expand", borderaxespad=0.)
# drop the columns we created for data visualiztion
df.drop(['int_round','issue_year','issue_month'], axis=1, inplace=True)
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['loan_status',"term","grade",'home_ownership'], prefix = ['loan_status',"term","grade",'home_ownership'])



print(df_onehot.head())
df_correlations = df_onehot.corr()



trace = go.Heatmap(z=df_correlations.values,

                   x=df_correlations.columns,

                   y=df_correlations.columns,

                  colorscale=[[0.0, 'rgb(165,0,38)'], 

                              [0.1111111111111111, 'rgb(215,48,39)'], 

                              [0.2222222222222222, 'rgb(244,109,67)'], 

                              [0.3333333333333333, 'rgb(253,174,97)'], 

                              [0.4444444444444444, 'rgb(254,224,144)'], 

                              [0.5555555555555556, 'rgb(224,243,248)'], 

                              [0.6666666666666666, 'rgb(171,217,233)'], 

                              [0.7777777777777778, 'rgb(116,173,209)'], 

                              [0.8888888888888888, 'rgb(69,117,180)'], 

                              [1.0, 'rgb(49,54,149)']],

            colorbar = dict(

            title = 'Level of Correlation',

            titleside = 'top',

            tickmode = 'array',

            tickvals = [-0.52,0.2,0.95],

            ticktext = ['Negative Correlation','Low Correlation','Positive Correlation'],

            ticks = 'outside'

        )

                  )





layout = {"title": "Correlation Heatmap"}

data=[trace]



fig = dict(data=data, layout=layout)

iplot(fig, filename='labelled-heatmap')
df_onehot.corr()["loan_status_Current"].sort_values(ascending=False).head(15)
df_onehot.corr()["loan_status_Charged Off"].sort_values(ascending=False).head(15) 
df_onehot.corr()["loan_status_Fully Paid"].sort_values(ascending=False).head(15)
df_onehot.corr()["loan_status_Late (31-120 days)"].sort_values(ascending=False).head(15)
%reset -sf
import numpy as np

import pandas as pd

import os

import gc

gc.collect()
#del df, df_onehot

gc.collect()

df = pd.read_csv('../input/loan.csv', na_values=['#NAME?'], low_memory=False) # '#NAME?' in the datafile will be converted to NaN
df["loan_status"].value_counts()
# get the number of missing data points per column

missing_values_count = df.isnull().sum()

print(missing_values_count)
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

              " columns with missing values.")

        return mis_val_table_ren_columns
# Missing values statistics

miss_values = null_values(df)

miss_values.head(50)
# Drop irrelevant columns

df.drop(['id', 'member_id', 'emp_title', 'url', 'desc', 'zip_code', 'title'], axis=1, inplace=True)
df.drop(['dti_joint','annual_inc_joint','verification_status_joint','il_util','mths_since_rcnt_il','total_cu_tl',

         'inq_fi','all_util','max_bal_bc','open_rv_24m','open_rv_12m','total_bal_il','open_il_24m','total_bal_il',

         'open_il_24m','open_il_6m','open_acc_6m','open_il_12m','inq_last_12m','mths_since_last_record','mths_since_last_major_derog'],axis=1, inplace=True)
miss_values = null_values(df)

miss_values.head(20)
df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(df['mths_since_last_delinq'].median())


# Determining the loans that are bad from loan_status column



#bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", 

#            "Late (16-30 days)", "Late (31-120 days)"]





from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder





label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(df["loan_status"])

df['loan_status']= integer_encoded
unique, counts = np.unique(integer_encoded, return_counts=True)

print('Unique values:\n', np.asarray((unique, counts)).T)
a=label_encoder.inverse_transform(unique)

a
# maybe move it before define y

df['issue_d']= pd.to_datetime(df['issue_d']).apply(lambda x: int(x.strftime('%Y')))

df['last_pymnt_d']= pd.to_datetime(df['last_pymnt_d'].fillna('2016-01-01')).apply(lambda x: int(x.strftime('%m')))

df['last_credit_pull_d']= pd.to_datetime(df['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))

df['earliest_cr_line']= pd.to_datetime(df['earliest_cr_line'].fillna('2007-08-01')).apply(lambda x: int(x.strftime('%m')))

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
unique, counts = np.unique(df['loan_status'], return_counts=True)

print('Unique values:\n', np.asarray((unique, counts)).T) #check unique values
#df.dropna(inplace=True)
miss_values = null_values(df[(df["loan_status"]==3) | (df["loan_status"]==4)]) # loan_status 3,4 depend on these 3 columns, we need to keep them for tree boosting algorithms

miss_values.head(20)

#tot_coll_amt,tot_cur_bal, total_rev_hi_lim
no_three_column=df.drop(['tot_coll_amt','tot_cur_bal', 'total_rev_hi_lim'],1)

missing=null_values(no_three_column)

missing.head(20)
no_three_column.dropna(inplace=True)
no_three_column.shape
#df[(df["loan_status"]==3) | (df["loan_status"]==4)].dropna(inplace=True)
df2=df.loc[(df["loan_status"]!=3) | (df["loan_status"]!=4)].dropna()
df=pd.concat([df2,df[(df["loan_status"]==3) | (df["loan_status"]==4)]],0)
null_values(df) # these missing values are important for decision trees since they are only related to loan_status=3,4
df.shape
no_three_column.head(10)
X = no_three_column.drop('loan_status',1)

y = no_three_column['loan_status']
#build test and training sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#y_train.shape

unique, counts = np.unique(y_test, return_counts=True)

print('Unique values:\n', np.asarray((unique, counts)).T)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# log_reg_sm = LogisticRegression()

log_reg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



normal_ypred = log_reg.predict(X_test)

print(accuracy_score(y_test, normal_ypred))
#from sklearn.preprocessing import StandardScaler

# normalize dataset

#norm_X = StandardScaler() # initiate scaler

#X_train_norm = norm_X.fit_transform(X_train) # get normalization parameters based on train dataset 

#X_test_norm = norm_X.transform(X_test) # apply normalization to test data based on train dataset parameters
X = df.drop('loan_status',1)

y = df['loan_status']
#build test and training sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
import gc # clean unused variables from RAM

gc.collect()
#y_train.shape

unique, counts = np.unique(y_test, return_counts=True)

print('Unique values:\n', np.asarray((unique, counts)).T)
import lightgbm as lgb



param = {'boosting_type': 'gbdt','num_leaves':35,'nthread': 4, 'num_trees':100, 'objective':'multiclass', 'metric' : 'softmax',

        'num_class': 10}

train_data = lgb.Dataset(X_train, y_train, silent=False) 

test_data = lgb.Dataset(X_test, y_test, silent=False) 

model = lgb.train(param, train_set = train_data, num_boost_round=20, verbose_eval=4)
preds = model.predict(X_test, num_iteration = model.best_iteration)
from sklearn.metrics import accuracy_score

#predicted=np.where(preds > 0.5, 1, 0)

#preds[preds > 0.5] = 1

#preds[preds <= 0.5] = 0

#

(preds)

predictions = []



for x in preds:

    predictions.append(np.argmax(x))
acc_lgbm = accuracy_score(y_test,predictions)

print('Overall accuracy of Light GBM model:', acc_lgbm)
np.unique(predictions)
lgb.plot_importance(model, max_num_features=21, importance_type='split')

#tot_coll_amt,tot_cur_bal, total_rev_hi_lim
import seaborn as sns

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

sns.set_style("whitegrid")

import matplotlib.pyplot as plt
def conf_m(cm):

  labels = ['Charged Off', 'Current', 'Default',

         'not meet policy. Status:Charged Off',

         'not meet policy. Status:Fully Paid', 'Fully Paid',

         'In Grace Period', 'Issued', 'Late (16-30 days)',

         'Late (31-120 days)']

  plt.figure(figsize=(8,6))

  sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

  plt.title('Confusion Matrix')

  plt.ylabel('True Class')

  plt.xlabel('Predicted Class')

  return plt.show()
#Print Confusion Matrix

plt.figure()

cm = confusion_matrix(y_test, predictions)

conf_m(cm)
import imblearn

from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import OneSidedSelection

del X, y, X_train, X_test, y_train, y_test

gc.collect()
X = no_three_column.drop('loan_status',1)

y = no_three_column['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
oss = OneSidedSelection()

X_resampled, y_resampled = oss.fit_sample(X, y)
np.unique(y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=14)
train_data = lgb.Dataset(X_train, y_train, silent=False) 

test_data = lgb.Dataset(X_test, y_test, silent=False) 

param = {'boosting_type': 'gbdt','num_leaves':45,'nthread': 4, 'objective':'multiclass', 'metric' : 'softmax',

        'num_class': 10}

model = lgb.train(param, train_set = train_data, verbose_eval=5, num_boost_round=40)
preds = model.predict(X_test, num_iteration = model.best_iteration)
from sklearn.metrics import accuracy_score



(preds)

predictions = []



for x in preds:

    predictions.append(np.argmax(x))
acc_lgbm = accuracy_score(y_test,predictions)

print('Overall accuracy of Light GBM model:', acc_lgbm)
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, predictions)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
plt.figure()

cm = confusion_matrix(y_test, predictions)

conf_m(cm)
# this is required for XG boost because column names containing those [] < characters can not be trained

#import re

#regex = re.compile(r"\[|\]|<", re.IGNORECASE)

#X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

#X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

from xgboost import XGBClassifier





model = XGBClassifier(objective='multi:softmax', nthread=-1 )

eval_set = [(X_test, y_test)]

model.fit(X_train, y_train, early_stopping_rounds=1, eval_metric="merror", eval_set=eval_set, verbose=True )

# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, predictions)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
#Print Confusion Matrix

plt.figure()

cm = confusion_matrix(y_test, predictions)

labels = ['Charged Off', 'Current', 'Default',

       'Does not meet the credit policy. Status:Charged Off',

       'Does not meet the credit policy. Status:Fully Paid', 'Fully Paid',

       'In Grace Period', 'Issued', 'Late (16-30 days)',

       'Late (31-120 days)']

plt.figure(figsize=(8,6))

sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

plt.show()
from xgboost import plot_importance

from matplotlib import pyplot



plot_importance(model)

pyplot.show()