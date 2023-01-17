# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df
df.info()
df.columns
cat_df = df.select_dtypes(include=['int64']).copy()

cat_df = cat_df.drop(columns="ID")#delete ID from categorical data -> not useful

cat_df.columns
cat_df.shape
cat_df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}, inplace=True)
encode_columns=['SEX','MARRIAGE','EDUCATION']

for i in encode_columns:

    cat_df=pd.get_dummies(cat_df, columns=[i])
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

cat_df.columns

unique_status = np.unique(cat_df[['PAY_0']])

print("total unique statuses:", len(unique_status))

print(unique_status)
monthes=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for i in monthes:

    cat_df=pd.get_dummies(cat_df, columns=[i])

bins = [21, 30, 40, 50, 60, 76]

group_names = ['21-30', '31-40', '41-50', '51-60', '61-76']

age_cats = pd.cut(cat_df['AGE'], bins, labels=group_names)

cat_df['age_cats'] = pd.cut(cat_df['AGE'], bins, labels=group_names)

cat_df=pd.get_dummies(cat_df, columns=['age_cats'])
cat_df.head()
cat_df.columns
len(cat_df.columns)
cat_df.dtypes
len(cat_df.columns)
num_df = df.select_dtypes(include=['float64']).copy()

num_df.columns
bills=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5','BILL_AMT6']

col_names=['Q_BILL_AMT1', 'Q_BILL_AMT2', 'Q_BILL_AMT3', 'Q_BILL_AMT4','Q_BILL_AMT5', 'Q_BILL_AMT6']

i=0#counter 



for col in bills:

    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)

    i+=1

    

num_df.columns
num_df.head()
pays=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6','LIMIT_BAL']

col_names=['Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']

i=0#counter 



for col in pays:

    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)

    i+=1

    

num_df.columns
encode_columns=['Q_BILL_AMT1', 'Q_BILL_AMT2','Q_BILL_AMT3', 'Q_BILL_AMT4', 'Q_BILL_AMT5', 'Q_BILL_AMT6','Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']

for i in encode_columns:

    num_df=pd.get_dummies(num_df, columns=[i])
num_df.head()
num_df.columns
len(num_df.columns)
num_df['late_payer']=df['PAY_0'].apply(lambda x: 1 if x > 1 else 0)



num_df['late_payer'].head()
bill_mons=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

cols=['OVER_BILL_AMT1','OVER_BILL_AMT2','OVER_BILL_AMT3','OVER_BILL_AMT4','OVER_BILL_AMT5','OVER_BILL_AMT6']

i=0#counter



for mon in bill_mons:

    num_df[cols[i]]=df[mon].apply(lambda x: 1 if x < 0 else 0)

    i+=1

    

num_df['OVER_BILL_AMT1'].head()    
data = pd.concat([cat_df, num_df], axis=1)
data.to_csv('mycsvfile.csv',index=False)
data.head()
len(data.columns)
data.dtypes
data_=data[data.columns[~data.columns.isin(['default.payment.next.month'])]]#already does not have ID
target=data['default.payment.next.month']
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

from sklearn import preprocessing

import matplotlib.pyplot as plt 



# create X (features) and y (response)

X = data_

y = target



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# check classification scores of logistic regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Train/Test split results:')

print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))

print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))



idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95



#plot

plt.figure()

plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')

plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)

plt.ylabel('True Positive Rate (recall)', fontsize=14)

plt.title('Receiver operating characteristic (ROC) curve')

plt.legend(loc="lower right")

plt.show()



print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  

      "and a specificity of %.3f" % (1-fpr[idx]) + 

      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
# precision-recall curve and f1

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from matplotlib import pyplot



# split into train/test sets

# create X (features) and y (response)

X = data_

y = target



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# fit a model

model = LogisticRegression(solver='lbfgs')

model.fit(X_train, y_train)

# predict probabilities

lr_probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

# predict class values

yhat = model.predict(X_test)

lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)

lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores

print('Logistic: f1=%.3f AUPR=%.3f' % (lr_f1, lr_auc))

# plot the precision-recall curves

no_skill = len(y_test[y_test==1]) / len(y_test)

pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')

# axis labels

pyplot.xlabel('Recall')

pyplot.ylabel('Precision')

# show the legend

pyplot.legend()

# show the plot

pyplot.show()
import h2o

h2o.init()
#Import H2O and other libraries that will be used in this tutorial 

import matplotlib as plt

%matplotlib inline



#Import the Estimators

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators import H2ORandomForestEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator



#Import h2o grid search 

import h2o.grid 

from h2o.grid.grid_search import H2OGridSearch
clus=h2o.H2OFrame(data_)

clus.head()
import os



startup  = '/home/h2o/bin/aquarium_startup'

shutdown = '/home/h2o/bin/aquarium_stop'



if os.path.exists(startup):

    os.system(startup)

    local_url = 'http://localhost:54321/h2o'

    aquarium = True

else:

    local_url = 'http://localhost:54321'

    aquarium = False
h2o.init(url=local_url)
orig_data=h2o.import_file("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
target=orig_data['default.payment.next.month']
target.head()
clus['default.payment.next.month']=orig_data['default.payment.next.month']
train, test = clus.split_frame([0.8], seed=42)
print("train:%d test:%d" % (train.nrows,test.nrows))
y = 'default.payment.next.month'



ignore = ["default.payment.next.month"] 



x = list(set(train.names) - set(ignore))



print(x)

train.names
glm = H2OGeneralizedLinearEstimator(family = "binomial", seed=42, model_id = 'default_glm')
%time glm.train(x = x, y = y, training_frame = train)
glm
glm.plot(metric='negative_log_likelihood')
glm.varimp_plot()
glm.predict(test).head(10)
default_glm_perf=glm.model_performance(test)
print("AUC: ",default_glm_perf.auc())
print("ACCURACY: ",default_glm_perf.accuracy())
print("F1 score: ",default_glm_perf.F1())
from h2o.model.confusion_matrix import ConfusionMatrix

rf = H2ORandomForestEstimator (seed=42, model_id='default_rf')

%time rf.train(x=x, y=y, training_frame=train)
rf
rf.plot(metric='AUTO')
rf.varimp_plot(20)
clus['default.payment.next.month'] = clus['default.payment.next.month'].asfactor()    



train,test = clus.split_frame([0.7], seed=42)



y = 'default.payment.next.month'



x = list(set(train.names))



gbm= H2OGradientBoostingEstimator(seed=42, model_id='default_gbm')

%time gbm.train(x=x, y=y, training_frame=train)
gbm
default_gbm_perf=gbm.model_performance(test)
default_gbm_perf.accuracy()
gbm.varimp_plot(20)