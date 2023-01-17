#import library

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn import model_selection, svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import xgboost as xgb

%matplotlib inline
df = pd.read_csv("../input/bank-additional.csv") 

# Replacing chars with numbers

numeric_data = df

label = LabelEncoder()

dicts = {}



X = [

                   'age', 'job', 'marital',

                   'education', 'default', 'housing',

                   'loan','contact',

                   'month','day_of_week','duration', 'campaign',

                   'pdays','previous',

                   'poutcome', 'emp.var.rate',

                   'cons.price.idx', 'cons.conf.idx',

                   'euribor3m', 'nr.employed'

]



fields = X

fields.append('y')





for f in fields:

    label.fit(df[f].drop_duplicates())

    dicts[f] = list(label.classes_)

    numeric_data[f] = label.transform(df[f])    



target = numeric_data['y']

numeric_data = numeric_data.drop(['y'], axis=1)     



# Looking for most valuable columns in our dataset

# k-value affect auc final score and roc curve

numeric_data_best = SelectKBest(f_classif, k=7).fit_transform(numeric_data, target)



#looking for null data

df.isnull().sum()
# comparing best model

model_lr = LogisticRegression(penalty='l1', tol=0.01) 

model_dt = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)

model_svc = svm.SVC() 

model_svc = SVC(kernel='rbf', random_state=0)

model_bnn = MLPClassifier()



ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = model_selection.train_test_split(numeric_data_best, target, test_size=0.3) 



results = {}

kfold = 10



results['LogisticRegression_best_params'] = model_selection.cross_val_score(model_lr, numeric_data_best, target, cv = kfold).mean()

results['DecisionTree_best_params'] = model_selection.cross_val_score(model_dt, numeric_data_best, target, cv = kfold).mean()

results['SVC_best_params'] = model_selection.cross_val_score(model_svc, numeric_data_best, target, cv = kfold).mean()

results['NN_best_params'] = model_selection.cross_val_score(model_bnn, numeric_data_best, target, cv = kfold).mean()



results['LogisticRegression_all_params'] = model_selection.cross_val_score(model_lr, numeric_data, target, cv = kfold).mean()

results['DecisionTree_all_params'] = model_selection.cross_val_score(model_dt, numeric_data, target, cv = kfold).mean()

results['SVC_all_params'] = model_selection.cross_val_score(model_svc, numeric_data, target, cv = kfold).mean()

results['NN_all_params'] = model_selection.cross_val_score(model_bnn, numeric_data, target, cv = kfold).mean()    

# ROC with all parameters

roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(numeric_data, target, test_size=0.25) 

roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 



models = [

    {

        'label' : 'SVC_all_params',

        'model': model_svc,

        'roc_train': roc_train_all,

        'roc_test': roc_test_all,

        'roc_train_class': roc_train_all_class,        

        'roc_test_class': roc_test_all_class,        

    },        

    {

        'label' : 'LogisticRegression_all_params',

        'model': model_lr,

        'roc_train': roc_train_all,

        'roc_test': roc_test_all,

        'roc_train_class': roc_train_all_class,        

        'roc_test_class': roc_test_all_class,        

    },

    {

        'label' : 'DecisionTree_all_params',

        'model': model_dt,

        'roc_train': roc_train_all,

        'roc_test': roc_test_all,

        'roc_train_class': roc_train_all_class,        

        'roc_test_class': roc_test_all_class,        

    },

    {

        'label' : 'NN_all_params',

        'model': model_bnn,

        'roc_train': roc_train_all,

        'roc_test': roc_test_all,

        'roc_train_class': roc_train_all_class,        

        'roc_test_class': roc_test_all_class,        

    }

]





plt.clf()

plt.figure(figsize=(8,6))



for m in models:

    m['model'].probability = True

    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])

    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])

    roc_auc  = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))





plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=0, fontsize='small')

plt.show()
# ROC with best parameters

roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(numeric_data, target, test_size=0.25) 

roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 



models = [

    {

        'label' : 'SVC_best_params',

        'model': model_svc,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },        

    {

        'label' : 'LogisticRegression_best_params',

        'model': model_lr,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },

    {

        'label' : 'DecisionTree_best_params',

        'model': model_dt,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },

    {

        'label' : 'NN_best_params',

        'model': model_bnn,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    }

]





plt.clf()

plt.figure(figsize=(8,6))



for m in models:

    m['model'].probability = True

    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])

    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])

    roc_auc  = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))





plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=0, fontsize='small')

plt.show()
xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(numeric_data,target,feature_names = numeric_data.columns.values)

model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=100)



fig,ax=plt.subplots(figsize = (13,19))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

plt.show()