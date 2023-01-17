# Importig libs

from xgboost import XGBClassifier

import xgboost



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn import model_selection, svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



%matplotlib inline
initial_data = pd.read_csv("../input/mushrooms.csv") 



# Replacing chars with numbers, removing  useless records

numeric_data = initial_data

label = LabelEncoder()

dicts = {}



fields_without_class = [

     'cap-shape', 'cap-surface', 

     'cap-color', 'bruises', 'odor', 

     'gill-attachment', 'gill-spacing', 'gill-size',

     'gill-color', 'stalk-shape', 'stalk-root', 

     'stalk-surface-above-ring', 'stalk-surface-below-ring',

     'stalk-color-above-ring', 'stalk-color-below-ring', 

     'veil-type', 'veil-color', 'ring-number', 'ring-type', 

     'spore-print-color', 'population', 'habitat'

]



fields = fields_without_class

fields.append('class')





for f in fields:

    label.fit(initial_data[f].drop_duplicates())

    dicts[f] = list(label.classes_)

    numeric_data[f] = label.transform(initial_data[f])    



target = numeric_data['class']

numeric_data = numeric_data.drop(['class'], axis=1)     



# Looking for most valuable columns in our dataset

numeric_data_best = SelectKBest(f_classif, k=6).fit_transform(numeric_data, target)
%matplotlib inline



# Trying to find best model

model_rfc = RandomForestClassifier(n_estimators = 70)

model_knc = KNeighborsClassifier(n_neighbors = 18) 

model_lr = LogisticRegression(penalty='l1', tol=0.01) 

model_gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)

model_svc = svm.SVC() 

model_xgb = XGBClassifier()

model_svc = SVC(kernel='rbf', random_state=0)



ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 



results = {}

kfold = 5



results['RandomForestClassifier_best_params'] = model_selection.cross_val_score(model_rfc, numeric_data_best, target, cv=kfold).mean()

results['KNeighborsClassifier_best_params'] = model_selection.cross_val_score(model_knc, numeric_data_best, target, cv=kfold).mean()

results['LogisticRegression_best_params'] = model_selection.cross_val_score(model_lr, numeric_data_best, target, cv = kfold).mean()

results['GradientBoosting_best_params'] = model_selection.cross_val_score(model_gb, numeric_data_best, target, cv = kfold).mean()

results['SVC_best_params'] = model_selection.cross_val_score(model_svc, numeric_data_best, target, cv = kfold).mean()

results['XGB_best_params'] = model_selection.cross_val_score(model_xgb, numeric_data_best, target, cv = kfold).mean()



results['RandomForestClassifier_all_params'] = model_selection.cross_val_score(model_rfc, numeric_data, target, cv=kfold).mean()

results['KNeighborsClassifier_all_params'] = model_selection.cross_val_score(model_knc, numeric_data, target, cv=kfold).mean()

results['LogisticRegression_all_params'] = model_selection.cross_val_score(model_lr, numeric_data, target, cv = kfold).mean()

results['GradientBoosting_all_params'] = model_selection.cross_val_score(model_gb, numeric_data, target, cv = kfold).mean()

results['SVC_all_params'] = model_selection.cross_val_score(model_svc, numeric_data, target, cv = kfold).mean()

results['XGB_all_params'] = model_selection.cross_val_score(model_xgb, numeric_data, target, cv = kfold).mean()

    



plt.bar(range(len(results)), results.values(), align='center')

plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')

plt.show()

# ROC

roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(numeric_data, target, test_size=0.25) 

roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 



models = [

    {

        'label' : 'GradientBoosting_best_params',

        'model': model_gb,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,                

    },

    {

        'label' : 'RandomForestClassifier_best_params',

        'model': model_rfc,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },

    {

        'label' : 'XGB_best_params',

        'model': model_gb,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },    

    {

        'label' : 'SVC_best_params',

        'model': model_svc,

        'roc_train': roc_train_best,

        'roc_test': roc_test_best,

        'roc_train_class': roc_train_best_class,        

        'roc_test_class': roc_test_best_class,        

    },        

    {

        'label' : 'KNeighborsClassifier_all_params',

        'model': model_knc,

        'roc_train': roc_train_all,

        'roc_test': roc_test_all,

        'roc_train_class': roc_train_all_class,        

        'roc_test_class': roc_test_all_class,        

    },

    {

        'label' : 'LogisticRegression_all_params',

        'model': model_knc,

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