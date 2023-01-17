import pandas as pd, numpy as np

train = pd.read_csv('../input/training.csv')

test = pd.read_csv('../input/testing.csv')
test['Made Donation in March 2007'] = 'NaN'

data = train.append(test)



# feature engineering



data['Months Donating'] = data['Months since First Donation'] - data['Months since Last Donation']



data['Donations per Months Donating'] = data["""Total Volume Donated (c.c.)"""]/data['Months Donating']

data['Donations per Months Donating'] = data['Donations per Months Donating'].replace(np.inf, 999)



data['Donations per Months since First Donation'] = data["""Total Volume Donated (c.c.)"""]/data['Months since First Donation']



data['Donation Counts per Months Donating'] = data['Number of Donations']/data['Months Donating']

data['Donation Counts per Months Donating'] = data['Donation Counts per Months Donating'].replace(np.inf, 999)



data['Donation Counts per Months since First Donating'] = data['Number of Donations']/data['Months since First Donation']

data['Donation Counts per Months since First Donating'] = data['Donation Counts per Months since First Donating'].replace(np.inf, 999)



data['Donation Volume per Donation'] = (data["""Total Volume Donated (c.c.)"""]/data['Number of Donations']).replace(np.inf, 999)

data['Unknown per Donation'] = (data["Unnamed: 0"]/data['Number of Donations']).replace(np.inf, 999)
test = data[data['Made Donation in March 2007'] == 'NaN']

test.drop(["Made Donation in March 2007"], axis = 1)

train = data[data['Made Donation in March 2007'] != 'NaN']
X = train.drop(['Made Donation in March 2007'], axis = 1)

y = train['Made Donation in March 2007']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42)
from catboost import Pool, CatBoostClassifier



train_pool = Pool(X_train, y_train, cat_features = [])

test_pool = Pool(X_test, y_test, cat_features = [])



model = CatBoostClassifier(

    depth = 4,

    random_seed = 42, 

    eval_metric = 'AUC',

    iterations = 1000,

    class_weights = [1, 3],

    verbose = True,

    loss_function= 'Logloss'

     )



model.fit(

    train_pool, 

    cat_features = None,

    eval_set = test_pool, 

    use_best_model = True,

    verbose = 100

    )
'''

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

'''
# predictions

predictions = model.predict(X_test).astype('int')

predictions_probs = model.predict_proba(X_test)

y_test = y_test.astype('int')
# MODEL EVALUATION



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

print('Accuracy: ', str(accuracy_score(y_test, predictions)))

print('Precision: ', str(precision_score(y_test, predictions)))

print('Recall: ', str(recall_score(y_test, predictions)))

print('F1: ', str(f1_score(y_test, predictions)))

print('Area under ROC Curve: ', str(roc_auc_score(y_test, predictions_probs[:,1])))

print('GINI: ', str(-1 + 2*roc_auc_score(y_test, predictions_probs[:,1])))



tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()



print('True Negatives: ', str(tn))

print('True Positives: ', str(tp))

print('False Negatives: ', str(fn))

print('False Positives: ', str(fp))
feature_importance = model.get_feature_importance(train_pool)

feature_names = X_train.columns

feature_imp = pd.DataFrame([feature_names, feature_importance])

final = feature_imp.transpose()

final.sort_values(by = 1, ascending = False, inplace = True)

pd.set_option('display.max_colwidth', -1)

final.head(500)