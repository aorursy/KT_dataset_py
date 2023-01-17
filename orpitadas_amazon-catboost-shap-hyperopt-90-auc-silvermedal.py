import shap

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, log_loss

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, cv, Pool

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

from itertools import combinations



%matplotlib inline

sns.set(style='ticks')

pd.options.display.max_columns = 500

pd.options.display.max_rows = 500
# Read in data

test = pd.read_csv("../input/amazon-employee-access-challenge/test.csv")

train = pd.read_csv("../input/amazon-employee-access-challenge/train.csv")
def performance(model, X_test, y_test):

    

    """

    Accepts a fitted model and an evaluation dataset at input.

    Prints the confusion matrix, classification_report & auc score. 

    Also, displays Precision-Recall curve & ROC curve.

    """

    

    # Make predictions on test set

    y_pred=model.predict(X_test)

    y_pred=np.round(y_pred)

    

    # Confusion matrix

    print(confusion_matrix(y_test, y_pred))

    

    # AUC score

    y_pred_prob = model.predict_proba(X_test)

    print("AUC score: ", roc_auc_score(y_test, y_pred_prob[:,1]))

    

    # Logloss

    print("Logloss : ", log_loss(y_test, y_pred_prob))



    # Accuracy, Precision, Recall, F1 score

    print(classification_report(y_test, y_pred))

    

    # Precision-Recall curve

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])

    plt.grid(True)

    plt.show()



    # ROC curve

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

    plt.plot([0, 1], [0, 1],'k--')

    plt.plot(fpr, tpr, label='Neural Network')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show()

print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))
train.head()
print(train.isnull().any()) 

print(test.isnull().any())
# Compare number of Unique Categorical labels for train and test



unique_train= pd.DataFrame([(col,train[col].nunique()) for col in train.columns], 

                           columns=['Columns', 'Unique categories'])

unique_test=pd.DataFrame([(col,test[col].nunique()) for col in test.columns],

                columns=['Columns', 'Unique categories'])

unique_train=unique_train[1:]

unique_test=unique_test[1:]



fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].bar(unique_train.Columns, unique_train['Unique categories'])

ax[1].bar(unique_test.Columns, unique_test['Unique categories'])

plt.xticks(rotation=90)
sns.countplot(train.ACTION)
# Check for duplicated rows



if (sum(train.duplicated()), sum(test.duplicated())) == (0,0):

    print('No duplicated rows')

else: 

    print('train: ',sum(train.duplicated()))

    print('test: ',sum(train.duplicated()))
# Check for duplicated columns                          



for col1,col2 in combinations(train.columns, 2):

    condition1=len(train.groupby([col1,col2]).size())==len(train.groupby([col1]).size())

    condition2=len(train.groupby([col1,col2]).size())==len(train.groupby([col2]).size())

    condition3=(train[col1].nunique()==train[col2].nunique())

    if (condition1 | condition2) & condition3:

        print(col1,col2)

        print('Potential Categorical column duplication')
train.groupby(['ROLE_TITLE', 'ROLE_CODE']).mean()
np.random.seed(123)
# Drop duplicated column

train.drop('ROLE_CODE', axis=1, inplace=True)

test.drop('ROLE_CODE', axis=1, inplace=True)
# Split into features and target

y = train['ACTION']

X = train.drop('ACTION', axis=1)



# Split into train & validation set

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
cat_features = [*range(8)]
model = CatBoostClassifier(custom_metric=['TotalF1'], early_stopping_rounds=100, eval_metric='AUC')



model.fit(X_train, y_train, cat_features=cat_features,

          eval_set=(X_val, y_val), plot=True, verbose=False, use_best_model=True)
performance(model, X_val, y_val)
feat_imp=model.get_feature_importance(prettified=True)

plt.bar(feat_imp['Feature Id'], feat_imp['Importances'])

plt.xlabel('Features')

plt.ylabel('Feature Importance')

plt.xticks(rotation=90)
sub=pd.read_csv("../input/amazon-employee-access-challenge/sampleSubmission.csv")

sum(test.id==sub.Id), test.shape



#sub.to_csv('amazon1.csv', index=False, header=True)
y_pred=model.predict_proba(test.drop('id', axis=1))

sub.Action=y_pred[:,1]
sub.to_csv('amazon1.csv', index=False, header=True)

sub.head()
model.get_all_params()
"""

COmmented out as it takes too long to run. 

Under construction, some things can be improved.

space = {

    'depth': hp.quniform("depth", 1, 16, 1),

    'border_count': hp.quniform('border_count', 32, 255, 1),

    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),

    #'rsm': hp.uniform('rsm', 0.1, 1), # use only when task_type is default CPU

    'scale_pos_weight': hp.uniform('scale_pos_weight', 0.06, 1), # Can be set only when loss_function is default Logloss

    #'loss_function' : hp.choice('loss_function', ['Logloss', 'CrossEntropy'])

}





def hyperparameter_tuning(space):

    model = CatBoostClassifier(depth=int(space['depth']),

                               border_count=space['border_count'],

                               l2_leaf_reg=space['l2_leaf_reg'],

                               #rsm=space['rsm'],

                               scale_pos_weight=space['scale_pos_weight']

                               #loss_function=space['loss_function'],

                               task_type='GPU', # change to CPU when working on personal system

                               eval_metric='AUC'

                               early_stopping_rounds=100,

                              thread_count=-1)



    model.fit(X_train, y_train, cat_features=cat_features,use_best_model=True,

              verbose=False, eval_set=(X_val, y_val))



    preds_class = model.predict_proba(X_val)

    #score = classification_report(y_val, preds_class, output_dict=True)['0']['f1-score']

    score = roc_auc_score(y_val, preds_class[:,1])

    return{'loss': 1-score, 'status': STATUS_OK}





best = fmin(fn=hyperparameter_tuning,

            space=space,

            algo=tpe.suggest,

            max_evals=50)



print(best)

"""
# Best of the tuned models

model = CatBoostClassifier(border_count=248, depth=4, l2_leaf_reg=4.830204209625978,

                           scale_pos_weight=0.4107081177319144, 

                           eval_metric='AUC',

                           use_best_model=True,

                          early_stopping_rounds=100)

best=model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), use_best_model=True,

          verbose=False, plot=False)
performance(model, X_val, y_val)
model = CatBoostClassifier(border_count=248, depth=4, l2_leaf_reg=4.830204209625978,

                           scale_pos_weight=0.4107081177319144, iterations = 400)

model.fit(X_train, y_train, cat_features=cat_features,

          verbose=False, plot=False)

shap.initjs()

explainer = shap.TreeExplainer(model)
print('Probability of class 1 = {:.4f}'.format(model.predict_proba(X_train.iloc[2:3])[0][1]))

#print('Formula raw prediction = {:.4f}'.format(model.predict(X_train.iloc[0:1], prediction_type='RawFormulaVal')[0]))
shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=cat_features))

shap.force_plot(explainer.expected_value, shap_values[2,:], X_train.iloc[2,:])
shap.summary_plot(shap_values, X_train)
model = CatBoostClassifier(border_count=248, depth=4, l2_leaf_reg=4.830204209625978,

                           scale_pos_weight=0.4107081177319144,

                           loss_function='Logloss',

                           eval_metric='AUC',

                           use_best_model=True,

                          early_stopping_rounds=100)

cv_data = cv(Pool(X_train, y_train, cat_features=cat_features), params=model.get_params(),

             verbose=False)
score = np.max(cv_data['test-AUC-mean'])

print('AUC score from cross-validation: ', score)
cv_data['test-AUC-mean'].plot()

plt.xlabel('Iterations')

plt.ylabel('test-AUC-Mean')
clf = xgb.XGBClassifier()

clf.fit(X_train, y_train)
performance(clf, X_val, y_val)
train_data = lgb.Dataset(X_train, label=y_train)
param = {'objective': 'binary'}

param['metric'] = 'auc'

bst = lgb.train(train_set=train_data, params=param)
y_pred_prob=bst.predict(X_val)

y_pred=np.round(y_pred_prob)



# Confusion matrix

print(confusion_matrix(y_val, y_pred))
# AUC score

print("AUC score: ", roc_auc_score(y_val, y_pred_prob))
# Logloss

print("Logloss : ", log_loss(y_val, y_pred_prob))
# Accuracy, Precision, Recall, F1 score

print(classification_report(y_val, y_pred))