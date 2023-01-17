import pandas as pd # (e.g. pd.read_csv)

transaction = pd.read_csv('../input/anomaly-detection/creditcard.csv')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    transaction.drop('Class', axis=1), transaction['Class'], test_size=0.3, random_state=42)
import numpy as np # linear algebra

params = {

    'num_leaves': [500,600,700,800],

    'feature_fraction': list(np.arange(0.1,0.5,0.1)),

    'bagging_fraction': list(np.arange(0.1,0.5,0.1)),

    'min_data_in_leaf': [100,120,140,160],

    'learning_rate': [0.05],

    'reg_alpha': list(np.arange(0.1,0.5,0.1)),

    'reg_lambda': list(np.arange(0.1,0.5,0.1)),

}
import lightgbm as lgbm

model = lgbm.LGBMClassifier(random_state=42,metric='auc',verbosity=-1,objective='binary',max_depth=-1)
from sklearn.model_selection import RandomizedSearchCV

grid = RandomizedSearchCV(model,param_distributions=params,n_iter=15,cv=3,scoring='roc_auc')
grid.fit(X_train,y_train)
grid.best_params_
grid.best_score_
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test,grid.predict_proba(X_test)[:,1]))
from sklearn.metrics import classification_report

print(classification_report(y_test,grid.predict(X_test)))
params = {'reg_lambda': 0.4,

 'reg_alpha': 0.4,

 'num_leaves': 700,

 'min_data_in_leaf': 120,

 'learning_rate': 0.05,

 'feature_fraction': 0.2,

 'bagging_fraction': 0.1}





params = {

 'reg_lambda': 0.1,

 'reg_alpha': 0.1,

 'num_leaves': 800,

 'min_data_in_leaf': 100,

 'learning_rate': 0.05,

 'feature_fraction': 0.4,

 'bagging_fraction': 0.1,

 'verbosity' : -1,

  'objective' : 'binary',

  'random_state' : 42,

  'metric' : 'auc',

  'max_depth' : -1,

  'boosting_type': 'gbdt',

}
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import gc



n_folds = 5

folds = KFold(n_splits=n_folds)

columns = X_train.columns

y_preds = np.zeros(X_test.shape[0])

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):

    X_train, X_valid = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]

    y_train, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    

    temp_train = lgbm.Dataset(X_train, label=y_train)

    temp_valid = lgbm.Dataset(X_valid, label=y_valid)

    clf = lgbm.train(params,temp_train, 10000, valid_sets = [temp_train, temp_valid],

                      verbose_eval=200, early_stopping_rounds=500)

    

    y_pred_valid = clf.predict(X_valid)

    print("AUC: ",roc_auc_score(y_valid, y_pred_valid))

    y_preds += clf.predict(X_test) / n_folds

    

    #del X_train, X_valid, y_train, y_valid

    gc.collect()
import seaborn as sns

import matplotlib.pylab as plt



Mat1 = pd.DataFrame({"Class":y_test, "Prediction": y_preds.tolist()})

sns.lmplot( x="Class", y="Prediction", data=Mat1, fit_reg=False, hue='Class', height=8, aspect=17/8.27)
sns.distplot(Mat1['Prediction'], hist=False)

plt.legend()
sns.jointplot(x="Class", y="Prediction", data=Mat1)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



def plot_conf_Mat(Matrix):

# Creates a confusion matrix

    cm = confusion_matrix(Matrix['Class'].astype(np.int64), Matrix['Prediction'].astype(np.int64)) 



# Transform to df for easier plotting

    cm_df = pd.DataFrame(cm,

                     index = ['Class','Prediction'], 

                     columns = ['Class','Prediction'])



    plt.figure(figsize=(5.5,4))

    sns.heatmap(cm, annot=True)

    plt.title('CM \nAccuracy:{0:.3f}'.format(accuracy_score(Matrix['Class'].astype(np.int64), Matrix['Prediction'].astype(np.int64))))

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



plot_conf_Mat(Mat1)