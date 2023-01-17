import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import roc_auc_score

from sklearn import metrics

%matplotlib inline

plt.rcParams['figure.figsize'] = [20,15]



#Show all columns

pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/train.csv', index_col=0)

train.shape
test = pd.read_csv('../input/test.csv', index_col=0)

test.head()
train_test = [train, test]

len(train_test)
train.TARGET.value_counts()*100/train.shape[0]
Nb_target=train["TARGET"].value_counts(normalize=True).plot(kind="bar")

plt.title("TARGET")
delete = []

for col in train.columns:

    if train[col].std() == 0:

        delete.append(col) 
for df in train_test:

    df.drop(delete, axis=1, inplace=True)
def remove_duplicate(df,list_cols):

    delete = []

    for i in range(len(list_cols)-1):

        v = df[list_cols[i]].values

        for j in range(i+1,len(list_cols)):

            if np.array_equal(v,df[list_cols[j]].values):

                delete.append(list_cols[j])

    return delete
delete = remove_duplicate(train,train.columns)

for df in train_test:

    df.drop(delete, axis=1, inplace=True)

print(train.shape)

print(test.shape)
train.var3.value_counts().head()
train = train.replace(-999999,2)
train.var3.value_counts().head()
X = train.iloc[:,:-1]

y = train.TARGET



#X['n0'] = (X==0).sum(axis=1)

#train['n0'] = X['n0']
from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_classif,chi2

from sklearn.preprocessing import Binarizer, scale



#pourcentage des features à selectionner 3 /100 ==> pour 300 : environ 10 variables 

p = 3



X_bin = Binarizer().fit_transform(scale(X))

selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)

selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)



#Khi 2

chi2_selected = selectChi2.get_support() # renvoi un mask

chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]

print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),chi2_selected_features))



#Fisher amélioré

f_classif_selected = selectF_classif.get_support()

f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]

print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),f_classif_selected_features))



#Intersection khi2 & Fisher

selected = chi2_selected & f_classif_selected

print('Chi2 & F_classif selected {} features'.format(selected.sum()))

features = [ f for f,s in zip(X.columns, selected) if s]

print (features)



#Result

#Chi2 selected 9 features ['var15', 'ind_var5', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 'saldo_var30', 'var36', 'num_meses_var5_ult3'].



#F_classif selected 10 features ['var15', 'ind_var5', 'ind_var30', 'num_var4', 'num_var5', 'num_var30', 'num_var35', 'num_var42', 'var36', 'num_meses_var5_ult3'].



#Chi2 & F_classif selected 8 features ['var15', 'ind_var5', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 'var36', 'num_meses_var5_ult3']


X_sel = X[features]



X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.25)
# Use xgboost model

ratio = float(np.sum(y == 1)) / np.sum(y==0)



clf = xgb.XGBClassifier(missing=9999999999,

                max_depth = 5,

                n_estimators=1000,

                learning_rate=0.1, 

                nthread=4,

                subsample=1.0,

                colsample_bytree=0.5,

                min_child_weight = 3,

                scale_pos_weight = ratio,

                reg_alpha=0.03,

                seed=1301)

#Fit model

                

clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",

        eval_set=[(X_train, y_train), (X_test, y_test)])

        

# Accuracy

print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

    

#Overall AUC: 0.8033776289761753
sel_test = test[features]

#select best interation of xgboost

y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

y_pred

score_serie=pd.Series(clf.get_booster().get_fscore())

score_serie.sort_values().plot(kind="barh", title="Features importance xgboost")

kaggel_sub = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})

kaggel_sub.to_csv("submission.csv", index=False)