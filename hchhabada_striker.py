# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import numpy as np

from  sklearn.metrics import accuracy_score

from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler

from sklearn.base import TransformerMixin



def gini(actual, pred, cmpcol = 0, sortcol = 1):

     assert( len(actual) == len(pred) )

     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

     totalLosses = all[:,0].sum()

     giniSum = all[:,0].cumsum().sum() / totalLosses

 

     giniSum -= (len(actual) + 1) / 2.

     return giniSum / len(actual)

 

def gini_normalized(a, p):

     return gini(a, p) / gini(a, a)

 

def eval_gini(y_true, y_prob):

    """

    Original author CPMP : https://www.kaggle.com/cpmpml

    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = eval_gini(labels, preds)

    return [('gini', gini_score)]



print("Reading Input")

df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")





class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)



target = df['target']

df.drop('target',axis=1)



big_X = df.append(df_test)

big_X_imputed = DataFrameImputer().fit_transform(big_X)





df = big_X_imputed[0:df.shape[0]]

df_test = big_X_imputed[df.shape[0]::]







# Prepare the inputs for the model





df['target'] = target

y = []

def upsampling(df):

    df_majority=df[df.target==0]

    df_minority=df[df.target==1]

    # Upsample minority class

    df_minority_upsampled = resample(df_minority, 

                                     replace=True,     # sample with replacement

                                     n_samples=473518,    # to match majority class

                                     random_state=123) # reproducible results



    # Combine majority class with upsampled minority class  n_samples=573518

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    

    return df_upsampled



ids = df_test['id'].as_matrix()

df_test.drop('id',axis=1)



#upsampling

df_upsampled = upsampling(df)

y= df_upsampled['target'].as_matrix().astype(float)

df_upsampled.drop('target',axis=1)

df_upsampled.drop('id',axis=1)



X = df_upsampled.as_matrix()

X_test_new = df_test.as_matrix()









scaler = StandardScaler()

# Fit only to the training data

scaler.fit(X_test_new)

# Now apply the transformations to the data:

#X_train = scaler.transform(X_train)

X_test_new = scaler.transform(X_test_new)



scaler.fit(X)

X = scaler.transform(X)



from sklearn import decomposition

pca = decomposition.PCA(n_components=4)



X = pca.fit_transform(X)



X_test_1 = pca.fit_transform(X_test_new)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.10)



print("Model Building")

from xgboost import XGBClassifier



n_estimators = 300

n_splits = 4

print("Kfold")

from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)



'''

param = {

 'n_estimators':[100,150,200,250],

 'max_depth':[2,3,4,5,6,7,8,9],

 'min_child_weight':[2,3,4,5],

 'colsample_bytree':[0.2,0.6,0.8],

 'colsample_bylevel':[0.2,0.6,0.8],

 'learning_rate':[0.02,0.04,0.2]

}

from sklearn.grid_search import GridSearchCV

gsearch1 = GridSearchCV(estimator = XGBClassifier( 

        objective= "binary:logistic", 

        seed=1), 

    param_grid = param, 

    scoring='neg_log_loss',

    cv=4,

    verbose = 1)

    

# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

# mean_squared_error alternative.



gsearch1.fit(X_train, y_train)

print(gsearch1.bestscore)

print(gsearch1.bestparams)

'''



clf = XGBClassifier(n_estimators=n_estimators,

                        max_depth=5,

                        objective="binary:logistic",

                        learning_rate=.02, 

                        subsample=.8, 

                        colsample_bytree=.8,

                        gamma=1,

                        reg_alpha=0,

                        reg_lambda=1,

                        nthread=2)



import matplotlib.pyplot as plt

import scikitplot as skplt

from sklearn import metrics

for train_index, test_index in folds.split(X,y):

    

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]





    print("Model fitting")

    # Fit the best algorithm to the data. 

    clf.fit(X_train, y_train)

    



    predictions = clf.predict(X_test)

    predictions_prob = clf.predict_proba(X_test)

    print(accuracy_score(y_test, predictions))

    predictions_prob_test = clf.predict_proba(X_test_1)

    skplt.metrics.plot_precision_recall_curve(y_test, predictions_prob)

    plt.show()

    #roc

    preds_roc = predictions_prob[:,1]

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds_roc)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()











df2 = pd.DataFrame({'id' : ids})

df2['target'] = predictions_prob_test[:,1]

df2.to_csv("submission_5_xgbKold_1.csv",index= False)

print(df2)

#print(predictions_prob)

#print(predictions_prob[:,1])









print("Full OOF score : %.6f" % gini_normalized(y_test,predictions_prob[:,1]))



#print(accuracy_score(y_test, predictions))







#print(accuracy_score(y_test, predictions))



#print(pd.DataFrame(pca.components_,columns=df_norm.columns,index = ['PC-1','PC-2']))

#list(X)

# Any results you write to the current directory are saved as output.