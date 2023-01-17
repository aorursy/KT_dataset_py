import pandas as pd
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
import warnings

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,classification_report, recall_score, precision_recall_curve

# Define random state
random_state = 2018
np.random.seed(random_state)
warnings.filterwarnings('ignore')
%matplotlib inline
train=pd.read_csv("../input/minor-project-2020/train.csv")
test=pd.read_csv("../input/minor-project-2020/test.csv")
train=train.drop(labels="id",axis=1)
test=test.drop(labels="id",axis=1)
xtrain=train.drop(labels="target",axis=1)
ytrain=train[["target"]]
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler , CondensedNearestNeighbour
ada = ADASYN(random_state=random_state)
CNN =  CondensedNearestNeighbour(random_state=random_state)
rand = RandomUnderSampler(random_state=random_state , sampling_strategy=0.08)
X,y = rand.fit_resample(xtrain, ytrain)
y.value_counts()

class Create_ensemble(object):
    def __init__(self, n_splits, base_models):
        self.n_splits = n_splits
        self.base_models = base_models

    def predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        no_class = len(np.unique(y))

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                     random_state = random_state).split(X, y))

        train_proba = np.zeros((X.shape[0], no_class))
        test_proba = np.zeros((T.shape[0], no_class))
        
        train_pred = np.zeros((X.shape[0], len(self.base_models)))
        test_pred = np.zeros((T.shape[0], len(self.base_models)* self.n_splits))
        f1_scores = np.zeros((len(self.base_models), self.n_splits))
        recall_scores = np.zeros((len(self.base_models), self.n_splits))
        
        test_col = 0
        for i, clf in enumerate(self.base_models):
            
            for j, (train_idx, valid_idx) in enumerate(folds):
                
                X_train = X[train_idx]
                Y_train = y[train_idx]
                X_valid = X[valid_idx]
                Y_valid = y[valid_idx]
                
                clf.fit(X_train, Y_train)
                
                valid_pred = clf.predict(X_valid)
                recall  = recall_score(Y_valid, valid_pred, average='macro')
                f1 = f1_score(Y_valid, valid_pred, average='macro')
                
                recall_scores[i][j] = recall
                f1_scores[i][j] = f1
                
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)
                test_col += 1
                
                ## Probabilities
                valid_proba = clf.predict_proba(X_valid)
                train_proba[valid_idx, :] = valid_proba
                test_proba  += clf.predict_proba(T)
                
                print( "Model- {} and CV- {} recall: {}, f1_score: {}".format(i, j, recall, f1))
                
            test_proba /= self.n_splits
            
        return train_proba, test_proba, train_pred, test_pred
LR=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                                        intercept_scaling=1,class_weight="balanced", random_state=random_state, solver='lbfgs',
                                        max_iter=100, multi_class='auto', verbose=0, warm_start=True,
                                        n_jobs=None, l1_ratio=None)
base_models2 = [LR]
n_splits = 5
Model2_LR_balanced = Create_ensemble(n_splits = n_splits, base_models = base_models2)
train_proba, test_proba, train_pred, test_pred = Model2_LR_balanced.predict(X, y, test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y, train_pred, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y, train_pred, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y, train_pred)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y, train_pred)))
test1=pd.read_csv("../input/minor-project-2020/test.csv")
r=[]
for i,id in enumerate(test1["id"]):
  r.append([id,test_proba[i][1]])
submission = pd.DataFrame(r, columns = ['id', 'target'])
submission.to_csv("./submission_ratio_0.08_balanced_LR2.csv",index=False)