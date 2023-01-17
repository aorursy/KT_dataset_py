# Quick load dataset and check
import pandas as pd
import numpy as np
import os
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
os.listdir('../input')
if ~running_local:
    path = "../input/ml2020-zsbnn-uibk/"
else:
    path = "../"

filename = path + "train_set.csv"
data_train = pd.read_csv(filename)
filename = path + "test_set.csv"
data_test = pd.read_csv(filename)
data_train.describe()
data_test.describe()
## Select target and features

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



## Select target and features
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]




from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


smote= SMOTE(sampling_strategy='auto', random_state=7)


x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

x_smote, y_smote = smote.fit_sample(x_train, y_train)

clf = DecisionTreeClassifier(min_impurity_decrease = 0.00, class_weight='balanced')
clf = clf.fit(x_smote, y_smote)
y_pred = clf.predict(x_val)

sum(y_pred==y_val)/len(y_val)

def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
sum(y_pospred==y_pos)/len(y_pos)
X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
data_test_X = data_test.drop(columns=['id'])
y_target = clf.predict(data_test_X)
sum(y_target==0)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
data_out
