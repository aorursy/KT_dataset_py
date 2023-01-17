# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gs = pd.read_csv('../input/gender_submission.csv')
training_colums = [
    'Pclass',
    'gender','Age','SibSp',
    'Parch'
]
bdt = GradientBoostingClassifier()
m = {'m' : 1, 'f' : 0}
df_train['gender'] = df_train['Sex'].str[0].str.lower().map(m)
df_test['gender'] = df_test['Sex'].str[0].str.lower().map(m)
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(df_train[training_colums])
df_train_modify = df_train.copy()
cols_with_missing = (col for col in df_train_modify.columns 
                                 if df_train_modify[col].isnull().any())
for col in cols_with_missing:
    df_train_modify[col + '_was_missing'] = df_train_modify[col].isnull()
    
df_train_modify[training_colums] = pd.DataFrame(my_imputer.fit_transform(df_train_modify[training_colums]))
df_train_modify[training_colums].columns = df_train[training_colums].columns
df_test_modify = df_test.copy()
cols_with_missing = (col for col in df_test_modify.columns 
                                 if df_test_modify[col].isnull().any())
for col in cols_with_missing:
    df_test_modify[col + '_was_missing'] = df_test_modify[col].isnull()
df_test_modify[training_colums] = pd.DataFrame(my_imputer.fit_transform(df_test_modify[training_colums]))
df_test_modify[training_colums].columns = df_test[training_colums].columns
%time
#df_train_new = df_train.dropna()
bdt.fit(df_train_modify[training_colums], df_train_modify['Survived'])
y_score = bdt.predict_proba(df_train_modify[training_colums])[:,1]
fpr, tpr, threshold = roc_curve(df_train_modify['Survived'],y_score)

n_sig = 1200
n_bkg = 23000
S = tpr*n_sig
B = fpr*n_bkg
FoM = S/np.sqrt(S+B)
plt.plot(fpr, tpr, label='Roc_Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot(threshold, FoM, label="FoM")
plt.xlabel('BDT cut value')
plt.xlim([0,1])
best_cut = threshold[np.argmax(FoM[~np.isnan(FoM)])]
plt.axvline(x=best_cut, color = 'red', linestyle = '--', label=f'Best cut: {best_cut:.2f}')
plt.legend()


