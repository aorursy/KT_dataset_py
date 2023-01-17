# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
train_transaction.set_index('TransactionID', inplace=True)

train_identity.set_index('TransactionID', inplace=True)

test_transaction.set_index('TransactionID', inplace=True)

test_identity.set_index('TransactionID', inplace=True)
import numpy as np



def smaller_sample(data, fraction=.01):

    indexes = np.random.randint(0, high=data.shape[0], size=int(fraction*data.shape[0]))

    return data.iloc[indexes]
s_train_identity = smaller_sample(train_identity)

s_train_transaction = smaller_sample(train_transaction)

s_test_transaction = smaller_sample(test_transaction)

s_test_identity = smaller_sample(test_identity)
train_transaction['isFraud'].value_counts()
X_balanced = pd.concat([

    train_transaction[train_transaction['isFraud']==1], 

    smaller_sample(train_transaction[train_transaction['isFraud']==0], fraction=20663/569877)

])
X_balanced['isFraud'].value_counts()
X_balanced_dropna = X_balanced.dropna(axis=1)
X_balanced_dropna.head()
from sklearn.preprocessing import OneHotEncoder



one_hot = OneHotEncoder()

X_balanced_dropna_cat = X_balanced_dropna[['ProductCD']]



X_balanced_dropna_cat_oh = one_hot.fit_transform(X_balanced_dropna_cat)
X_balanced_dropna_cat_oh = pd.DataFrame(X_balanced_dropna_cat_oh.toarray(), columns=one_hot.categories_)
X_train = pd.concat([

    X_balanced_dropna_cat_oh, 

    X_balanced_dropna.drop(['ProductCD', 'card1'], axis=1).reset_index()

], axis=1)
X_train.corr()['isFraud']
X_train_featselection = X_train.iloc[:, [0, 4, 14, 18]]
y = X_train['isFraud']
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier



rfc = RandomForestClassifier(n_estimators=100)

sgd = SGDClassifier(loss='log')

gnb = GaussianNB()

knc = KNeighborsClassifier()
from sklearn.model_selection import cross_val_predict



y_pred_rfc = cross_val_predict(rfc, X_train_featselection, y, cv=5, method='predict_proba')

y_pred_sgd = cross_val_predict(sgd, X_train_featselection, y, cv=5, method='predict_proba')

y_pred_gnb = cross_val_predict(gnb, X_train_featselection, y, cv=5, method='predict_proba')

y_pred_knc = cross_val_predict(knc, X_train_featselection, y, cv=5, method='predict_proba')
from sklearn.metrics import roc_auc_score



print("y_pred_rfc",roc_auc_score(y, y_pred_rfc[:, 1]))

print("y_pred_sgd",roc_auc_score(y, y_pred_sgd[:, 1]))

print("y_pred_gnb",roc_auc_score(y, y_pred_gnb[:, 1]))

print("y_pred_knc",roc_auc_score(y, y_pred_knc[:, 1]))
tx_ids = train_identity.index.values

df_merged = pd.concat([train_transaction.loc[tx_ids], train_identity], axis=1)
df_merged['isFraud'].value_counts()
df_merged_balanced = pd.concat([

    df_merged[df_merged['isFraud']==1], 

    smaller_sample(df_merged[df_merged['isFraud']==0], fraction=11318/132915)

])
df_merged_balanced['isFraud'].value_counts()
cat_cols = ["ProductCD", "card1", "card2", "card3", "card4","card5","card6", "addr1", "addr2","P_emaildomain", "R_emaildomain",

            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo", "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", 

           "id_18", "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", 

           "id_31", "id_32", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]



df_cat = df_merged_balanced.loc[:, cat_cols]

df_num = df_merged_balanced.drop(cat_cols, axis=1)

y = df_num['isFraud']
drop_threshold = 20 # in %



cat_pct_missing = df_cat.describe().iloc[0].apply(lambda x:  (22636-x) / 22636 * 100) # Number of Missing Values

cat_drop_cols = cat_pct_missing[cat_pct_missing > drop_threshold]

cat_impute_cols = cat_pct_missing[(cat_pct_missing < drop_threshold) & (cat_pct_missing > 0)]



num_pct_missing = df_num.describe().iloc[0].apply(lambda x:  (22636-x) / 22636 * 100) # Number of Missing Values

num_drop_cols = num_pct_missing[num_pct_missing > drop_threshold]

num_impute_cols = num_pct_missing[(num_pct_missing < drop_threshold) & (num_pct_missing > 0)]
df_cat_dropcols = df_cat.drop(cat_drop_cols.index.values, axis=1)

df_num_dropcols = df_num.drop(num_drop_cols.index.values, axis=1)
import numpy as np

from sklearn.impute import SimpleImputer



imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')



df_imputed_cat = imp_cat.fit_transform(df_cat_dropcols)

df_imputed_num = imp_num.fit_transform(df_num_dropcols)
from sklearn.preprocessing import OneHotEncoder



one_hot = OneHotEncoder()



df_imputed_cat_oh = one_hot.fit_transform(df_imputed_cat)



df_imputed_cat_oh = pd.DataFrame(df_imputed_cat_oh.toarray())

df_imputed_num = pd.DataFrame(df_imputed_num)
X_train = pd.concat([df_imputed_num,df_imputed_cat_oh], axis=1)
corrs = []



for i in range(X_train.shape[1]):

    corrs.append(np.corrcoef(X_train.iloc[:, i].values, y.values)[0,1])
import heapq

import math 



abs_corrs = [abs(number) for number in corrs]

clean_corrs = [corr for corr in abs_corrs if not math.isnan(corr)]

indexes = []

for element in heapq.nlargest(35, clean_corrs):

    indexes.append(abs_corrs.index(element))

    
indexes.pop(0)

X_selected = X_train.iloc[:, indexes]
rfc = RandomForestClassifier(n_estimators=100)

sgd = SGDClassifier(loss='log')

gnb = GaussianNB()

knc = KNeighborsClassifier()



from sklearn.model_selection import cross_val_predict



y_pred_rfc = cross_val_predict(rfc, X_selected, y, cv=5, method='predict_proba')

#y_pred_sgd = cross_val_predict(sgd, X_train_featselection, y, cv=5, method='predict_proba')

#y_pred_gnb = cross_val_predict(gnb, X_train_featselection, y, cv=5, method='predict_proba')

#y_pred_knc = cross_val_predict(knc, X_train_featselection, y, cv=5, method='predict_proba')



from sklearn.metrics import roc_auc_score



print("y_pred_rfc",roc_auc_score(y, y_pred_rfc[:, 1]))

#print("y_pred_sgd",roc_auc_score(y, y_pred_sgd[:, 1]))

#print("y_pred_gnb",roc_auc_score(y, y_pred_gnb[:, 1]))

#print("y_pred_knc",roc_auc_score(y, y_pred_knc[:, 1]))
X_selected.head()