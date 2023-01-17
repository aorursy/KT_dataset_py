import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



import gc
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv", index_col='id')
train['missing_count'] = train.apply(lambda x: x.count(), axis=1)

test['missing_count'] = test.apply(lambda x: x.count(), axis=1)
#Replace missing data with "-1"

train = train.fillna(-1)

test = test.fillna(-1)
# Binary Features

# bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



dft = pd.get_dummies(train, columns=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'], drop_first=True)

dfte = pd.get_dummies(test, columns=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'], drop_first=True)







# Nominal Features (with more than 2 and less than 15 values)

# nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']



dft1 = pd.get_dummies(dft, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

                     ,drop_first=True

                     )



dfte1 = pd.get_dummies(dfte, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

                       ,drop_first=True

                      )



# Ordinal Features (with more than 2 and less than 15 values)

# ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']



#Only ord_0 is numerical values;

#We need to transform ord_1, ord_2 and ord_3 to set it in the correctly order to feed the machine learning model

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)





# Transforming ordinal Features - train

dft1.ord_1 = dft1.ord_1.astype(ord_1)

dft1.ord_2 = dft1.ord_2.astype(ord_2)

dft1.ord_3 = dft1.ord_3.astype(ord_3)

dft1.ord_4 = dft1.ord_4.astype(ord_4)



# test dataset - test

dfte1.ord_1 = dfte1.ord_1.astype(ord_1)

dfte1.ord_2 = dfte1.ord_2.astype(ord_2)

dfte1.ord_3 = dfte1.ord_3.astype(ord_3)

dfte1.ord_4 = dfte1.ord_4.astype(ord_4)



# Getting the codes of ordinal categoy's - train

dft1.ord_1 = dft1.ord_1.cat.codes

dft1.ord_2 = dft1.ord_2.cat.codes

dft1.ord_3 = dft1.ord_3.cat.codes

dft1.ord_4 = dft1.ord_4.cat.codes



# Geting the codes of ordinal categoy's - test

dfte1.ord_1 = dfte1.ord_1.cat.codes

dfte1.ord_2 = dfte1.ord_2.cat.codes

dfte1.ord_3 = dfte1.ord_3.cat.codes

dfte1.ord_4 = dfte1.ord_4.cat.codes







# Ordinal Feature - High Cardinality Features [ord_5]

dft1['ord_5'] = dft1.ord_5.astype('str')

dfte1['ord_5'] = dfte1.ord_5.astype('str')



### Credit of this features to: 

## https://www.kaggle.com/gogo827jz/catboost-baseline-with-feature-importance



import string



# Then encode 'ord_5' using ACSII values



# Option 1: Add up the indices of two letters in string.ascii_letters

dft1['ord_5_oe_add'] = dft1['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

dfte1['ord_5_oe_add'] = dfte1['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



# Option 2: Join the indices of two letters in string.ascii_letters

dft1['ord_5_oe_join'] = dft1['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

dfte1['ord_5_oe_join'] = dfte1['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))



# Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately

dft1['ord_5_oe1'] = dft1['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

dfte1['ord_5_oe1'] = dfte1['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))



dft1['ord_5_oe2'] = dft1['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

dfte1['ord_5_oe2'] = dfte1['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))



for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:

    dft1[col]= dft1[col].astype('float64')

    dfte1[col]= dfte1[col].astype('float64')

    

    

    

# Date features

# date_cols = ['day', 'month']





# Transfer the cyclical features into two dimensional sin-cos features

# https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning



def date_cyc_enc(dft1, col, max_vals):

    dft1[col + '_sin'] = np.sin(2 * np.pi * dft1[col]/max_vals)

    dft1[col + '_cos'] = np.cos(2 * np.pi * dft1[col]/max_vals)

    return dft1



dft1 = date_cyc_enc(dft1, 'day', 7)

dfte1 = date_cyc_enc(dfte1, 'day', 7) 



dft1 = date_cyc_enc(dft1, 'month', 12)

dfte1 = date_cyc_enc(dfte1, 'month', 12)



# NOTE, I discovered it on: kaggle.com/gogo827jz/catboost-baseline-with-feature-importance
# Nominal Features - High Cardinality Features

# high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



#Hashing trick:



high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for col in high_card_feats:

    dft2[f'hash_{col}'] = dft2[col].apply( lambda x: hash(str(x)) % 5000 )

    dfte2[f'hash_{col}'] = dfte2[col].apply( lambda x: hash(str(x)) % 5000 )

    

    

#Encoding:

from sklearn.preprocessing import LabelEncoder



# Label Encoding

for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']:

    if dft2[f].dtype=='object' or dfte2[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(dft2[f].values) + list(dfte2[f].values))

        dft2[f'le_{f}'] = lbl.transform(list(dft2[f].values))

        dfte2[f'le_{f}'] = lbl.transform(list(dfte2[f].values))  
k = dft2.corr().unstack().sort_values()

k.head(20)
#Remove notneeded columns:

dft2.drop(['bin_4_N', 'bin_3_F', 'bin_2_0.0', 'bin_1_0.0', 

                #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9',

               #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',

                #'freq_nom_5','freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',

                'bin_0_0.0', 'nom_0_Blue', 'nom_4_Bassoon', 'nom_2_Axolotl', 'nom_1_Polygon', 'nom_3_India',

              ], axis=1, inplace=True)



dfte2.drop(['bin_4_N', 'bin_3_F', 'bin_2_0.0', 'bin_1_0.0',

              #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9', 

              #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',

              #'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',

              'bin_0_0.0', 'nom_0_Blue', 'nom_4_Bassoon', 'nom_2_Axolotl', 'nom_1_Polygon','nom_3_India',

              ], axis=1, inplace=True)
#X=train.drop(['target'],axis=1)

#y=train['target']
dft2
#Remove notneeded columns:

dft2.drop(['day', 'month', 'ord_5', 

                #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9',

               #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',

                #'freq_nom_5','freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',

                'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

              ], axis=1, inplace=True)



dfte2.drop(['day', 'month', 'ord_5',

              #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9', 

              #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',

              #'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',

              'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

              ], axis=1, inplace=True)
 #Import data for target encoding:

train1 = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test1 = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
train1['missing_count'] = train1.apply(lambda x: x.count(), axis=1)

test1['missing_count'] = test1.apply(lambda x: x.count(), axis=1)
#Replace missing data with "-1"

train1 = train1.fillna(-1)

test1 = test1.fillna(-1)
# Binary Features

# bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



dft = pd.get_dummies(train1, columns=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'], drop_first=True)

dfte = pd.get_dummies(test1, columns=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'], drop_first=True)







# Nominal Features (with more than 2 and less than 15 values)

# nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']



dft1 = pd.get_dummies(dft, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

                     ,drop_first=True

                     )



dfte1 = pd.get_dummies(dfte, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

                       ,drop_first=True

                      )



# Ordinal Features (with more than 2 and less than 15 values)

# ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']



dft1 = pd.get_dummies(dft1, columns=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'],\

                          prefix=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

                     ,drop_first=True

                     )



dfte1 = pd.get_dummies(dfte1, columns=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'],\

                          prefix=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

                       ,drop_first=True

                      )



 

    

# Date features

# date_cols = ['day', 'month']



dft1 = pd.get_dummies(dft1, columns=['day', 'month'],\

                          prefix=['day', 'month']

                     ,drop_first=True

                     )



dfte1 = pd.get_dummies(dfte1, columns=['day', 'month'],\

                          prefix=['day', 'month']

                       ,drop_first=True

                      )



#concat tables:

dft2 = pd.concat([train, dft1.reindex(train.index)], axis=1)

dfte2 = pd.concat([test, dfte1.reindex(test.index)], axis=1)
## Target Encoding:

train.sort_index(inplace=True)

train_y = train['target']; test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True); test.drop('id', axis=1, inplace=True)

from sklearn.metrics import roc_auc_score

cat_feat_to_encode = train.columns.tolist();  smoothing=0.20

#cat_feat_to_encode = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

smoothing=0.20

import category_encoders as ce

oof = pd.DataFrame([])

from sklearn.model_selection import StratifiedKFold

for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, train_y);  train = oof.sort_index(); test = ce_target_encoder.transform(test)
train.columns
train.rename(columns={0: "bin_0.1", 1: "bin_1.1", 2: "bin_2.1", 3: "bin_3.1", 4: "bin_4.1", 5: "nom_0.1", 6: "nom_1.1", 7: "nom_2.1", 8: "nom_3.1", 9:"nom_4.1", 10:"nom_5.1", 11:"nom_6.1", 12:"nom_7.1", 13:"nom_8.1", 14:"nom_9.1", 15:"ord_0.1", 16:"ord_1.1", 17:"ord_2.1", 18:"ord_3.1", 19:"ord_4.1", 20:"ord_5.1", 21:"day.1", 22:"month.1", 23:"missing_count.1"}, inplace=True)
test.columns
test.rename(columns={0: "bin_0.1", 1: "bin_1.1", 2: "bin_2.1", 3: "bin_3.1", 4: "bin_4.1", 5: "nom_0.1", 6: "nom_1.1", 7: "nom_2.1", 8: "nom_3.1", 9:"nom_4.1", 10:"nom_5.1", 11:"nom_6.1", 12:"nom_7.1", 13:"nom_8.1", 14:"nom_9.1", 15:"ord_0.1", 16:"ord_1.1", 17:"ord_2.1", 18:"ord_3.1", 19:"ord_4.1", 20:"ord_5.1", 21:"day.1", 22:"month.1", 23:"missing_count.1"}, inplace=True)
# Identify Highly Correlated Features:



# Create correlation matrix

corr_matrix = dft2.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

to_drop
dft2[['missing_count',

 'missing_count',

 'bin_0_1.0',

 'bin_1_1.0',

 'bin_2_0.0',

 'bin_2_1.0',

 'bin_4_N',

 'bin_4_Y',

 'nom_0_Blue']]
# Identify Highly Correlated Features:



# Create correlation matrix

corr_matrix = dfte2.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

to_drop
from IPython.display import display



pd.options.display.max_columns = None

display(dft2)
#Remove notneeded columns:

dft2.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

    #'bin_1_1.0', 'bin_2_1.0', 'bin_4_Y'

    #'missing_count', 'bin_0_1.0', 'bin_2_0.0', 

               # 'bin_3_F', 'bin_4_N', 'nom_0_Blue', 'target'

              ], axis=1, inplace=True)



dfte2.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

    #'bin_1_1.0', 'bin_2_1.0', 'bin_4_Y'

    #'missing_count', 'bin_0_1.0', 'bin_2_0.0', 

               # 'bin_3_F', 'bin_4_N', 'nom_0_Blue', 'target'

              ], axis=1, inplace=True)
['missing_count',

 'missing_count',

 'bin_0_1.0',

 'bin_1_1.0',

 'bin_2_0.0',

 'bin_2_1.0',

 'bin_3_F',

 'bin_4_N',

 'bin_4_Y',

 'nom_0_Blue']
dft2.columns
dft2['ord_0_2']=dft2['ord_0'].map(str)+dft2['ord_2'].map(str)
import statsmodels.api as sm

X2 = sm.add_constant(dff)

est = sm.OLS(dffte, X2)

est2 = est.fit()

print(est2.summary())
#  Random over-sampling:



# Class count

count_class_0, count_class_1 = dft2.target.value_counts()



# Divide by class

df_class_0 = dft2[dft2['target'] == 0]

df_class_1 = dft2[dft2['target'] == 1]



df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
train = df_test_over

train
#Model XGBoost:

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(train, train_y)
df_test_over.sort_index(inplace=True)

train_y = df_test_over['target']; test_id = dfte2['id']

df_test_over.drop(['target', 'id'], axis=1, inplace=True); dfte2.drop('id', axis=1, inplace=True)
# Standardizing the features

from sklearn.preprocessing import StandardScaler



m = StandardScaler().fit_transform(df_test_over)
# Make an instance of the Model



from sklearn.decomposition import PCA

pca = PCA(.95)
pca.fit(dft2)
pca.n_components_
dft2 = pca.transform(dft2)

dfte2 = pca.transform(dfte2)
#Model Logistic Regression:

from sklearn import linear_model

glm = linear_model.LogisticRegression( random_state=1, solver='lbfgs', max_iter=5000, fit_intercept=True, penalty='none', verbose=0); glm.fit(dft2, train_y)
# Submission:

pd.DataFrame({'id': test_id, 'target': glm.predict_proba(dfte2)[:,1]}).to_csv('submission_1.3.csv', index=False)