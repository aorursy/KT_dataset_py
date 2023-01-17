# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np

import missingno as msno
from sklearn.model_selection import train_test_split 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
train = pd.read_csv('../input/widsdatathon2020/training_v2.csv')
test = pd.read_csv('../input/widsdatathon2020/unlabeled.csv')
test.shape
y = train.hospital_death
y.fillna(0, inplace = True)
train = train.drop(['hospital_death'], axis = 1)
train = train.drop(['encounter_id', 'hospital_id', 'patient_id', 'icu_id'], axis = 1)

# Saving the Test Dataset's Encounter IDs in a separate list to be used to prepare the Output File Later

encounter_id = test.encounter_id
test = test.drop(['encounter_id', 'hospital_id', 'patient_id', 'icu_id'], axis = 1)
train.shape, test.shape

train = train.drop(['ethnicity', 'gender'], axis = 1)
test = test.drop(['ethnicity', 'gender'], axis = 1)
total = train.isnull().sum().sort_values(ascending=False)
percent_1 = train.isnull().sum()/train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head()
#Missing Values in Test
total = test.isnull().sum().sort_values(ascending=False)
percent_1 = test.isnull().sum()/test.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head()
categorical_objecttype = train.select_dtypes('object').columns  
non_cat = train.select_dtypes('float').apply(pd.Series.nunique, axis = 0).sort_values(ascending=True) 
non_cat = pd.DataFrame(non_cat, columns = ['Unique_Values'])
non_cat['colname'] = non_cat.index
non_cat = non_cat.reset_index()
non_cat.drop(['index'], axis =1, inplace = True)
categorical_floattype = non_cat.colname[:18]
len(categorical_objecttype), len(categorical_floattype)
for col in categorical_objecttype:
    train[col].fillna("missing", inplace = True)

for col in categorical_floattype:
    train[col].fillna("missing", inplace = True)
for col in categorical_objecttype:
    test[col].fillna("missing", inplace = True)
for col in categorical_floattype:
    test[col].fillna("missing", inplace = True)
train.shape, test.shape
traintest = pd.concat([train, test])
traintest = pd.get_dummies(traintest, prefix=categorical_objecttype, columns=categorical_objecttype)
traintest = pd.get_dummies(traintest, prefix=categorical_floattype, columns=categorical_floattype)
traintest.shape
import sklearn 
from sklearn.metrics import precision_score, recall_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
traintest = pd.concat([train, test])
object_types = np.where(traintest.dtypes == 'object')[0]
cols = traintest.columns
columns = []
for x in object_types:
    columns.append(cols[x])
for x in object_types:
    traintest.iloc[:,x] = traintest.iloc[:,x].astype(str)
traintest[columns] = traintest[columns].fillna("")
train = traintest[:train.shape[0]]
train.shape
test = traintest[train.shape[0]:]
test.shape
# Split into 80% training and 20% testing set
X_train, X_test, Y_train, Y_test = train_test_split(train, y, test_size = 0.2, random_state = 5, stratify = y)
cat_boost = CatBoostClassifier(learning_rate=0.05, iterations=200)
cat_boost.fit(X_train, Y_train, cat_features = columns,plot=True)
# Predict the test set labels
best_preds = cat_boost.predict(X_test)
train_preds = cat_boost.predict(X_train)
print("Train Accuracy = {}".format(accuracy_score(Y_train, train_preds)))
print("Test Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
test_pred = cat_boost.predict_proba(test)
test_pred = pd.DataFrame(test_pred, columns = ["Prob_Survival", "Pred_Death"])
test_pred.Prob_Survival = 1 - test_pred.Prob_Survival
final_dict = {'encounter_id' : encounter_id, 'hospital_death': test_pred.Prob_Survival}
Result = pd.DataFrame(final_dict)
Result.head()
Result.to_csv("0_CATBoost_Imputed.csv", index = False)
fea_imp = pd.DataFrame({'imp': cat_boost.feature_importances_, 'col': X_train.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');
fea_imp1 = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-20:]
fea_imp1.col


train_new = traintest[:train.shape[0]]

train_new.shape
red_train= train[fea_imp1.col.values]
red_test = test[fea_imp1.col.values]
categorical_objecttype_red = red_train.select_dtypes('object').columns  
categorical_objecttype_red
non_cat = red_train.select_dtypes('float').apply(pd.Series.nunique, axis = 0).sort_values(ascending=True) 
non_cat = pd.DataFrame(non_cat, columns = ['Unique_Values'])
non_cat['colname'] = non_cat.index
non_cat = non_cat.reset_index()
non_cat.drop(['index'], axis =1, inplace = True)
categorical_objecttype_red

red_train = red_train.fillna(red_train.mean())
red_test = red_test.fillna(red_train.mean())
red_train
red_train = pd.get_dummies(red_train, prefix=categorical_objecttype_red, columns=categorical_objecttype_red)
red_test = pd.get_dummies(red_test, prefix=categorical_objecttype_red, columns=categorical_objecttype_red)

#red_test = pd.get_dummies(red_test, prefix=categorical_objecttype_red, columns=categorical_objecttype_red)
red_train
import statsmodels.api as sm
logit_model=sm.Logit(y,red_train)
result=logit_model.fit()
print(result.summary2())
## private score: 0.89414 public : 0.89843
X_train, X_test, Y_train, Y_test = train_test_split(red_train, y, 
                                                    test_size=0.3, random_state=0, stratify=y)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
best_preds = logreg.predict_proba(X_test)[:,1]
train_preds = logreg.predict_proba(X_train)[:,1]
print("Train ROC-AUC Score = {}".format(roc_auc_score(Y_train, train_preds)))
print("Test ROC-AUC Score = {}".format(roc_auc_score(Y_test, best_preds)))
test_pred = logreg.predict_proba(red_test)
test_pred = pd.DataFrame(test_pred, columns = ["Prob_Survival", "Pred_Death"])
test_pred.Prob_Survival = 1 - test_pred.Prob_Survival
final_dict = {'encounter_id' : encounter_id, 'hospital_death': test_pred.Prob_Survival}
Result = pd.DataFrame(final_dict)
Result.to_csv("lronmeanimputation.csv", index = False)

del X_train['apache_4a_hospital_death_prob']
del X_train['apache_4a_icu_death_prob']

del X_test['apache_4a_hospital_death_prob']
del X_test['apache_4a_icu_death_prob']

X_train.shape, X_test.shape
cat_boost.fit(X_train, Y_train, cat_features = columns,plot=True)
# Predict the test set labels
best_preds = cat_boost.predict(X_test)
train_preds = cat_boost.predict(X_train)
print("Train Accuracy = {}".format(accuracy_score(Y_train, train_preds)))
print("Test Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
del test['apache_4a_hospital_death_prob']
del test['apache_4a_icu_death_prob']
test_pred = cat_boost.predict_proba(test)
test_pred = pd.DataFrame(test_pred, columns = ["Prob_Survival", "Pred_Death"])
test_pred.Prob_Survival = 1 - test_pred.Prob_Survival
final_dict = {'encounter_id' : encounter_id, 'hospital_death': test_pred.Prob_Survival}
Result = pd.DataFrame(final_dict)
Result.head()
Result.to_csv("1_CATBoost_Imputed_103Features.csv", index = False)
fea_imp = pd.DataFrame({'imp': cat_boost.feature_importances_, 'col': X_train.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');
# private score: 0.89319 public : 0.89749