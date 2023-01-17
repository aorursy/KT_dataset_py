# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/inputnovartis/train_data.csv')
df_test = pd.read_csv('/kaggle/input/inputnovartis/test_data.csv')
all_data = pd.concat([df_train, df_test], axis=0)
dummy = df_train
percent_missing = df_train.isnull().sum() * 100 / len(df_train)
missing_value_df_test = pd.DataFrame({'column_name': df_train.columns,
                                 'percent_missing': percent_missing})
missing_value_df_test.sort_values(by=['percent_missing'],ascending=False)
percent_missing = df_test.isnull().sum() * 100 / len(df_test)
missing_value_df = pd.DataFrame({'column_name': df_test.columns,
                                 'percent_missing': percent_missing})
missing_value_df
plt.figure(figsize = (7,10))
sns.boxplot(df_train.loyaltyClass, df_train.creditRisk)
plt.figure(figsize = (7,10))
sns.boxplot(df_train.loyaltyClass, df_train.staySpend)
sns.countplot(df_train['loyaltyClass'])
plt.hist(all_data['age']);
plt.title('Distribution of age variable');
plt.hist(all_data['creditRisk']);
plt.title('Distribution of creditRisk variable');
plt.hist(all_data['staySpend']);
plt.title('Distribution of staySpend variable');

plt.hist(all_data['pastStays1y']);
plt.title('Distribution of pastStays1y variable');
ct = pd.crosstab(df_train.profession, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.gender, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.highestDegree, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.maritalStatus, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.otherMembership, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.blogger, df_train.loyaltyClass, normalize = 'index')
ct
ct = pd.crosstab(df_train.amexCard, df_train.loyaltyClass, normalize = 'index')
ct

ct = pd.crosstab(df_train.purposeTravel, df_train.loyaltyClass, normalize = 'index')
ct
for i in ['profession', 'age', 'gender', 'noOfKids', 'pastStays1y']:
    df_train[i].fillna('NaN', inplace=True)
    df_test[i].fillna('NaN', inplace=True)
Y = dummy['loyaltyClass'].apply(lambda x: 'Gold' if x =='Gold' else 'NoGold')
Y
X_train = df_train.drop(['loyaltyClass','ID','age', 'noOfKids'], axis=1).values
X_test = df_test.drop(['ID','age', 'noOfKids'],axis = 1).values

X_train.shape, Y.shape, X_test.shape
X_train[0]
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
preds = pd.DataFrame()
kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()
i = 0
for train, test in kfold.split(X_train):
    x_train, x_test = X_train[train], X_train[test]
    y_train, y_test = Y[train], Y[test]
    eval_pool = Pool(x_test, y_test,cat_features=[0,1,2,3,5,7,9,10])
    
    
    model = CatBoostClassifier(random_state=27, max_depth=4,loss_function='Logloss', devices="0:1", n_estimators=1000, verbose=100,eval_metric='AUC')
    model.fit(x_train, y_train,cat_features=[0,1,2,3,5,7,9,10],eval_set=eval_pool,early_stopping_rounds=25)
    preds = model.predict(x_test)
    score = accuracy_score(list(preds.flatten()),list(y_test), normalize=False) / len(list(y_test))
    print(classification_report(list(preds.flatten()),list(y_test)))
    scores.append(score)
    print(score)
    i=i+1
#print("Average: ", sum(scores)/len(scores))
feature_importance_df = pd.DataFrame()
features = ['Profession', 'gender', 'highestDegree', 'maritalStatus', 'creditRisk', 'otherMembership', 'pastStays1y'
           ,'blogger', 'articles', 'amexCard', 'purposeTravel', 'staySpend']
feature_importance_df["Feature"] = features
feature_importance_df["importance"] = model.feature_importances_
feature_importance_df.sort_values(by=['importance'],ascending=False)
preds = model.predict_proba(X_test)
#y = 1 if preds[:,0] > 0.3 else y = 0
#output = preds[:,0].apply(lambda x: 1 if x > 0.35 else 0)
output = []
for i in preds[:,0]:
    if(i > 0.285):
        output.append("Gold")
    else:
        output.append("NoGold")
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(output,list(y_test)))
preds = model.predict_proba(X_test)
f_output = []
for i in preds[:,0]:
    if(i > 0.285):
        f_output.append("Gold")
    else:
        f_output.append("NoGold")
len(f_output)
Labeled_df_test = df_test
Labeled_df_test['label'] = f_output
new_test = Labeled_df_test[Labeled_df_test['label']=="NoGold"]
df_train['loyaltyClass']
new_train = df_train[df_train['loyaltyClass'] !='Gold']
new_train.shape
X_train,Y = new_train.drop(['loyaltyClass','ID','age', 'noOfKids'], axis=1).values, new_train['loyaltyClass'].values
X_test = new_test.drop(['ID','age', 'noOfKids','label'],axis = 1).values

X_train.shape, Y.shape, X_test.shape
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
preds = pd.DataFrame()
kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()
i = 0
for train, test in kfold.split(X_train):
    x_train, x_test = X_train[train], X_train[test]
    y_train, y_test = Y[train], Y[test]
    eval_pool = Pool(x_test, y_test,cat_features=[0,1,2,3,5,7,9,10])
    
    
    model = CatBoostClassifier(random_state=27, max_depth=4,loss_function='Logloss', devices="0:1", n_estimators=1000, verbose=100,eval_metric='AUC')
    model.fit(x_train, y_train,cat_features=[0,1,2,3,5,7,9,10],eval_set=eval_pool,early_stopping_rounds=25)
    preds = model.predict_proba(x_test)
    #score = accuracy_score(list(preds.flatten()),list(y_test), normalize=False) / len(list(y_test))
    #print(classification_report(list(preds.flatten()),list(y_test)))
    #scores.append(score)
    #print(score)
    #i=i+1
#print("Average: ", sum(scores)/len(scores))
preds = model.predict(X_test)
preds
new_test['label_2'] = preds
final_label = []
j=0
check = 0
for i in Labeled_df_test['label']:
    if i=='Gold':
        final_label.append("Gold")
        check=check+1
    else:
        final_label.append(preds[j])
        j = j+1
sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['PredictedClass'] = final_label
sub.to_csv('Ensemble_Submission.csv',index=False)
