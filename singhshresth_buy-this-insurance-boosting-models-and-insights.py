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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,f1_score
sns.set(style='whitegrid')
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
submission = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train.head()
train.describe()
plt.figure(figsize = (12,8))
sns.heatmap(train.corr(),annot=True, cmap='viridis')
sns.countplot(train['Response'])
sns.boxplot(x='Response',y='Vintage',data=train)
sns.boxplot(x='Response',y='Age',data=train)
plt.figure(figsize=(8,6))
sns.distplot(train['Policy_Sales_Channel'], bins=10)
plt.figure(figsize=(8,6))
sns.distplot(train['Age'])
plt.figure(figsize=(8,6))
#train['Region_Code'].hist()
sns.distplot(train['Region_Code'], bins=8)
pd.crosstab(train['Response'],train['Previously_Insured'])
train['source'] = 'train'
test['source'] = 'test'
df = pd.concat([train,test])


df['Policy_Region'] = df['Policy_Sales_Channel'].astype(str) + '_' + df['Region_Code'].astype(str)
df['Vehicle_License'] = df['Vehicle_Age'].astype(str) + '_' + df['Driving_License'].astype(str)
df.head()
from sklearn.preprocessing import LabelEncoder
cat_cols = ['Gender','Driving_License','Region_Code','Previously_Insured',
                'Vehicle_Damage','Policy_Sales_Channel','Policy_Region',
                'Vehicle_Age','Vintage','Annual_Premium','Vehicle_License']
label = 'Response'

def categorical_encoding(data, cat_cols):
    label_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].unique().tolist())
        df[col] = le.transform(df[col])
        label_dict[col] = le
    le = LabelEncoder()
    df[label] = le.fit_transform(df[[label]])
    label_dict[label] = le
    return df, label_dict
df, label_dict = categorical_encoding(df, cat_cols)
from sklearn.preprocessing import KBinsDiscretizer
premium_discretizer = KBinsDiscretizer(n_bins = 8, encode = 'ordinal', strategy = 'quantile')
df['Premium_bins'] = premium_discretizer.fit_transform(df['Annual_Premium'].values.reshape(-1,1)).astype(int)

age_discretizer = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile')
df['Age_bins'] = age_discretizer.fit_transform(df['Age'].values.reshape(-1,1)).astype(int)
gender_counts = df['Gender'].value_counts().to_dict()
df['Gender_Count'] = df['Gender'].map(gender_counts)

previous_insured_counts = df['Previously_Insured'].value_counts().to_dict()
df['Pre_Insured_Counts'] = df['Previously_Insured'].map(previous_insured_counts)

vehicle_age_counts = df['Vehicle_Age'].value_counts().to_dict()
df['vehicle_counts_age'] = df['Vehicle_Age'].map(vehicle_age_counts)

vehicle_dam_count = df['Vehicle_Damage'].value_counts().to_dict()
df['Vehicle_Damage_Count'] = df['Vehicle_Damage'].map(vehicle_dam_count)

df['Policy_Per_Region'] = df.groupby('Region_Code')['Policy_Sales_Channel'].transform('nunique')
df['Policy_Per_Region_Sum'] = df.groupby('Region_Code')['Policy_Sales_Channel'].transform('sum')

df['Vintage'] = df['Vintage'] / 365 
#df['Previous_Insure_Region'] = df.groupby('Region_Code')['Previously_Insured'].transform('sum')
df['Premium_Per_Region'] = df.groupby('Region_Code')['Annual_Premium'].transform('sum')
df['Premium_Per_Policy'] = df.groupby('Policy_Sales_Channel')['Annual_Premium'].transform('sum')
df['Policy_Per_Premium_Bin'] = df.groupby('Premium_bins')['Policy_Sales_Channel'].transform('nunique')
df['Premium_Per_Age_Bin'] = df.groupby('Age_bins')['Annual_Premium'].transform('mean')
df['Mean_Premium_Per_Region'] = df.groupby('Region_Code')['Annual_Premium'].transform('mean')
float_col = df.select_dtypes(include=['float'])
for col in float_col:
  df[col] = df[col].astype('int64')
final_train = df[df['source']=='train']
target = final_train['Response']
final_train = final_train.drop(columns=['id', 'source', 'Response'])
final_test = df[df['source']=='test']
final_test_id = final_test['id']
final_test = final_test.drop(columns=['id', 'source', 'Response'])
from sklearn.model_selection import train_test_split, KFold,  StratifiedShuffleSplit
X = final_train
y = train['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=150303,stratify=y,shuffle=True)
# XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
probs = np.zeros(shape=(len(final_test)))
scores = []
avg_loss = []

X_train, y_train = final_train, target
seeds = [1]

for seed in range(len(seeds)):
    print(' ')
    print('#'*100)
    print('Seed', seeds[seed])
    sf = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
    for i, (idxT, idxV) in enumerate(sf.split(X_train, y_train)):
        print('Fold', i)
        print('Rows of Train= ', len(idxT), 'Rows of Holdout = ', len(idxV))
        clf = XGBClassifier(n_estimators=100000,
                           max_depth=7,
                           min_child_weight = 5,
                           learning_rate=0.03,
                            subsample=0.7,
                            colsample_bytree=0.8,
                            gamma=0.2,
                            scale_pos_weight = 1,
                            objective='binary:logistic',
                            random_state=1)
        preds = clf.fit(X_train.iloc[idxT], y_train.iloc[idxT],
                       eval_set=[(X_train.iloc[idxV], y_train.iloc[idxV])],
                       verbose=100, eval_metric=['auc', 'logloss'],
                       early_stopping_rounds=40)
        probs_oof = clf.predict_proba(X_train.iloc[idxV])[:,1]
        probs += clf.predict_proba(final_test)[:,1]
        roc = roc_auc_score(y_train.iloc[idxV], probs_oof)
        scores.append(roc)
        avg_loss.append(clf.best_score)
        print("ROC_AUC= ", roc)
        print('#'*100)
        
print("Loss= {0:0.5f}, {1:0.5f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))
print('%.6f (%.6f)' % (np.array(scores).mean(), np.array(scores).std()))
p1 = probs / 5
p1
from catboost import CatBoostClassifier
model = CatBoostClassifier()
probs = np.zeros(shape=(len(final_test)))
scores = []
avg_loss = []

X_train, y_train = final_train, target
seeds = [1]
model = model.fit(X_train, y_train,cat_features=cat_cols,eval_set=(X_test, y_test),plot=True,early_stopping_rounds=40,verbose=100)
y_pred = model.predict(X_test)
probs_cat_train = model.predict_proba(X_train)[:, 1]
probs_cat_test = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_train, probs_cat_train)
roc_auc_score(y_test, probs_cat_test)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(15).plot(kind='barh')
plt.show()
cat_pred= model.predict_proba(final_test)[:, 1]
#submission['Response'] = cat_pred
submission['Response'] = 0.75 * cat_pred + 0.25 * p1
submission
submission.to_csv("result.csv", index=False)