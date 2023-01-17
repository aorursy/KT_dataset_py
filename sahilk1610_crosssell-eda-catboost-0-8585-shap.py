

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from collections import Counter
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from tqdm import tqdm

train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
train.head()
#to check if there are any missing values in the dataset
train.isnull().sum()
train.info()
def metric(model, target):
    return roc_auc_score(target,model.predict_proba(X_valid)[:,1])


def split(df, y,number = 30000):
    df = df.sample(number)
    return df, y[df.index]

def feat_imp(df, model):
    return pd.DataFrame({'cols': df.columns, 'Imp': model.feature_importances_}).sort_values('Imp', ascending = False)

train['Response'].plot(kind = 'hist')
train.columns
cols = [ 'Age', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']


for i, j in enumerate(cols):
 
    sns.distplot(train[j])
    plt.show()
sns.countplot(train['Gender'])
df_g = train.groupby(['Gender', 'Response']).agg({'Response': 'count'}).unstack()
print(df_g)
df_g.plot(kind = 'bar', stacked = True)
plt.show()
plt.rcParams['figure.figsize'] = 12, 8
sns.boxplot(train['Gender'], train['Annual_Premium'], hue = train['Response'])
sns.countplot(train['Vehicle_Age'])
df2 = train.groupby(['Vehicle_Age', 'Response']).agg({'Response': 'count'}).unstack()
print(df2)
df2.plot(kind = 'bar', stacked =True)
plt.show()
sns.countplot(train['Vehicle_Damage'])
df3 = train.groupby(['Vehicle_Damage', 'Response']).agg({'Response': 'count'}).unstack()
print(df3)
df3.plot(kind = 'bar', stacked =True)
plt.show()

train['Region_Code'].nunique()
plt.rcParams['figure.figsize'] = 18, 8
df_rc = train.groupby(['Region_Code', 'Response']).agg({'Response': 'count'}).unstack()
df_rc.plot(kind = 'bar', stacked = True)
plt.show()
Counter(train['Driving_License'])
Counter(train['Previously_Insured'])
df_pi = train.groupby(['Previously_Insured', 'Response']).agg({'Response': "count"}).unstack()
df_pi.plot(kind = 'bar', stacked = True)
plt.show()

test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
test.head(2)
df_full = pd.concat([train.iloc[:,:-1], test])
df_full.shape
df_full['Age_bin_round'] = np.array(np.floor(
                              np.array(df_full['Age']) / 5.))
df_full[[ 'Age', 'Age_bin_round']].iloc[1071:1076]

df_full["Annual_Premium_log"] = np.log((df_full['Annual_Premium']))
df_full[['Annual_Premium', 'Annual_Premium_log']]
df_full_copy = df_full.copy()
Lb = LabelEncoder()
for i in df_full.columns:
    if df_full[i].dtype == 'object':
        df_full[i] = Lb.fit_transform(df_full[i])
df_full = df_full.drop('id', axis =1)
df_full.head()
X = df_full[:train.shape[0]]
y = train['Response']

X, y = split(X, y, 40000)
X.shape, y.shape

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
%%time
Rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=25, max_features=0.5, max_depth = 10, random_state=42)
model_Rf = Rf.fit(X_train, y_train)
print(metric(model_Rf, y_valid))
feat10 = feat_imp(X_train, model_Rf)
feat10
to_keep = feat10[feat10['Imp'] > 0.03].cols
len(to_keep)
df_to_keep = X[to_keep]
df_to_keep.shape
X_train, X_valid, y_train, y_valid = train_test_split(df_to_keep, y, test_size = 0.2, random_state = 42)
%%time
Rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=25, max_features=0.5, max_depth = 10, random_state=42)
model_Rf = Rf.fit(X_train, y_train)
print(metric(model_Rf, y_valid))
df_full_copy.info()
df_full_copy['Gender'] = df_full_copy['Gender'].astype('category')
df_full_copy['Driving_License'] = df_full_copy['Driving_License'].astype('category')
df_full_copy['Previously_Insured'] = df_full_copy['Previously_Insured'].astype('category')
df_full_copy['Vehicle_Age'] = df_full_copy['Vehicle_Age'].astype('category')
df_full_copy['Vehicle_Damage'] = df_full_copy['Vehicle_Damage'].astype('category')
df_full_copy['Region_Code'] = df_full_copy['Region_Code'].astype('int').astype('category')
df_full_copy['Policy_Sales_Channel'] = df_full_copy['Policy_Sales_Channel'].astype('int').astype('category')
cat_features = [ 'Gender', 'Region_Code', 'Vehicle_Age', 'Driving_License', 
                'Previously_Insured',  'Vehicle_Damage', 'Policy_Sales_Channel'
               ]

df_full_copy = df_full_copy.drop('id', axis = 1)
X = df_full_copy[:train.shape[0]]
y = train['Response']


#X, y = split(X, y , 40000)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf = CatBoostClassifier(
    iterations= 600, 
    learning_rate=0.08, 
    random_seed = 78,
    custom_loss=['AUC'],
    od_type = "Iter",
    depth= 11,
    #l2_leaf_reg= 3,
    bootstrap_type = 'Bernoulli',
)

clf.fit(X_train,y_train, 
        cat_features = cat_features, 
        verbose=False,
        eval_set = (X_valid, y_valid),
        plot = True,
        
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
print(metric(clf, y_valid))
import shap

# DF, based on which importance is checked
X_importance = X_valid

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance)
sns.heatmap(df_full_copy.corr(), annot=True)
