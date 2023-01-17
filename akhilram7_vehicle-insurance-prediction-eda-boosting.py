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
#Visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Model Selection and Metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, reset_parameter
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score, precision_score, auc, roc_curve

# Hyperparamter Tuning
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

train =pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train.head(500)
sns.set(style="darkgrid")
plt.style.use('fivethirtyeight')

plt.subplot(221)
colors = ['#1849CA', 'crimson']
plt.title('Insurance Clients based on Gender',fontsize=15)
circle = plt.Circle((0, 0), 0.6, color = 'white')
train['Gender'].value_counts().plot(kind='pie', figsize=(8, 8), rot=1, colors=colors, autopct = '%.2f%%')
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

plt.subplot(222)
colors = ['lightblue', 'crimson', 'pink', '#1849CA']
explode = [0, 0.075, 0, 0.075]
plt.title('Health Insuranced Clients',fontsize=15)
circle = plt.Circle((0, 0), 0.6, color = 'white')
health = train[['Gender','Previously_Insured']].values.tolist()
health = pd.DataFrame([h[0] + ' Insured' if h[1] == 1 else h[0] for h in health ],columns=['Gen_Ins'])
health['Gen_Ins'].value_counts().plot(kind='pie', explode=explode, figsize=(16, 16), rot=1, colors=colors, autopct = '%.2f%%')
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

plt.subplot(223)
colors = ['#1849CA', 'crimson']
plt.title('Vehicle damage based on Gender',fontsize=15)
circle = plt.Circle((0, 0), 0.6, color = 'white')
train['Vehicle_Damage'].value_counts().plot(kind='pie',figsize=(16, 16), rot=1, colors=colors, autopct = '%.2f%%')
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

plt.subplot(224)
colors = ['#1849CA', 'pink', 'lightblue', 'crimson']
explode = [0.075, 0, 0, 0.075]
plt.title('Vehicle Insuranced Clients',fontsize=15)
circle = plt.Circle((0, 0), 0.6, color = 'white')
health = train[['Gender','Vehicle_Damage']].values.tolist()
health = pd.DataFrame([h[0] + ' Insured' if h[1] == 'Yes' else h[0] for h in health ],columns=['Veh_Ins'])
health['Veh_Ins'].value_counts().plot(kind='pie', explode=explode, figsize=(16, 16), rot=1, colors=colors, autopct = '%.2f%%')
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()


plt.show()
plt.subplot(311)
plt.title('Age Distribution',fontsize=15)
men = train[train['Gender']=='Male']
women = train[train['Gender']=='Female']
a = sns.kdeplot(men['Age'], shade='True', legend='True', label='Male')
b = sns.kdeplot(women['Age'], shade='True', legend='True', label='Female')

plt.subplot(312)
plt.title('Health Insured Clients Distribution',fontsize=15)
health = train[['Gender','Previously_Insured','Age']]
health = health[health['Previously_Insured'] == 1]
men = health[health['Gender']=='Male']
women = health[health['Gender']=='Female']
a = sns.kdeplot(men['Age'], shade='True', legend='True', label='Male')
b = sns.kdeplot(women['Age'], shade='True', legend='True', label='Female')

plt.subplot(313)
plt.title('Vehicle Insured Clients Distribution',fontsize=15)
health = train[['Gender','Response','Age']]
health = health[health['Response'] == 1]
men = health[health['Gender']=='Male']
women = health[health['Gender']=='Female']
a = sns.kdeplot(men['Age'], shade='True', legend='True', label='Male')
b = sns.kdeplot(women['Age'], shade='True', legend='True', label='Female')

plt.tight_layout()
plt.show()
ax = sns.catplot(data=train, x='Vehicle_Age', hue='Response', col='Vehicle_Damage', kind='count')
plt.subplot(211)
plt.title('Effect of Annual Premium on Response',fontsize=15)
ax = sns.violinplot(data=train[train['Annual_Premium']<100000], y="Annual_Premium", x="Response")
plt.subplot(212)
plt.title('Effect of Vintage on Response',fontsize=15)
bx = sns.violinplot(data=train, y="Vintage", x="Response")
plt.tight_layout()
plt.show()
train['Gender_Code'] = pd.CategoricalIndex(train['Gender']).codes
train['Vehicle_Age_code'] = pd.CategoricalIndex(train['Vehicle_Age']).codes
train['Vehicle_Damage_code'] = pd.CategoricalIndex(train['Vehicle_Damage']).codes 
model_train = train[['Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                   'Policy_Sales_Channel', 'Gender_Code',
                   'Vehicle_Age_code', 'Vehicle_Damage_code']]

scaler = StandardScaler()

for param in ['Age',
              'Driving_License',
              'Region_Code',
              'Previously_Insured',
              'Policy_Sales_Channel',
              'Gender_Code',
              'Vehicle_Age_code',
              'Vehicle_Damage_code']:
    model_train[param] = scaler.fit_transform(model_train[param].values.reshape(-1, 1))
    
X_train, X_test, y_train, y_test = train_test_split(model_train, train['Response'], test_size = 0.2, shuffle = True)
model = {
    "Decision Tree": DecisionTreeClassifier(), 
    "SGD" : SGDClassifier(), 
    "Random Forest" : RandomForestClassifier(), 
    "Gradient Boosting" : GradientBoostingClassifier(),
    "XGBoost" : XGBClassifier(),
    "CatBoost" : CatBoostClassifier(),
    "LGBM" : LGBMClassifier()
        }

scores = []
prob_score = {}
for mod in model:
    classifier = model[mod]
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    try:
        score = classifier.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test, score, average='weighted')
        prob_score[mod] = score
    except:
        roc = 0
    scores.append([
        mod,
        accuracy_score(y_test, pred),
        f1_score(y_test, pred, average='weighted'),
        precision_score(y_test, pred, average='weighted'),
        recall_score(y_test, pred, average='weighted'),
        roc
    ])
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

scores_df  = pd.DataFrame(scores)
index_model = {count: s for count, s in enumerate(scores_df[0])}
col = {count+1: s for count, s in enumerate(['Accuracy','F1 Score','Precision','Recall','ROC AUC'])}
scores_df = scores_df.drop(0, axis=1)
scores_df = scores_df.rename(columns=col, index=index_model)
scores_df.style.apply(highlight_max)
plt.title('ROC Curves of Classifiers')
plt.xlabel('Precision')
plt.ylabel('Recall')

for key in prob_score:
    fpr, tpr, _ = roc_curve(y_test, prob_score[key])
    plt.plot(fpr, tpr, label=key)

plt.plot((0,1), ls='dashed',color='black')
plt.legend()
plt.show()
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}



clf = LGBMClassifier(max_depth=-1, random_state=15, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=100,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=15,
    verbose=True)


# Uncomment to perform Randomsearch
# gs.fit(X_train, y_train, **fit_params)
# print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

Randomsearch_params = {'colsample_bytree': 0.6261473679815167, 'min_child_samples': 237, 'min_child_weight': 0.001, 'num_leaves': 28, 'reg_alpha': 10, 'reg_lambda': 10, 'subsample': 0.7567691135431514} 
def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

#set optimal parameters
clf_sw = LGBMClassifier(**clf.get_params())
clf_sw.set_params(**Randomsearch_params)
clf_sw.fit(X_train,y_train, **fit_params, callbacks=[reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])