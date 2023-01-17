# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualizatin
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
sns.set(style="ticks", context="talk")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import optuna
from optuna.samplers import TPESampler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
sample_sub = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')
print('Shape of train: {}'.format(train.shape))
print('Shape of test: {}'.format(test.shape))
train.head()
train['Response'] = le.fit_transform(train['Response'])
train['Response'].value_counts().plot.pie(autopct = '%1.1f%%',colors=['Orange','Blue'], figsize = (7,7))
train.isna().sum()/train.shape[0]*100
sns.countplot(train['Gender'], hue = train['Response'],palette=['Orange','Purple'])
f,ax = plt.subplots(nrows=2,ncols=1,figsize=(30,10))
axx = ax.flatten()
#plt.figure(figsize=(30,10))
sns.distplot(train['Age'],ax=axx[0], color='Blue')
sns.boxplot(train['Age'],ax=axx[1],color='Orange')
age_grp_20_to_30 = train[ train['Age'] <31]
age_grp_31_to_40 = train[ train['Age'].between(31,40)]
age_grp_41_to_50 = train[ train['Age'].between(41,50)]
age_grp_50_to_60 = train[ train['Age'].between(51,60)]
age_grp_old = train[ train['Age'] >60]

age_grp = [age_grp_20_to_30,age_grp_31_to_40,age_grp_41_to_50,age_grp_50_to_60,age_grp_old]
age_grp_name = ['age_grp_20_to_30','age_grp_31_to_40','age_grp_41_to_50','age_grp_50_to_60','age_grp_old']
age_grp_dict = dict(zip(age_grp_name, age_grp))
f,ax = plt.subplots(nrows=2, ncols=3, figsize = (20,10))
axx = ax.flatten()
for pos,tup in enumerate(age_grp_dict.items()):
    axx[pos].set_title(tup[0])
    data = tup[1]
    data['Response'].value_counts().plot.pie(autopct='%1.1f%%', ax = axx[pos],colors=['Red','Blue'])
f,ax = plt.subplots(nrows=2, ncols=3, figsize = (20,10))
axx = ax.flatten()
plt.title('Response Percentage of Different Age Groups with Genders',fontsize=40,x=-0.5,y=2.5)
for pos,tup in enumerate(age_grp_dict.items()):
    axx[pos].set_title(tup[0])
    temp = tup[1]
    temp.groupby('Gender')['Response'].value_counts().plot.pie(autopct='%1.1f%%', ax = axx[pos],colors=['Orange','Purple'])
sns.catplot(x = 'Gender', y="Age",hue = 'Response', data=train)
train['Driving_License'].value_counts().plot.pie(autopct='%1.1f%%',colors = ['Blue','Red'])
f,ax = plt.subplots(nrows=1,ncols=2,figsize = (20,5))
axx = ax.flatten()
#plt.title('Driving_License wise Response',fontsize=40,x=-0.5,y=2)
axx[0].set_title('Driving_Licence = 1')
axx[1].set_title('Driving_Licence = 0')
train[ train['Driving_License'] == 1]['Response'].value_counts().plot.pie(autopct='%1.1f%%',colors = ['Blue','Red'],ax=axx[0])
train[ train['Driving_License'] == 0]['Response'].value_counts().plot.pie(autopct='%1.1f%%',colors = ['Blue','Red'],ax=axx[1])
plt.figure(figsize = (40,10))
plt.title('Region Wise Response Count',fontsize=50)
sns.countplot(train['Region_Code'], hue = train['Response'],palette=['Red','Blue'])
u_region = train['Region_Code'].unique()
region_perc = {}
for i in u_region:
    total_region = train[ train['Region_Code'] == i].shape[0]
    buy_region = train[ (train['Region_Code'] == i) & train['Response'] == 1].shape[0]
    region_perc[i] = (buy_region/total_region)*100

region_perc = sorted(region_perc.items(), key=lambda x: x[1], reverse=True)
region_perc = list(zip(*region_perc))

region = np.array(region_perc[0])
region_perc = np.array(region_perc[1])
region = pd.DataFrame(region)
region_perc = pd.DataFrame(region_perc)

region_res_perc = pd.concat((region,region_perc), axis=1)
region_res_perc.columns = ['Region_Code', 'Buy_Percentage']
plt.figure(figsize=(40,10))
plt.title('Region Wise Buying Percentage',fontsize=50)
ax = sns.barplot(x = region_res_perc['Region_Code'], y = region_res_perc['Buy_Percentage'])
plt.figure(figsize=(15,5))
sns.countplot(train['Previously_Insured'],hue=train['Response'],palette=['Brown','Purple'])
plt.figure(figsize=(7,7))
train['Vehicle_Age'].value_counts().plot.pie(autopct='%1.1f%%', colors = ['r', 'b', 'g'])
plt.figure(figsize = (30,10))
sns.countplot(train['Vehicle_Age'], hue = train['Response'])
ls = train['Vehicle_Age'].unique()
f,ax = plt.subplots(nrows=1, ncols=3,figsize = (30,10))
axx = ax.flatten()
for pos,val in enumerate(ls):
    axx[pos].set_title(str(val))
    train[ train['Vehicle_Age'] == val]['Response'].value_counts().plot.pie(autopct = '%1.1f%%',ax = axx[pos], colors=['Purple', 'Orange'])
sns.countplot(train['Vehicle_Damage'], hue = train['Response'])
f,ax = plt.subplots(nrows=2,ncols=1,figsize=(30,20))
axx = ax.flatten()
#plt.figure(figsize=(30,10))
sns.distplot(train['Annual_Premium'],ax=axx[0], color='Blue')
sns.boxplot(train['Annual_Premium'],ax=axx[1],color='Orange')
plt.figure(figsize=(40,10))
sns.distplot(train[ train['Annual_Premium'] < 100000]['Annual_Premium'])#.plot.hist(bins = 500, frequency=(0,10000))
start = 0
step = 10000
ls = []
for _ in range(10):
    ls.append((start,step))
    start = step
    step+=10000

for tup in ls:
    count = train[ train['Annual_Premium'].between(tup[0],tup[1])].shape[0]
    percentage = train[ (train['Annual_Premium'].between(tup[0], tup[1])) & (train['Response'] == 1)].shape[0]/train[ train['Annual_Premium'].between(tup[0], tup[1])].shape[0]*100
    print('Number of Customers with Annual_Premium Between {} : {} and Insurance Buy Percentage:{}'.format(tup,count,percentage))
plt.figure(figsize=(40,10))
train['Policy_Sales_Channel'].value_counts().plot.bar()
f,ax = plt.subplots(nrows=2,ncols=1,figsize=(30,20))
axx = ax.flatten()
sns.distplot(train['Vintage'],ax=axx[0], color='Blue')
sns.boxplot(train['Vintage'],ax=axx[1],color='Orange')
train['is_train'] = 1
test['is_train'] = 0
test['Response'] = None

data = pd.concat((train,test))
data.set_index('id',inplace=True)
data.shape
sns.boxplot('Age', data=data, orient='v', color='Red')
sns.boxplot('Annual_Premium', data=data,orient='v', color='red')
f,ax = plt.subplots(nrows=1,ncols=2,figsize = (40,10))
axx = ax.flatten()
sns.kdeplot(data['Annual_Premium'], legend=False,ax = axx[0])
sns.kdeplot(np.log(data['Annual_Premium']), legend=False,ax = axx[1]) # after using log transformation
corr_check = data.copy()

col_ls = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

for col in col_ls:
    corr_check[col] = le.fit_transform(corr_check[col])
plt.figure(figsize=(20,10))
sns.heatmap(corr_check.corr(), annot=True, square=True,annot_kws={'size': 10})
train['Vehicle_Age']=train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
train['Gender']=train['Gender'].replace({'Male':1,'Female':0})
train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})

test['Vehicle_Age']=test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
test['Gender']=test['Gender'].replace({'Male':1,'Female':0})
test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})
# Changing Datatype
train['Region_Code']=train['Region_Code'].astype(int)
test['Region_Code']=test['Region_Code'].astype(int)
train['Policy_Sales_Channel']=train['Policy_Sales_Channel'].astype(int)
test['Policy_Sales_Channel']=test['Policy_Sales_Channel'].astype(int)
features=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
X=train[features]
y=train['Response']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=150303,stratify=y,shuffle=True)
catb = CatBoostClassifier()
catb= catb.fit(X_train, y_train,cat_features=cat_col,eval_set=(X_test, y_test),early_stopping_rounds=30,verbose=100)
y_pred = catb.predict(X_test)
proba = catb.predict_proba(X_test)[:, 1]
print('CatBoost Base Accuracy : {}'.format(accuracy_score(y_test,y_pred)))
print('CatBoost Base ROC_AUC_SCORE: {}'.format(roc_auc_score(y_test,proba)))
model = LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]
print('LGBM Base Accuracy : {}'.format(accuracy_score(y_test,y_pred)))
print('LGBM Base ROC_AUC_SCORE: {}'.format(roc_auc_score(y_test,proba)))
def create_model(trial):
    max_depth = trial.suggest_int("max_depth", 2, 30)
    n_estimators = trial.suggest_int("n_estimators", 1, 500)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
    num_leaves = trial.suggest_int("num_leaves", 2, 5000)
    min_child_samples = trial.suggest_int('min_child_samples', 3, 200)
    reg_alpha = trial.suggest_int("reg_alpha", 1, 10)
    reg_lambda = trial.suggest_int("reg_lambda", 1, 10)
    model = LGBMClassifier(
        learning_rate=learning_rate, 
        n_estimators=n_estimators, 
        max_depth=max_depth,
        num_leaves=num_leaves, 
        min_child_samples=min_child_samples,
        random_state=0
    )
    return model

sampler = TPESampler(seed=0)
def objective(trial):
    model = create_model(trial)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test,proba)
    return score

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)

lgb_params = study.best_params
lgb_params['random_state'] = 0
lgb = LGBMClassifier(**lgb_params)
lgb.fit(X_train, y_train)
proba = lgb.predict_proba(X_test)[:,1]
print('Optimized LightGBM roc_auc_score', roc_auc_score(y_test, proba))
lgb
LGBM = lgb
LGBM.fit(X, y)
y_pred = LGBM.predict(X_test)
proba = LGBM.predict_proba(X_test)[:,1]
print('LGBM Tuned Accuracy : {}'.format(accuracy_score(y_test,y_pred)))
print('LGBM Tuned ROC_AUC_SCORE: {}'.format(roc_auc_score(y_test,proba)))
LGBM_proba = LGBM.predict_proba(test[features])[:, 1] # Class 1 probability of LGBM model
cat_proba = catb.predict_proba(test[features])[:, 1] # Class 1 probability of CatBoost model
submit_proba = ((LGBM_proba * 0.45) + (cat_proba * 0.55))/2

sample_sub['Response'] = submit_proba

# sample_sub.to_csv() --- > Add your path here
