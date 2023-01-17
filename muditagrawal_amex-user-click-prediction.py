## importing libraries ##

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import StratifiedKFold, GridSearchCV , train_test_split

from tqdm import tqdm_notebook

import warnings

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import gc

import featuretools as ft

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('../input/train.csv', parse_dates= ['DateTime'])
test = pd.read_csv('../input/test.csv', parse_dates = ['DateTime'])
hist = pd.read_csv('../input/historical_user_logs.csv', parse_dates= ['DateTime'])
data.head()
hist.head()
data.info()
sns.heatmap(data.isnull())
# data imputation

data = data.drop('product_category_2', axis = 1) # dropping the column 

# for rest of the columns with missing values, imputing using forward fill.

data['city_development_index'] = data['city_development_index'].fillna(method = 'ffill') 

data['gender'] = data['gender'].fillna(method = 'ffill')

data['user_group_id'] = data['user_group_id'].fillna(method = 'ffill')

data['age_level'] = data['age_level'].fillna(method = 'ffill')

data['user_depth'] = data['user_depth'].fillna(method = 'ffill')
data.info()
day = data.groupby('DateTime')['is_click'].sum()

day = day.resample('H').sum()

plt.figure(figsize=(20,5))

day.plot(kind='bar',grid = None)
part_day = day.loc[slice('2017-07-02','2017-07-03')]

plt.figure(figsize=(20,5))

part_day.plot(kind='bar',grid = None)
data1 = data.reset_index()

data1['weekday'] = data1['DateTime'].dt.day_name()

byday  = pd.DataFrame(data1.groupby('weekday')['is_click'].sum())

byday = byday.reset_index()

plt.figure(figsize=(20,5))

sns.barplot(data = byday , x= 'weekday', y = 'is_click')
user = data.groupby(['gender','product'])['is_click'].sum()

user = pd.DataFrame(user.reset_index())

plt.figure(figsize=(20,5))

sns.barplot(data = user, x= 'product', y = 'is_click', hue = 'gender',palette='Set1')
n_data = data.reset_index()

campaign= pd.DataFrame(n_data.groupby(['campaign_id','product'])['is_click'].sum())

campaign= campaign.reset_index()

campaign= campaign.groupby(['product'])[['campaign_id','is_click']].max()

campaign= campaign.sort_values('is_click',ascending = False).reset_index()

campaign.columns = ['product', 'campaign_id', 'max click in any campaign']

plt.figure(figsize=(15,5))

sns.barplot(y= 'product', x= 'max click in any campaign', palette = 'Set1', data = campaign, orient='h')
n_data = data.reset_index()

campaign= pd.DataFrame(n_data.groupby(['campaign_id','product'])['is_click'].sum())

campaign= campaign.reset_index()

campaign= campaign.groupby('campaign_id')[['product','is_click']].max()

campaign.sort_values('is_click',ascending = False)
plt.figure(figsize=(20,5))

sns.countplot(x= 'user_group_id', hue= 'gender', palette = 'Set1', data = data)
plt.figure(figsize=(15,5))

user_group = data.groupby('user_group_id')['is_click'].agg(['count','sum'])

user_group['%success']= round((user_group['sum']*100)/user_group['count'], 2)

user_group = user_group.reset_index()

sns.barplot(y= 'user_group_id', x= '%success', data = user_group, palette = 'Set1', order = user_group['%success'])
plt.figure(figsize=(15,3))

sns.countplot(x="product", hue= "is_click", palette = 'Set1', data =data )
plt.figure(figsize=(15,5))

sns.countplot(x="product", hue= "product_category_1", palette = 'Set1', data =data)
data1 = data[['user_depth', 'is_click']]

data1 = data.groupby(['user_depth','is_click']).size().unstack()

data1['success %'] = round(data1[1]*100/(data1[1]+data1[0]),2)

data1
print(data['is_click'].value_counts())

print(round(30057*100/(414991),2))  
data['weekday']=data['DateTime'].dt.day_name()

data['hour'] = data['DateTime'].dt.hour

data['minutes'] = data['DateTime'].dt.minute

data = data.drop(['DateTime','session_id'], axis = 1)

data.head()
es1 = ft.EntitySet()
es1 = es1.entity_from_dataframe(entity_id= 'hist', 

                                dataframe= hist,

                                make_index = True,

                                index = 'id',

                                time_index = 'DateTime',

                                variable_types={"user_id": ft.variable_types.Categorical})

                                       
es1['hist'].variables
es1 = es1.entity_from_dataframe(entity_id = 'data', 

                                dataframe= data, 

                                make_index= True, 

                                index = 'id',

                                variable_types={"user_id": ft.variable_types.Categorical, 

                                                'webpage_id': ft.variable_types.Categorical,

                                                'campaign_id': ft.variable_types.Categorical,

                                                'product_category_1': ft.variable_types.Categorical,

                                                'user_group_id':  ft.variable_types.Categorical,

                                                'age_level': ft.variable_types.Categorical,

                                                'user_depth': ft.variable_types.Categorical,

                                                'city_development_index': ft.variable_types.Categorical,

                                                'var_1': ft.variable_types.Categorical ,

                                                'is_click': ft.variable_types.Categorical

                                               })          
es1['data'].variables
relation = ft.Relationship(es1['data']['id'], es1['hist']['id'])
es1 = es1.add_relationship(relation)

es1
features, feature_names = ft.dfs(entityset= es1, 

                                 target_entity= 'data', 

                                  max_depth = 2

                                 )
features.info()
col =['product','gender','weekday','webpage_id','campaign_id','product_category_1','user_group_id','age_level','user_depth','var_1','city_development_index','MODE(hist.product)','MODE(hist.action)']

new_data = pd.get_dummies(features, columns = col, drop_first= True)
new_data.info()
X = new_data.drop('is_click', axis=1)

y = new_data['is_click']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3,  stratify = y,random_state = 101)
train = pd.concat([X_train, y_train], axis = 1)
t_0 = train[train['is_click'] == 0]

t_1 = train[train['is_click'] == 1]
t0_sub = t_0.sample(n = 27573, random_state= 101)

t1_sub = t_1.sample(n = 2000, random_state= 101)

train_sub = pd.concat([t0_sub,t1_sub], axis = 0)

train_sub = train_sub.sample(frac=1, random_state= 101)
Xtr_sub = train_sub.drop('is_click', axis=1)

ytr_sub = train_sub['is_click']
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 101)

X_sub, y_sub  = sm.fit_sample(Xtr_sub,ytr_sub)
X_sub.shape, y_sub.shape
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

rfe = RFE(model, n_features_to_select=75, verbose= 1)

fit = rfe.fit(X_sub,y_sub)
sum(fit.support_)
X_sub = pd.DataFrame(X_sub, columns= Xtr_sub.columns)

X_sub = np.array(X_sub.loc[:,fit.support_])

X_sub.shape
type(X_sub)
y_pred = fit.predict(X_test)

matrix =classification_report(y_test,y_pred)

print(matrix)
confusion_matrix(y_test,y_pred)
roc_auc_score(y_test,y_pred)
def best_model(estimator,grid, refit_score, scorer):

    grid_search = GridSearchCV(estimator, param_grid=grid, scoring= scorer, refit= refit_score, cv = skf,n_jobs= -1)

    grid_search.fit(X_sub, y_sub)

    

    pred = grid_search.predict(X_test)

    

    print('Best params for {}'.format(refit_score))

    print(grid_search.best_params_)

    

    print(pd.DataFrame(confusion_matrix(y_test, pred),columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

    

    print('roc-auc : %0.2f'  % roc_auc_score(y_test, pred))

    return grid_search
skf = StratifiedKFold(n_splits=5, random_state=101)

scorers = ['recall']

X_test = np.array(X_test.loc[:,fit.support_])
from sklearn.tree import DecisionTreeClassifier
dctree = DecisionTreeClassifier()
para_grid = {

    'criterion': ['entropy', 'gini'],

    'min_samples_split': [2],

    'max_depth': [30,35,40],

    'max_features': [20, 25,27]

}
best_model(estimator= dctree, grid= para_grid, refit_score= 'recall', scorer= scorers)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
para_grid = {

    'min_samples_split': [2], 

    'n_estimators' : [300],

    'max_depth': [25],

    'max_features': [40, 45],

}
best_model(estimator= rf, grid= para_grid, refit_score= 'recall', scorer= scorers)