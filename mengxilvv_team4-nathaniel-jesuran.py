data_dir = 'input/'
train_file = 'train_set_2.csv'
test_file = 'test_set_2.csv'
target_col = 'outcome'
random_state = 43
generate_predictions = False
skip_scaling = True
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
### Add features: 'total_campaign_creator', 'win_campaign_creator', win_rate_creator', 'day_since_last_attempt'
def get_creator_features(df_train,df_test):
    temp_train = df_train[['creator_id','id','launched_at','outcome']]
    temp_train['class'] = 'train' 
    temp_test = df_test[['creator_id','launched_at','id']]
    temp_test['class'] = 'test' 
    temp_test['outcome'] = 0
    
    temp_df = pd.concat([temp_train,temp_test])
    
    temp_df['launched_at_date'] = pd.to_datetime(temp_df['launched_at'])
    temp_df = temp_df.sort_values(by=['launched_at_date'])
    temp_df["rank"] = temp_df.groupby('creator_id')['launched_at_date'].rank("dense", ascending=True)
    temp_df_backup = temp_df.copy()
    
    part_df = []
    i=2
    while i<temp_df_backup['rank'].max():
        temp_df = temp_df_backup[temp_df_backup['rank']<i]
        ## 'total_campaign_creator', 'win_campaign_creator'
        funcdict={'id':'nunique','outcome':'sum'}
        namemap={'id':'total_campaign_creator','outcome':'win_campaign_creator'}
        df_creator_feature = temp_df.groupby(['creator_id']).agg(funcdict).reset_index()
        df_creator_feature.rename(columns = namemap,inplace=True)
        
        ## 'win_rate_creator'
        funcdict={'id':'nunique'}
        namemap={'id':'total2_campaign_creator'}
        df_creator_feature2 = temp_df[temp_df['class']=='train'].groupby(['creator_id']).agg(funcdict).reset_index()
        df_creator_feature2.rename(columns = namemap,inplace=True)
        
        df_creator_feature = df_creator_feature.merge(df_creator_feature2, on = 'creator_id', how = 'left')
        df_creator_feature['total2_campaign_creator'] = df_creator_feature['total2_campaign_creator'].fillna(1)
        df_creator_feature['win_rate_creator'] = df_creator_feature['win_campaign_creator']/df_creator_feature['total2_campaign_creator']
        df_creator_feature = df_creator_feature[['creator_id','total_campaign_creator','win_campaign_creator','win_rate_creator']]
        df_creator_feature['rank'] = i
        part_df.append(df_creator_feature)
        i = i+1
    df_creator_feature = pd.concat(part_df)
    
    temp_df = temp_df_backup
    ## 'day_since_last_attempt'
    temp_df['launched_at_date'] = pd.to_datetime(temp_df['launched_at'])
    temp_df = temp_df.sort_values(by=['launched_at_date'])
    temp_df['last_launched_at_date'] = temp_df.groupby(['creator_id'])['launched_at_date'].shift(1)
    temp_df['day_since_last_attempt'] = (pd.to_datetime(temp_df['launched_at_date'])-pd.to_datetime(temp_df['last_launched_at_date'])).dt.days
    
    #df_creator_feature3 = temp_df.drop_duplicates(subset=['creator_id'],keep ='last')
    df_creator_feature3 = temp_df[['creator_id','launched_at','day_since_last_attempt','rank']]
    df_creator_feature3 = df_creator_feature3.merge(df_creator_feature, on = ['creator_id','rank'], how = 'left')
    df_creator_feature3.drop(columns=['rank'],inplace=True)
    df_creator_feature3[['total_campaign_creator','win_campaign_creator','win_rate_creator']]=df_creator_feature3[['total_campaign_creator','win_campaign_creator','win_rate_creator']].fillna(0)
    
    ## Add features to train_set, test_set
    #df_train = df_train.merge(df_creator_feature, on = 'creator_id', how = 'left')
    df_train = df_train.merge(df_creator_feature3, on = ['creator_id','launched_at'], how = 'left')
    #df_test = df_test.merge(df_creator_feature, on = 'creator_id', how = 'left')
    df_test = df_test.merge(df_creator_feature3, on = ['creator_id','launched_at'], how = 'left')
    
    
    
    return df_train, df_test
    #return df_creator_feature3
def launch_hour_bins(launch_hour):
    '''
    takes in launch_hour column and outputs three one-hot encoded vectors.
    Base-case is hours 0-5.
    '''
    quartiles = pd.qcut(launch_hour, q = 4, labels = [0,1,2,3])
    one_hot_hours = pd.get_dummies(quartiles, drop_first = True)
    one_hot_hours.columns = ['launch_hour_6_11', 'launch_hour_12_17', 'launch_hour_18_23']
    return one_hot_hours

def deadline_hour_bins(deadline_hour):
    '''
    takes in deadline_hour column and outputs three one-hot encoded vectors.
    Base-case is hours 0-5.
    '''
    quartiles = pd.qcut(deadline_hour, q = 4, labels = [0,1,2,3])
    one_hot_hours = pd.get_dummies(quartiles, drop_first = True)
    one_hot_hours.columns = ['deadline_hour_6_11', 'deadline_hour_12_17', 'deadline_hour_18_23']
    return one_hot_hours
def funding_duration(deadline_datetime, launch_datetime):
    return [delta.days for delta in (deadline_datetime - launch_datetime)]
def kickstarter_ngram_freq(launch_year):
    '''
    Google ngram frequencies for "Kickstarter+kickstarter" scaled by 10e6 
    '''
    mapping = {2009: 2.8384, 
               2010: 5.7698, 
               2011: 8.3990, 
               2012: 11.2500, 
               2013: 15.7038, 
               2014: 19.0103, 
               2015: 21.9546, 
               2016: 24.5102, 
               2017: 26.2639}
    freqs = launch_year.apply(lambda x: mapping[x])
    return freqs
def aggregate_df(df, sorted=False):
    if sorted:
        return df.agg(['dtype', 'nunique', 'count']).T.sort_index()
    return df.agg(['dtype', 'nunique', 'count']).T
def get_features_time(df0):
    df = copy.deepcopy(df0)
    from calendar import monthrange
    import holidays
    us_holidays = holidays.UnitedStates()
    #note: next lines already in dataset
    #df['create2launch']=(pd.to_datetime(df['launched_at'])-pd.to_datetime(df['created_at'])).dt.days 
    #df['launch2deadline']=(pd.to_datetime(df['deadline'])-pd.to_datetime(df['launched_at'])).dt.days
    '''
    time related features
    '''
    df['created_at']=df['created_at'].astype('datetime64')
    df['deadline']=df['deadline'].astype('datetime64')
    df['launched_at']=df['launched_at'].astype('datetime64')
    df['create_hour'] = df['created_at'].dt.hour
    df['create_day'] = df['created_at'].dt.day
    df['create_month'] = df['created_at'].dt.month
    df['create_year'] = df['created_at'].dt.year
    df['create_weekday'] = df['created_at'].dt.weekday
    df['create_quarter']=(df['created_at'].dt.month-1)//3+1
    df['create_last7d_month']=[d>monthrange(y,m)[1]-7 
                               for y,m,d 
                               in zip(df['created_at'].dt.year,
                                      df['created_at'].dt.month,
                                      df['created_at'].dt.day)]
    df['create_day_in_qr']=((df['created_at'].dt.month-1)%3)*30+df['created_at'].dt.day
    df['create_holidays']=df['created_at'].isin(us_holidays)
    df['launched_hour'] = df['launched_at'].dt.hour
    df['launched_day'] = df['launched_at'].dt.day
    df['launched_month'] = df['launched_at'].dt.month
    df['launched_year'] = df['launched_at'].dt.year
    df['launched_weekday'] = df['launched_at'].dt.weekday
    df['launched_quarter']=(df['launched_at'].dt.month-1)//3+1
    df['launched_last7d_month']=[d>monthrange(y,m)[1]-7 
                               for y,m,d 
                               in zip(df['launched_at'].dt.year,
                                      df['launched_at'].dt.month,
                                      df['launched_at'].dt.day)]
    df['launched_day_in_qr']=((df['launched_at'].dt.month-1)%3)*30+df['launched_at'].dt.day
    df['launched_holidays']=df['launched_at'].isin(us_holidays)
    df['deadline_hour'] = df['deadline'].dt.hour
    df['deadline_day'] = df['deadline'].dt.day
    df['deadline_month'] = df['deadline'].dt.month
    df['deadline_year'] = df['deadline'].dt.year
    df['deadline_weekday'] = df['deadline'].dt.weekday
    df['deadline_quarter']=(df['deadline'].dt.month-1)//3+1
    df['deadline_last7d_month']=[d>monthrange(y,m)[1]-7 
                               for y,m,d 
                               in zip(df['deadline'].dt.year,
                                      df['deadline'].dt.month,
                                      df['deadline'].dt.day)]
    df['deadline_day_in_qr']=((df['deadline'].dt.month-1)%3)*30+df['deadline'].dt.day
    df['deadline_holidays']=df['deadline'].isin(us_holidays)
    ''' 
    $/ day
    '''
    df['goal/day']=df['goal']/df['duration']
    #df['staff_pick'] = df['staff_pick'].astype(int)
    #df['disable_communication'] = df['disable_communication'].astype(int)
#     if len(features) < 1:
#         return df
    return df
features = [
    'funding_duration',
    'goal', 
#     'create2launch', 'launch2deadline',
    'launch_hour_6_11', 'launch_hour_12_17', 'launch_hour_18_23',
    'deadline_hour_6_11', 'deadline_hour_12_17', 'deadline_hour_18_23',
    'kickstarter_ngram_freq',
    'staff_pick', 'disable_communication', 
    'create_hour',
#     'create_month', 'create_year', 
#     'create_weekday', 'create_quarter', - 
    'create_last7d_month', 'create_day_in_qr', 
    'create_holidays',
#     'launched_hour', 'launched_month', 'launched_year',
#     'launched_weekday', 'launched_quarter', 
    'launched_last7d_month',
    'launched_day_in_qr', 
#     'launched_holidays', 'deadline_weekday',
#     'deadline_quarter', 
    'deadline_last7d_month', 'deadline_day_in_qr',
    'deadline_holidays', 'goal/day'
#     'create_hour', 'create_day', 'create_month', 'create_year',
#     'launch_hour', 'launch_day', 'launch_month', 'launch_year',
#     'deadline_hour', 'deadline_day', 'deadline_month', 'deadline_year'
]
def get_features(df, features=[]):
    #df['launch_datetime'] = pd.to_datetime(df.launch_year*10000+df.launch_month*100+df.launch_day,format='%Y%m%d')
    #df['deadline_datetime'] = pd.to_datetime(df.deadline_year*10000+df.deadline_month*100+df.deadline_day,format='%Y%m%d')
    #df['funding_duration'] = funding_duration(df['deadline_datetime'], df['launch_datetime'])
    #df = pd.concat([df, launch_hour_bins(df['launch_hour'])], axis=1)
    #df = pd.concat([df, deadline_hour_bins(df['deadline_hour'])], axis=1)
    #df.drop(columns=['launch_hour', 'deadline_hour'])
    
    df['kickstarter_ngram_freq'] = kickstarter_ngram_freq(df['launch_year'])
    
    df=get_features_time(df)
    
#     df['create2launch']=(pd.to_datetime(df['launched_at'])-pd.to_datetime(df['created_at'])).dt.days
#     df['launch2deadline']=(pd.to_datetime(df['deadline'])-pd.to_datetime(df['launched_at'])).dt.days

    df['staff_pick'] = df['staff_pick'].astype(int)
    df['disable_communication'] = df['disable_communication'].astype(int)

#     df['create_hour'] = pd.to_datetime(df['created_at']).dt.hour
#     df['create_day'] = pd.to_datetime(df['created_at']).dt.day
#     df['create_month'] = pd.to_datetime(df['created_at']).dt.month
#     df['create_year'] = pd.to_datetime(df['created_at']).dt.year
    
#     df['NLP'] = df['blurb'].apply(lambda x: do_something(x), axis=1)
    if len(features) < 1:
        return df
    return df[features]
from sklearn.impute import SimpleImputer

def get_imputer(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='median', fill_value=None)
    imp.fit(x)
    return imp
def apply_imputer(f, x):
    x_imp = f.transform(x)
    return x_imp
def get_scaler(x_train):
    std = StandardScaler()
    std.fit(x_train)
    return std
def apply_scaler(f, x):
    x_std = f.transform(x)
    return x_std
# def get_model(x_train, y_train):
#     model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', intercept_scaling=1, max_iter=600, class_weight='balanced')
#     model.fit(x_train, y_train) 
#     return model
def tune_model(model, x_train, y_train, param_grid, kfold = StratifiedKFold(n_splits=5), \
               scoring='roc_auc_ovo_weighted'):
    gs = GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring=scoring, n_jobs=-1, verbose=1)
    gs.fit(x_train, y_train)
    return gs.best_estimator_#, gs.best_score_, gs.best_params_
def train_model(model, x_train, y_train, param_grid={}, kfold = StratifiedKFold(n_splits=5)):
    if len(param_grid)<1:
        model.fit(x_train, y_train) 
        return model
    return tune_model(model, x_train, y_train, param_grid, kfold)
def apply_model(model, x_test):
    y_pred = model.predict_proba(x_test)
    return y_pred[:,1]
def cross_validate(model, x_train, y_train, n_splits=5, n_repeats=3):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=kfold, n_jobs=-1)
    return scores.mean()
def get_roc_auc(y_test, y_pred):
    fpr, tpr, threshold = roc_curve(y_test, y_pred) 
    roc_auc = auc(fpr,tpr)
    return roc_auc
df_train = pd.read_csv(data_dir + train_file)
df_test = pd.read_csv(data_dir + test_file)
df_train, df_test = get_creator_features(df_train,df_test)
print (df_train.shape)
df_train.head()
train, val = train_test_split(df_train, test_size=0.2, stratify=df_train[target_col], random_state=42)
train.shape, val.shape
train_ids = train[['id']]
train_ids
val_ids = val[['id']]
val_ids
x_train_raw = train.drop(columns=target_col)
y_train = train[[target_col]]
x_train_raw.head()
x_val_raw = val.drop(columns=target_col)
y_val = val[[target_col]]
x_val_raw.head()
x_train_raw.shape
y_val = val[['id', target_col]]
y_val
y_val = val[[target_col]]
x_train_raw.shape
'''x_train = get_features(x_train_raw, features)
x_train.shape'''
#x_train
models = {}
param_grids = {}
name = 'RF'

model = RandomForestClassifier(class_weight='balanced', random_state=random_state)

param_grid = {
    "max_depth": [None, 20],
#     "max_features": ['auto', 'sqrt', 'log2'],
#     "min_samples_split": [2,10],
#     "min_samples_leaf": [3,10],
#     "bootstrap": [False],
#     "n_estimators": [100, 300],
#     "criterion": ['gini']
}

models[name] = model
param_grids[name] = param_grid
from sklearn.ensemble import GradientBoostingClassifier
name = 'XGB'

model = GradientBoostingClassifier( random_state=random_state)

param_grid = {
    "max_depth": [3,10,20],
    "n_estimators": [10, 50 ,100],
    "max_features": ['auto', 'sqrt', 10],
    "min_samples_split": [2,10],
    #"min_samples_leaf": [3,10],
    #"bootstrap": [False],
    #"criterion": ['gini']
}

models[name] = model
param_grids[name] = param_grid
name = 'ExtraTrees'
model = ExtraTreesClassifier(class_weight='balanced', random_state=random_state)

param_grid = {
    "max_depth": [20],
#     "max_features": ['auto', 'sqrt', 'log2'],
#     "min_samples_split": [2,10],
#     "min_samples_leaf": [3,10],
#     "bootstrap": [False],
#     "n_estimators": [100, 300],
#     "criterion": ['gini']
}

models[name] = model
param_grids[name] = param_grid
name = 'AdaBoost'
model = AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced'), random_state=random_state)

param_grid = {
#     "base_estimator__criterion": ['gini', 'entropy'],
#              "base_estimator__splitter": ['best', 'random'],
#              "algorithm": ['SAMME', 'SAMME.R'],
#              "n_estimators": [1,2],
#              "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.3, 1.5]
}

models[name] = model
param_grids[name] = param_grid
name = 'GradBoost'
model = GradientBoostingClassifier(random_state=random_state)

param_grid = {
    'loss': ['deviance'],
#     'n_estimators': [100],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [4, 8],
#     'min_samples_leaf': [100],
#     'max_features': ['auto', 'sqrt', 'log2']
}

models[name] = model
param_grids[name] = param_grid
name = 'SVC'
model = SVC(probability=True, class_weight='balanced', random_state=random_state)

param_grid = {
#     "kernel": ['rbf', 'sigmoid'],
#     "gamma": ['auto', 'scale', 0.01, 0.1, 1],
#     "C": [1, 10, 100, 1000]
}

models[name] = model
param_grids[name] = param_grid
name = 'MLP'
model = MLPClassifier(random_state=random_state, warm_start=True)

param_grid = {
    "hidden_layer_sizes": [(20, 20, 20)], 
    "activation": ["relu"], 
    "alpha": [0.1, 0.01, 0.001, 0.0001]
}

models[name] = model
param_grids[name] = param_grid

x_train_raw
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# model_name = 'SVC'
# model_name = 'ExtraTrees'
# model_name = 'AdaBoost'
#model_name = 'GradBoost'
#model_name='XGB'
model_name='RF'
model = models[model_name]


param_grid = param_grids[model_name]
#df_train = pd.read_csv(data_dir + train_file)
'''print (df_train.shape)
df_train.head()'''

df_train = pd.read_csv(data_dir + train_file)
df_pred = pd.read_csv(data_dir + test_file)
#df_lda_train = pd.read_csv(data_dir + 'train_set_with_lda_and_currency.csv')
#df_lda_test = pd.read_csv(data_dir + 'test_set_with_lda_and_currency.csv')
'''df_train=pd.merge(df_train,
         df_lda_train[['id','lda_topic','lcu_per_int','ppp_goal']].drop_duplicates(subset=['id']),
         how='left',
         on='id').fillna(0)
df_pred=pd.merge(df_pred,
         df_lda_test[['id','lda_topic','lcu_per_int','ppp_goal']].drop_duplicates(subset=['id']),
         how='left',
         on='id').fillna(0)
'''



x_train = df_train[[c for c in df_train if 'outcome' not in c]]
y_saved=df_train['outcome']
cat_features=['main_category',
             #'sub_category',
             'country',
             'location_state',
             'location_type',
                           
             ]
features_drop=['id','name','blurb','slug','currency','created_at','deadline','launched_at',
               'launch_datetime','deadline_datetime','sub_category',
               'urls','photo','creator_name','creator_id','creator_url']
num_features=[col for col in x_train if (col not in cat_features+features_drop) and ('category') not in col]
#y_saved=y_train
X = x_train[cat_features + num_features]
X=X.fillna(0)
X.info()
name = 'XGB'

model = RandomForestClassifier( random_state=random_state)

param_grid = {
    #'pca__n_components': [5, 15],
    "classifier__n_estimators": [100,300],
     "classifier__max_depth": [None, 10,20],
     "classifier__max_features": ['auto', 'sqrt', 10],
     #"classifier__min_samples_split": [2,10],
     #"classifier__min_samples_leaf": [1,3,10],
     #"classifier__bootstrap": [False],
     #"classifier__criterion": ['gini']
}

models[name] = model
param_grids[name] = param_grid
from sklearn.decomposition import PCA

y=y_saved


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)
#pca=PCA()
#pca.fit()

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, cat_features),
     ('num', numerical_pipe, num_features)])


pipe = Pipeline([
    ('preprocess', preprocessing),
    #('pca', PCA(n_components=None)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
#rf.fit(X_train, y_train)
param_grid
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
X.info()
kfold = StratifiedKFold(n_splits=5)
scoring='roc_auc_ovo_weighted'
gs = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring=scoring, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
model= gs.best_estimator_#, gs.best_score_, gs.best_params_
'''#model = train_model(model, x_train, y_train, param_grid)
model= train_model(rf, X_train, y_train, param_grid)

model'''
y_pred_val = apply_model(model, X_test)
cv_auc = cross_validate(model, X_train, y_train, n_splits=5, n_repeats=1)
cv_auc
auc_val = get_roc_auc(y_test, y_pred_val)
auc_val
'''from sklearn.metrics import roc_auc_score

y_pred_val=rf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_val)'''
#rf.predict_proba(X_test)[:,1]
#gs.fit(X, y)
model.fit(X,y)
x_pred = df_pred[[c for c in df_pred if 'outcome' not in c]]
x_pred=get_features(x_pred)
x_pred=x_pred[cat_features + num_features]
#y_test_off=df_['outcome']
y_pred_off= apply_model(model, x_pred)
df_pred['Predicted']=y_pred_off
df_pred[['id','Predicted']].drop_duplicates().to_csv('out_xgb.csv', index=False)

#pd.DataFrame(y_pred_off).to_csv('out_xgb.csv')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

# Apply the default theme
sns.set_theme()
ohe = (model.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = ohe.get_feature_names(input_features=cat_features)
feature_names = np.r_[feature_names, num_features]

tree_feature_importances = (
    model.named_steps['classifier'].feature_importances_)


#n_feat=len(feature_names)
n_feat=20
sorted_idx = tree_feature_importances.argsort()[-n_feat:]


y_ticks = np.arange(0, n_feat)
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances (MDI)")
fig.tight_layout()
plt.show()
result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
#n_feat=len(feature_names)
n_feat=20
sorted_idx = result.importances_mean.argsort()[-n_feat:]

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
n_feat=15
sorted_idx = result.importances_mean.argsort()[-n_feat:]
#plt.figure(figsize=[10,10])
fig, ax = plt.subplots(figsize=[8,8])
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
result = permutation_importance(model, X_train, y_train, n_repeats=10,
                                random_state=42, n_jobs=2)
#n_feat=len(feature_names)
n_feat=15
sorted_idx = result.importances_mean.argsort()[-n_feat:]

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (Train set)")
fig.tight_layout()
plt.show()
#!pip install lime
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from __future__ import print_function
from lime.lime_text import LimeTextExplainer


class_names=['staff']
#explainer = LimeTextExplainer(class_names=X_test.columns[sorted_idx])
explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                   feature_names=iris.feature_names,
                                                   class_names=iris.target_names,
                                                   discretize_continuous=True)



idx = 1
exp = explainer.explain_instance(X_test.iloc[idx], model.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])
X_test.iloc[idx]











x_val
len(y_pred_val)
val_ids
# y_pred = pd.DataFrame(columns=['Id', 'pred'])
# y_pred['Id'] = val_ids
y_pred = copy.deepcopy(val_ids)
y_pred['pred'] = y_pred_val
y_pred
%pwd
y_pred.to_csv('../predictions/%s.csv' % model_name)

if generate_predictions:
    x_test = pd.read_csv(data_dir + test_file)
    test_ids = x_test['id']

    x_test = get_features(x_test, features)
    x_test[features] = apply_imputer(imp, x_test)
    if not skip_scaling:
        x_test[features] = apply_scaler(scl, x_test)
    y_pred_test_array = apply_model(model, x_test)

    y_pred_test = pd.DataFrame()
    y_pred_test['Id'] = test_ids
    y_pred_test['Predicted'] = y_pred_test_array
    y_pred_test.set_index('Id', inplace=True)
    display(y_pred_test.head())

    y_pred_test.to_csv(data_dir + 'pred.csv')

