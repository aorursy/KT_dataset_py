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
from sklearn.model_selection import *
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit

# Classification models
import xgboost as xgb

import warnings
warnings.simplefilter('ignore')


from sklearn.model_selection import train_test_split , StratifiedKFold , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier , BaggingClassifier  , GradientBoostingClassifier , VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

train = pd.read_csv("../input/soil-sam/soil_samples_training.csv")
test = pd.read_csv("../input/soil-sam/soil_samples_test.csv")
train.isnull().sum()
test.isnull().sum()
train.describe()
train = train.apply(lambda x:x.fillna(x.median()) \
                if x.dtype == 'float' else \
                x.fillna(x.value_counts().index[0]))
test = test.apply(lambda x:x.fillna(x.median()) \
                if x.dtype == 'float' else \
                x.fillna(x.value_counts().index[0]))
features = train.columns.to_list()

target = 'origin'
features.remove(target)
features
train  = train.dropna()
train.shape
train.set_index('sample_id' , inplace = True)
train.head()
test.set_index('sample_id' , inplace = True)
test.head()
train["grain_shape"].value_counts()


sns.countplot(train["grain_shape"])

sns.countplot(train["grain_shape"] , hue = train["origin"])
train["grain_surface"].value_counts()
sns.countplot(train["grain_surface"] , hue = train["origin"])
train["grain_color"].value_counts()
sns.countplot(train["grain_color"] , hue = train["origin"])
'''particle_attached',
 'particle_spacing',
 'particle_width',
 'particle_color',
 'particle_distribution',
 'optical_density'''
train["particle_attached"].value_counts()
sns.countplot(train["particle_attached"] , hue = train["origin"])

sns.countplot(train["particle_spacing"] )
sns.countplot(train["particle_spacing"] , hue = train["origin"])
sns.countplot(train["particle_width"])
sns.countplot(train["particle_width"] , hue = train["origin"])
sns.countplot(train["particle_color"] , hue = train["origin"])
sns.countplot(train["particle_color"])
sns.countplot(train["particle_distribution"])
sns.countplot(train["particle_distribution"] , hue = train["origin"])
train.info()
sns.distplot(train["optical_density"])
sns.boxplot(x = train['origin'] , y = train['optical_density'] , width = 0.5 , notch = True)
features.remove('sample_id')
sns.distplot(train["pH"])
sns.countplot(train["particle_distribution"] , hue = train["origin"])
sns.boxplot(x = train['origin'] , y = train['pH'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['chlorate'] , width = 0.5 , notch = True)

train.columns
sns.boxplot(x = train['origin'] , y = train['nitrate'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['chloride'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['nitrate'] , width = 0.5 , notch = True)

'''nitrite',
       'sulphate', 'sulphite', 'phosphate', 'radioactivity',
       'isotope_diversity'
       '''
sns.boxplot(x = train['origin'] , y = train['sulphate'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['sulphite'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['phosphate'] , width = 0.5 , notch = True)

sns.boxplot(x = train['origin'] , y = train['radioactivity'] , width = 0.5 , notch = True)

sns.countplot(train['isotope_diversity'] , hue = train['origin'] )

train.head()
train.columns
for f in train.columns:
    if train[f].dtype=='object' and f!='origin':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
    elif train[f].dtype=='object':
        lbls = preprocessing.LabelEncoder()
        lbls.fit(list(train[f].values))
        train[f] = lbls.transform(list(train[f].values))
        

print(lbls.classes_)
print(features) 
train.columns 
def baseliner(X_data , y_data , scoring = 'accuracy' , cv=3 , Z=2):
    print("Baseliner")
    eval_dict = {}
    models = [
        KNeighborsClassifier() , lgb.LGBMClassifier()  ,
        xgb.XGBClassifier(objective = 'binary:logistic') , cat.CatBoostClassifier(verbose = 0) , GradientBoostingClassifier(), 
        RandomForestClassifier() , LogisticRegression() , DecisionTreeClassifier() , AdaBoostClassifier() , BaggingClassifier()
    ]
    
    print("Sk learn model name \t cv")
    print("-" *100)
    for index , model in enumerate(models , 0):
        #print(model , index)
        model_name = str(model).split("(")[0]
        eval_dict[model_name] = {}
        #classes = y_data.unique()
        result = cross_val_score(estimator = model , X=X_data , y = y_data , cv = 5 , scoring = scoring)
        #print('*' * 10 , result)
        eval_dict[model_name]['cv'] = result.mean()
        
        print("%s \t %.4f \t" % (model_name[:21] , eval_dict[model_name]['cv']))
        
              
target = "origin"

X, y = train.drop(target,axis = 1),train[target]
X.shape , y.shape
X.columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
X_train , X_val , y_train , y_val = train_test_split(X, y , test_size = 0.3 , random_state = 41)
baseliner(X_data = X , y_data = y ,cv=10)
test.shape
test.head()

features
features

train = train.reset_index()
gss = GroupShuffleSplit(n_splits=10, train_size=0.9, random_state=42)
test_prob_preds = np.zeros(test.shape[0])
print(test_prob_preds)
for idx, (train_idx, valid_idx) in enumerate(gss.split(train[features], train[target], train['sample_id']), 1):
    print("-"*50)
    print("Iteration Number  : {}".format(idx))
    MAX_ROUNDS=2000
    early_stopping_rounds=100
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'learning_rate': 0.1,
        'num_round': MAX_ROUNDS,
        'max_depth': 8,
        'seed': 41,
    }

    X_train, X_valid, y_train, y_valid = train[features].iloc[train_idx], train[features].iloc[valid_idx], train[target].iloc[train_idx], train[target].iloc[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
    )
    print("Best Iteration :: {} \n".format(model.best_iteration))

    # Plotting Importances
    fig, ax = plt.subplots(figsize=(24, 24))
    xgb.plot_importance(model, height=0.4, ax=ax)
    preds = model.predict(xgb.DMatrix(test[features]), ntree_limit=model.best_ntree_limit)
    
    test_prob_preds += preds
print(test_prob_preds)
print(gss.n_splits)
test_prob_preds /= gss.n_splits
print(test_prob_preds)
print(test_prob_preds.shape, test_prob_preds[: 5])
target
def k_fold_validation(model , train , features, target , n_splits = 5):
    X = train[features].copy()
    y = train[target].copy()
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5 )
    kf.get_n_splits(X)
    res = []
    for train_index, test_index in kf.split(X):
        X_train,X_test = X.iloc[train_index],X.iloc[test_index]
        y_train  , y_test = y.iloc[train_index] , y.iloc[test_index]
        model.fit(X_train , y_train)
        y_pred = model.predict(X_test)
        res.append(accuracy_score(y_test,y_pred))
    print("Accuracy",np.array(res).mean())
model = RandomForestClassifier()
k_fold_validation(model , train , features , target ,  n_splits = 5)
model.fit(train[features] , train[target])
features
y_pred = model.predict(test[features])
y_2 = lbls.inverse_transform(y_pred)
test = test.reset_index()
test["predicted_origin"] = pd.DataFrame(y_2)
test.columns
test = test[['sample_id' , 'predicted_origin']]
test['predicted_origin'].value_counts()
test.to_csv('./anshaj_upadhyay.csv' , index = False)
test['predicted_origin'].value_counts()
test.head(10)
X = test_prob_preds > 0.5 
Y = pd.DataFrame(X , columns = ['gg'])
dict = {False : 'alien' , True:'earth'}
type(Y)
Y.columns
Y['pop']= Y['gg'].map(dict)
Y['pop'].value_counts()
