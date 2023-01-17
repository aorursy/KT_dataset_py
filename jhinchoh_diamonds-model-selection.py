from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



import seaborn as sns

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



seed=8

models = []

mse = []

r2 = []



df = pd.read_csv('/kaggle/input/diamonds.csv')

del df['Unnamed: 0']
df.isna().sum()
print(df.color.unique())

print(df.cut.unique())

print(df.clarity.unique())



# One-hot encode categorical values

lst_onehot = ['color','cut','clarity']

df_s = df[lst_onehot]

df_o = pd.get_dummies(df_s)

df = df.drop(lst_onehot,axis = 1)

df = pd.concat([df,df_o], axis=1)
# Log transformation

lst_log = ['carat','price']

for i in lst_log:

    df[i] = np.log1p(df[i])
sns.distplot(df.carat) #log transformation

plt.show()

sns.distplot(df.price) #log transformation

plt.show()
# Scaling

from sklearn.preprocessing import RobustScaler



rob_scaler = RobustScaler()

lst_scale = ['depth','table']



for i in lst_scale:

    df[i] = rob_scaler.fit_transform(df[i].values.reshape(-1,1))
df = df[(df.depth < 7) & (df.depth > -7)]

df = df[(df.table < 4) & (df.table > -3)]

df = df[(df.x < 10) & (df.x > 2.5)]

df = df[df.y < 12]

df = df[(df.z < 10) & (df.z > 1)]
sns.distplot(df.depth) 

plt.show()

sns.distplot(df.table)

plt.show()

sns.distplot(df.x) 

plt.show()

sns.distplot(df.y) 

plt.show()

sns.distplot(df.z) 

plt.show()
df.head()
#JC: Split the target column from the feature columns

X = df.drop('price', axis = 1)

y = df['price']

    

X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=8,test_size=0.2)
def evaluate(model, name):

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    models.append(name) 

    mse.append(mean_squared_error(Y_test, Y_pred)) 

    r2.append(r2_score(Y_test, Y_pred)) 



def search_grid(model):

    model.fit(X_train, Y_train)

    model.best_params_

    return(model.best_estimator_)
def model_knr(name):

    model_default = KNeighborsRegressor(n_jobs = -1)

    param_grid = {

        'metric': ['euclidean','manhattan'],

        'weights': ['uniform', 'distance'],

        'n_neighbors': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')



def model_gbr(name):

    model_default = GradientBoostingRegressor()

    param_grid = {

        "learning_rate": [0.075, 0.1, 0.15, 0.2],

        'min_samples_leaf': [3, 4, 5],

        'min_samples_split': [8, 10, 12],

        "max_depth": [3,5,8],

        "subsample": [0.5, 0.8, 0.9, 1.0],

        'n_estimators': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')



def model_rfr(name):

    model_default = RandomForestRegressor(random_state = seed, n_jobs = -1, verbose = 0)

    param_grid = {

        'bootstrap': [True],

        'max_depth': [80, 90, 100],

        'max_features': [2, 3],

        'min_samples_leaf': [3, 4, 5],

        'min_samples_split': [8, 10, 12],

        'n_estimators': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')

    

def model_xgbr(name):

    model_default = XGBRegressor(random_state = seed,n_jobs = -1, verbose = 0)

    param_grid = {

        'max_depth': [3, 4, 5],

        'subsample': [0.9, 1.0],

        'colsample_bytree': [0.9, 1.0],

        'learning_rate': [0.05, 0.1, 0.5]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')
def run_models():

    model_knr('KNR')

    model_gbr('GBR')

    model_rfr('RFR')

    model_xgbr('XGBR')

    

run_models()

pd.DataFrame({"model":models, "mse":mse, "r2":r2})
best_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1.0, gamma=0,

             importance_type='gain', learning_rate=0.5, max_delta_step=0,

             max_depth=5, min_child_weight=1, missing=None, n_estimators=100,

             n_jobs=-1, nthread=None, objective='reg:linear', random_state=8,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1.0, verbose=0, verbosity=1)



best_model