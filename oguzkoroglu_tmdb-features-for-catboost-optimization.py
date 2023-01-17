# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing, model_selection, neighbors, svm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MultiLabelBinarizer

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.impute import SimpleImputer



from catboost import CatBoostRegressor, Pool



from tqdm import tqdm

import json

import ast



from datetime import datetime
TRAIN_DATA_PATH = "../input/train.csv"

TEST_DATA_PATH = "../input/test.csv"

SUBMISSON_PATH = "../input/sample_submission.csv"

LABEL_COL_NAME = "revenue"
def date(x):

    x=str(x)

    year=x.split('/')[2]

    if int(year)<19:

        return x[:-2]+'20'+year

    else:

        return x[:-2]+'19'+year



def isNaN(x):

    return str(x) == str(1e400 * 0)



def getIsoListFormJson(data, isoKey='id', forceInt=False):

    datas = data.values.flatten()

    ids = []

    for c in (datas):    

        ccc = []

        if isNaN(c) == False:

            c = json.dumps(ast.literal_eval(c))        

            c = json.loads(c)            

            for cc in c:

                if forceInt:

                    ccStr = int(cc[isoKey])

                else:

                    ccStr = str(cc[isoKey])

                ccc.append(ccStr)

        else:

            if forceInt:

                ccc.append(0)

            else:

                ccc.append('0')

        ids.append(ccc)    

    return np.array(ids)



def distributeIdsOverData(data, colName, isoKey='id', forceInt=True):

    arr = getIsoListFormJson(data[colName], isoKey, forceInt)    



    gsi = -1

    for gs in tqdm(arr):

        gsi += 1

        gs.sort()

        for g in gs:

            gi = gs.index(g)

            try:

                data.loc[gsi, f"{colName}_{gi}"] = float(g)                

            except :

                data.loc[gsi, f"{colName}_{gi}"] = g                

            

    data.drop(colName, axis=1, inplace=True)

    print(f"{colName} distributed over data, cols: {len(data.columns)}")



def imput_title(df):

    for index, row in df.iterrows():

        if row['title'] == "none":

            df.at[index,'title'] = df.loc[index]['original_title']

    return df    

    

def prepareData(data):    

    data = imput_title(data)



    data["different_title"] = data["original_title"] != data["title"]



    data.drop("overview", axis=1, inplace=True)

    data.drop("poster_path", axis=1, inplace=True)

    data.drop('imdb_id', axis=1, inplace=True)    



    data["belongs_to_collection"] = getIsoListFormJson(data["belongs_to_collection"])



    cast = data['cast'].fillna('none')

    cast = cast.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

    data['num_cast'] = cast.apply(lambda x: len(x) if x != {} else 0)

    # Get the sum of each of the cast genders in a film: 0 `unknown`, 1 `female`, 2 `male`

    data['genders_0_cast'] = cast.apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    data['genders_1_cast'] = cast.apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    data['genders_2_cast'] = cast.apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    distributeIdsOverData(data,'cast','cast_id')



    crew = data['crew'].fillna('none')

    crew = crew.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))    

    data['num_crew'] = crew.apply(lambda x: len(x) if x != {} else 0)    

    # Get the sum of each of the cast genders in a film: 0 `unknown`, 1 `female`, 2 `male`

    data['genders_0_crew'] = crew.apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    data['genders_1_crew'] = crew.apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    data['genders_2_crew'] = crew.apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    distributeIdsOverData(data,'crew','name',False) 



    distributeIdsOverData(data,'genres')

    

    keywords = data['Keywords'].fillna('none')

    keywords = keywords.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

    data['num_keywords'] = keywords.apply(lambda x: len(x) if x != {} else 0)

    distributeIdsOverData(data,'Keywords')



    data["Has_HomePage"] = list(map(lambda c: float(c is not np.nan), data["homepage"]))

    data.drop('homepage', axis=1, inplace=True)



    data["IsReleased"] = list(map(lambda c: float(c == "Released"), data["status"]))

    data.drop("status", axis=1, inplace=True)

  

    data["original_title_len"] = list(map(lambda c: float(len(str(c))), data["original_title"]))

    data.drop("original_title", axis=1, inplace=True)

    

    production_companies = data['production_companies'].fillna('none')

    production_companies = production_companies.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

    data['num_production_companies'] = production_companies.apply(lambda x: len(x) if x != {} else 0)

    distributeIdsOverData(data,'production_companies')    



    production_countries = data['production_countries'].fillna('none')

    production_countries = production_countries.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

    data['num_production_countries'] = production_countries.apply(lambda x: len(x) if x != {} else 0)    

    distributeIdsOverData(data,'production_countries','iso_3166_1',False)



    data['release_date']=data['release_date'].fillna('1/1/90').apply(lambda x: date(x))

    data['release_date']=data['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))

    data['release_day']=data['release_date'].apply(lambda x:x.weekday())

    data['release_month']=data['release_date'].apply(lambda x:x.month)

    data['release_year']=data['release_date'].apply(lambda x:x.year)

    data.drop('release_date', axis=1, inplace=True)

    

    spoken_languages = data['spoken_languages'].fillna('none')

    spoken_languages = spoken_languages.apply(lambda x: {} if x == 'none' else ast.literal_eval(x))

    data['num_spoken_languages'] = spoken_languages.apply(lambda x: len(x) if x != {} else 0)

    distributeIdsOverData(data,'spoken_languages','iso_639_1',False)



    data["tagline_len"] = list(map(lambda c: float(len(str(c))), data["tagline"]))

    data.drop("tagline", axis=1, inplace=True)



    data["title_len"] = list(map(lambda c: float(len(str(c))), data["title"]))

    data.drop("title", axis=1, inplace=True)    



    data.fillna(0, inplace=True)

    data["budget"] = np.log1p(SimpleImputer(missing_values=0, strategy="median", verbose=1).fit_transform(data["budget"].values.reshape(-1,1)))

    #data["budget"] = np.log1p(data["budget"])



    data[LABEL_COL_NAME] = np.log1p(data[LABEL_COL_NAME])
train = pd.read_csv(TRAIN_DATA_PATH, index_col='id')

print("Train Data Loaded")

test = pd.read_csv(TEST_DATA_PATH, index_col = 'id')

print("Test Data Loaded")
if not os.path.exists("all_data.pickle"):   

    ##FILLING MISSIN BUDGET DATA

    train.loc[16,'revenue'] = 192864          # Skinning

    train.loc[90,'budget'] = 30000000         # Sommersby          

    train.loc[118,'budget'] = 60000000        # Wild Hogs

    train.loc[149,'budget'] = 18000000        # Beethoven

    train.loc[313,'revenue'] = 12000000       # The Cookout 

    train.loc[451,'revenue'] = 12000000       # Chasing Liberty

    train.loc[464,'budget'] = 20000000        # Parenthood

    train.loc[470,'budget'] = 13000000        # The Karate Kid, Part II

    train.loc[513,'budget'] = 930000          # From Prada to Nada

    train.loc[797,'budget'] = 8000000         # Welcome to Dongmakgol

    train.loc[819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

    train.loc[850,'budget'] = 90000000        # Modern Times

    train.loc[1112,'budget'] = 7500000        # An Officer and a Gentleman

    train.loc[1131,'budget'] = 4300000        # Smokey and the Bandit   

    train.loc[1359,'budget'] = 10000000       # Stir Crazy 

    train.loc[1542,'budget'] = 1              # All at Once

    train.loc[1542,'budget'] = 15800000       # Crocodile Dundee II

    train.loc[1571,'budget'] = 4000000        # Lady and the Tramp

    train.loc[1714,'budget'] = 46000000       # The Recruit

    train.loc[1721,'budget'] = 17500000       # Cocoon

    train.loc[1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

    train.loc[2268,'budget'] = 17500000       # Madea Goes to Jail budget

    train.loc[2491,'revenue'] = 6800000       # Never Talk to Strangers

    train.loc[2602,'budget'] = 31000000       # Mr. Holland's Opus

    train.loc[2612,'budget'] = 15000000       # Field of Dreams

    train.loc[2696,'budget'] = 10000000       # Nurse 3-D

    train.loc[2801,'budget'] = 10000000       # Fracture



    test.loc[3889,'budget'] = 15000000       # Colossal

    test.loc[6733,'budget'] = 5000000        # The Big Sick

    test.loc[3197,'budget'] = 8000000        # High-Rise

    test.loc[6683,'budget'] = 50000000       # The Pink Panther 2

    test.loc[5704,'budget'] = 4300000        # French Connection II

    test.loc[6109,'budget'] = 281756         # Dogtooth

    test.loc[7242,'budget'] = 10000000       # Addams Family Values

    test.loc[7021,'budget'] = 17540562       #  Two Is a Family

    test.loc[5591,'budget'] = 4000000        # The Orphanage

    test.loc[4282,'budget'] = 20000000       # Big Top Pee-wee



    train.loc[391,'runtime'] = 86 #Il peor natagle de la meva vida

    train.loc[592,'runtime'] = 90 #А поутру они проснулись

    train.loc[925,'runtime'] = 95 #¿Quién mató a Bambi?

    train.loc[978,'runtime'] = 93 #La peggior settimana della mia vita

    train.loc[1256,'runtime'] = 92 #Cipolla Colt

    train.loc[1542,'runtime'] = 93 #Все и сразу

    train.loc[1875,'runtime'] = 86 #Vermist

    train.loc[2151,'runtime'] = 108 #Mechenosets

    train.loc[2499,'runtime'] = 108 #Na Igre 2. Novyy Uroven

    train.loc[2646,'runtime'] = 98 #同桌的妳

    train.loc[2786,'runtime'] = 111 #Revelation

    train.loc[2866,'runtime'] = 96 #Tutto tutto niente niente

    

    test.loc[4074,'runtime'] = 103 #Shikshanachya Aaicha Gho

    test.loc[4222,'runtime'] = 93 #Street Knight

    test.loc[4431,'runtime'] = 100 #Плюс один

    test.loc[5520,'runtime'] = 86 #Glukhar v kino

    test.loc[5845,'runtime'] = 83 #Frau Müller muss weg!

    test.loc[5849,'runtime'] = 140 #Shabd

    test.loc[6210,'runtime'] = 104 #Le dernier souffle

    test.loc[6804,'runtime'] = 145 #Chaahat Ek Nasha..

    test.loc[7321,'runtime'] = 87 #El truco del manco



    all_data = train.append(test)

    print("Preparing All Data")

    prepareData(all_data)    

    all_data.to_pickle("all_data.pickle")

    print("saved all data")

else: 

    all_data = pd.read_pickle("all_data.pickle")

    print("saved all data")
train = all_data[:len(train)]

test = all_data[len(train):]
train.head()
train.describe()
test.head()
test.describe()
X_tr = train.drop(LABEL_COL_NAME, axis = 1)

y_tr = train[LABEL_COL_NAME]

numerical_features = ['budget',

                      'popularity', 

                      'runtime', 

                      'title_len', 

                      'original_title_len', 

                      'tagline_len',

                      'num_crew',

                      'num_cast',

                      'num_keywords',

                      'num_production_companies',

                      'num_production_countries',

                      'num_spoken_languages']

cat_features = set(X_tr.columns) - set(numerical_features)

cat_features = [list(X_tr.columns).index(c) for c in cat_features]
#import required packages

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

import gc

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt.fmin import fmin

from hyperopt.pyll.stochastic import sample

#optional but advised

import warnings

warnings.filterwarnings('ignore')



#GLOBAL HYPEROPT PARAMETERS

NUM_EVALS = 100 #number of hyperopt evaluation rounds

N_FOLDS = 3 #number of cross-validation folds on data in each evaluation round



#LIGHTGBM PARAMETERS

LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM

LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM

EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 

EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric



#XGBOOST PARAMETERS

XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting

XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost

EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric

EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric



#CATBOOST PARAMETERS

CB_MAX_DEPTH = 6 #maximum tree depth in CatBoost

OBJECTIVE_CB_REG = 'RMSE' #CatBoost regression metric

OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric



def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False, cat_features=[]):

    

    #==========

    #LightGBM

    #==========

    

    if package=='lgbm':

        

        print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))

        #clear space

        gc.collect()

        

        integer_params = ['max_depth',

                         'num_leaves',

                          'max_bin',

                         'min_data_in_leaf',

                         'min_data_in_bin']

        

        def objective(space_params):

            

            #cast integer params from float to int

            for param in integer_params:

                space_params[param] = int(space_params[param])

            

            #extract nested conditional parameters

            if space_params['boosting']['boosting'] == 'goss':

                top_rate = space_params['boosting'].get('top_rate')

                other_rate = space_params['boosting'].get('other_rate')

                #0 <= top_rate + other_rate <= 1

                top_rate = max(top_rate, 0)

                top_rate = min(top_rate, 0.5)

                other_rate = max(other_rate, 0)

                other_rate = min(other_rate, 0.5)

                space_params['top_rate'] = top_rate

                space_params['other_rate'] = other_rate

            

            subsample = space_params['boosting'].get('subsample', 1.0)

            space_params['boosting'] = space_params['boosting']['boosting']

            space_params['subsample'] = subsample

            

            #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS

            cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,

                                early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)

            

            best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse

            #for classification, comment out the line above and uncomment the line below:

            #best_loss = 1 - cv_results['auc-mean'][-1]

            #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'

            return{'loss':best_loss, 'status': STATUS_OK }

        

        train = lgb.Dataset(data, labels)

                

        #integer and string parameters, used with hp.choice()

        boosting_list = [{'boosting': 'gbdt',

                          'subsample': hp.uniform('subsample', 0.5, 1)},

                         {'boosting': 'goss',

                          'subsample': 1.0,

                         'top_rate': hp.uniform('top_rate', 0, 0.5),

                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'

        metric_list = ['MAE', 'RMSE'] 

        #for classification comment out the line above and uncomment the line below

        #modify as required for other classification metrics classification

        #metric_list = ['auc']

        objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']

        objective_list_class = ['logloss', 'cross_entropy']

        #for classification set objective_list = objective_list_class

        objective_list = objective_list_reg



        space ={'boosting' : hp.choice('boosting', boosting_list),

                'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),

                'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),

                'max_bin': hp.quniform('max_bin', 32, 255, 1),

                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),

                'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),

                'lambda_l1' : hp.uniform('lambda_l1', 0, 5),

                'lambda_l2' : hp.uniform('lambda_l2', 0, 5),

                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),

                'metric' : hp.choice('metric', metric_list),

                'objective' : hp.choice('objective', objective_list),

                'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),

                'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)

            }

        

        #optional: activate GPU for LightGBM

        #follow compilation steps here:

        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/

        #then uncomment lines below:

        #space['device'] = 'gpu'

        #space['gpu_platform_id'] = 0,

        #space['gpu_device_id'] =  0



        trials = Trials()

        best = fmin(fn=objective,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=num_evals, 

                    trials=trials)

                

        #fmin() will return the index of values chosen from the lists/arrays in 'space'

        #to obtain actual values, index values are used to subset the original lists/arrays

        best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice

        best['metric'] = metric_list[best['metric']]

        best['objective'] = objective_list[best['objective']]

        

        #cast floats of integer params to int

        for param in integer_params:

            best[param] = int(best[param])

            

        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

        if diagnostic:

            return(best, trials)

        else:

            return(best)

    

    #==========

    #XGBoost

    #==========

    

    if package=='xgb':

        

        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))

        #clear space

        gc.collect()

        

        integer_params = ['max_depth']

        

        def objective(space_params):

            

            for param in integer_params:

                space_params[param] = int(space_params[param])

                

            #extract multiple nested tree_method conditional parameters

            #libera te tutemet ex inferis

            if space_params['tree_method']['tree_method'] == 'hist':

                max_bin = space_params['tree_method'].get('max_bin')

                space_params['max_bin'] = int(max_bin)

                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':

                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')

                    space_params['grow_policy'] = grow_policy

                    space_params['tree_method'] = 'hist'

                else:

                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')

                    space_params['grow_policy'] = 'lossguide'

                    space_params['max_leaves'] = int(max_leaves)

                    space_params['tree_method'] = 'hist'

            else:

                space_params['tree_method'] = space_params['tree_method'].get('tree_method')

                

            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS

            cv_results = xgb.cv(space_params, train, nfold=N_FOLDS, metrics=[EVAL_METRIC_XGB_REG],

                             early_stopping_rounds=100, stratified=False, seed=42)

            

            best_loss = cv_results['test-mae-mean'].iloc[-1] #or 'test-rmse-mean' if using RMSE

            #for classification, comment out the line above and uncomment the line below:

            #best_loss = 1 - cv_results['test-auc-mean'].iloc[-1]

            #if necessary, replace 'test-auc-mean' with 'test-[your-preferred-metric]-mean'

            return{'loss':best_loss, 'status': STATUS_OK }

        

        train = xgb.DMatrix(data, labels)

        

        #integer and string parameters, used with hp.choice()

        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'

        metric_list = ['MAE', 'RMSE'] 

        #for classification comment out the line above and uncomment the line below

        #metric_list = ['auc']

        #modify as required for other classification metrics classification

        

        tree_method = [{'tree_method' : 'exact'},

               {'tree_method' : 'approx'},

               {'tree_method' : 'hist',

                'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),

                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},

                                'grow_policy' : {'grow_policy':'lossguide',

                                                  'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES, 1)}}}]

        

        #if using GPU, replace 'exact' with 'gpu_exact' and 'hist' with

        #'gpu_hist' in the nested dictionary above

        

        objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']

        objective_list_class = ['reg:logistic', 'binary:logistic']

        #for classification change line below to 'objective_list = objective_list_class'

        objective_list = objective_list_reg

        

        space ={'boosting' : hp.choice('boosting', boosting_list),

                'tree_method' : hp.choice('tree_method', tree_method),

                'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),

                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),

                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),

                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),

                'gamma' : hp.uniform('gamma', 0, 5),

                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),

                'eval_metric' : hp.choice('eval_metric', metric_list),

                'objective' : hp.choice('objective', objective_list),

                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),

                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),

                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),

                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),

                'nthread' : -1

            }

        

        #optional: activate GPU for XGBoost

        #uncomment line below

        #space['tree_method'] = 'gpu_hist'

        

        trials = Trials()

        best = fmin(fn=objective,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=num_evals, 

                    trials=trials)

        

        best['tree_method'] = tree_method[best['tree_method']]['tree_method']

        best['boosting'] = boosting_list[best['boosting']]

        best['eval_metric'] = metric_list[best['eval_metric']]

        best['objective'] = objective_list[best['objective']]

        

        #cast floats of integer params to int

        for param in integer_params:

            best[param] = int(best[param])

        if 'max_leaves' in best:

            best['max_leaves'] = int(best['max_leaves'])

        if 'max_bin' in best:

            best['max_bin'] = int(best['max_bin'])

        

        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

        

        if diagnostic:

            return(best, trials)

        else:

            return(best)

    

    #==========

    #CatBoost

    #==========

    

    if package=='cb':

        

        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))

        

        #clear memory 

        gc.collect()

            

        integer_params = ['depth',

                          'one_hot_max_size', #for categorical data

                          'min_data_in_leaf',

                          'max_bin']

        

        def objective(space_params):

                        

            #cast integer params from float to int

            for param in integer_params:

                space_params[param] = int(space_params[param])

                

            #extract nested conditional parameters

            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':

                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')

                space_params['bagging_temperature'] = bagging_temp

                

            if space_params['grow_policy']['grow_policy'] == 'LossGuide':

                max_leaves = space_params['grow_policy'].get('max_leaves')

                space_params['max_leaves'] = int(max_leaves)

                

            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']

            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']

                           

            #random_strength cannot be < 0

            space_params['random_strength'] = max(space_params['random_strength'], 0)

            #fold_len_multiplier cannot be < 1

            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)

                       

            #for classification set stratified=True

            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 

                             early_stopping_rounds=25, stratified=False, partition_random_seed=42)

           

            #best_loss = cv_results['test-MAE-mean'].iloc[-1] 

            best_loss = cv_results['test-RMSE-mean'].iloc[-1] 

            

            #for classification, comment out the line above and uncomment the line below:

            #best_loss = cv_results['test-Logloss-mean'].iloc[-1]

            #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'

            

            return{'loss':best_loss, 'status': STATUS_OK}

        

        train = cb.Pool(data, labels.astype('float32'), cat_features=cat_features)

        

        #integer and string parameters, used with hp.choice()

        bootstrap_type = [

                          {'bootstrap_type':'Poisson'}, 

                          {'bootstrap_type':'Bayesian', 'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},

                          {'bootstrap_type':'Bernoulli'}] 

        LEB = ['No', 'AnyImprovement', 'Armijo'] #remove 'Armijo' if not using GPU

        #score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']

        grow_policy = [{'grow_policy':'SymmetricTree'},

                       {'grow_policy':'Depthwise'},

                       {'grow_policy':'Lossguide',

                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]

        eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']

        eval_metric_list_class = ['Logloss', 'AUC', 'F1']

        #for classification change line below to 'eval_metric_list = eval_metric_list_class'

        eval_metric_list = eval_metric_list_reg

                

        space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),

                'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254

                #'max_bin': 254,

                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),

                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),

                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),

                'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features

                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),

                'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),

                'eval_metric' : hp.choice('eval_metric', eval_metric_list),

                'objective' : OBJECTIVE_CB_REG,

                #'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown

                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),

                'grow_policy': hp.choice('grow_policy', grow_policy),

                #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only

                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),

                'od_type' : 'Iter',

                'od_wait' : 25,

                'task_type' : 'GPU',

                'verbose' : 0,

                'cat_features': cat_features

            }

        

        #optional: run CatBoost without GPU

        #uncomment line below

        #space['task_type'] = 'CPU'

            

        trials = Trials()

        best = fmin(fn=objective,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=num_evals, 

                    trials=trials)

        

        #unpack nested dicts first

        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']

        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']

        best['eval_metric'] = eval_metric_list[best['eval_metric']]

        

        #best['score_function'] = score_function[best['score_function']] 

        #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only

        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        

        

        #cast floats of integer params to int

        for param in integer_params:

            best[param] = int(best[param])

        if 'max_leaves' in best:

            best['max_leaves'] = int(best['max_leaves'])

        

        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

        

        if diagnostic:

            return(best, trials)

        else:

            return(best)

    

    else:

        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')     
cb_params = quick_hyperopt(X_tr, y_tr, 'cb', 15, cat_features=cat_features)

np.save('cb_params.npy', cb_params)

print(cb_params)
try:

    model = CatBoostRegressor(**cb_params, task_type='GPU')

    model.fit(X_tr, y_tr, cat_features=cat_features)    

except:

    print("GPU grow_policy error, just remove it")

    cb_params.pop('grow_policy')

    model = CatBoostRegressor(**cb_params, task_type='GPU')

    model.fit(X_tr, y_tr, cat_features=cat_features)
test = test.drop(LABEL_COL_NAME, axis = 1)

y_test = np.expm1(model.predict(test))
submission = pd.read_csv(SUBMISSON_PATH, index_col='id')

submission[LABEL_COL_NAME] = y_test[:-1]

submission.to_csv(f'submission.csv')

print(submission)