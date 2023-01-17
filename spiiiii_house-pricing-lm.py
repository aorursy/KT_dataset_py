from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import itertools

import functools

import time



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



import seaborn as sns



import sklearn

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import roc_curve, make_scorer, mean_squared_error

from sklearn.preprocessing import RobustScaler



from mlxtend.regressor import StackingCVRegressor



import lightgbm as lgb

import xgboost as xgb
%%script false --no-raise-error

train = pd.read_csv('./train.csv')

test = pd.read_csv('./test.csv')
# %%script false --no-raise-error

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


train_ids = train.Id

test_ids = test.Id

train_y = train.pop('SalePrice')


def plotting_3_chart(df, feature=None):

    ## Importing seaborn, matplotlab and scipy modules. 

    import seaborn as sns

    import matplotlib.pyplot as plt

    import matplotlib.gridspec as gridspec

    from scipy import stats

    import matplotlib.style as style

    style.use('fivethirtyeight')

    

    if type(df) == pd.Series:

        print("it's series")

        to_plot = df

    else:

        print("it's dataframe")

        to_plot = df.loc[:,feature]



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,7))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(to_plot, norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(to_plot, plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(to_plot, orient='v', ax = ax3 );


def allyficy (train, test):

    return pd.concat([train, test]).reset_index(drop=True)


class Get_rid_of_Nans(BaseEstimator, TransformerMixin):

    def __init__ (self):

        pass

    

    def fit (self, X, y=None):

        return self

    

    def Stringificy (self, X):

        # make int features to string ones should be

        X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        X['YrSold'] = X.YrSold.astype(str)

        X['YearRemodAdd'] = X.YearRemodAdd.astype(str)

        X['MoSold']= X['MoSold'].astype(str)

        X['YearBuilt']= X['YearBuilt'].astype(str)

        

    def Nans_from_str (self, X):

        # fill nas in string where should be zero

        missing_val_col = ["Alley", "PoolQC", "MiscFeature", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", 

                           "GarageCond", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

        for i in missing_val_col:

            X[i] = X[i].fillna('None')

        # fill nas in strings

        X['Electrical'] = X['Electrical'].fillna("SBrkr")

        X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])

        X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])

        X['Functional'] = X['Functional'].fillna(X['Functional'].mode()[0])

        X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])

        X['Utilities'] = X['Utilities'].fillna(X['Utilities'].mode()[0])

        X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])

            

    def Nans_from_int (self, X):

        # fill nas where should be zero

        ## These features are continous variable, we used "0" to replace the null values. 

        missing_val_col2 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',

                    'GarageArea', 'GarageCars', 'MasVnrArea']

        for i in missing_val_col2:

            X[i] = X[i].fillna(0)

            

    def Nans_from_float (self, X):

        # fill nas in floats

        X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))



    

    def transform (self, X, y=None):

        self.X = X.copy()

        self.Stringificy(self.X)

        self.Nans_from_int(self.X)

        self.Nans_from_str(self.X)

        self.Nans_from_float(self.X)

        return self.X

    

    def fit_transform (self, X, y=None):

        return self.fit(X).self.transform(X)


class Create_dummies (TransformerMixin):

    def __init__ (self):

        pass

    

    def fit (self):

        return self

    

    def transform (self, df):

        return pd.get_dummies(df).reset_index(drop=True)


class Overfit_reducer (TransformerMixin):

    def __init__ (self):

        pass

    

    def overfit_reducer(self, df):

        """

        This function takes in a dataframe and returns a list of features that are overfitted.

        """

        overfit = []

        for i in df.columns:

            counts = df[i].value_counts()

            zeros = counts.iloc[0]

            if zeros / len(df) * 100 > 99.94:

                overfit.append(i)

        overfit = list(overfit)

        print('list of overfitted features: ',overfit)

        return overfit

 

    

    def fit (self):

        return self

    

    def transform (self, df):

        return df.drop(self.overfit_reducer(df), axis=1)


class Add_new_features(TransformerMixin):

    def __init__ (self):

        pass

    

    def add_features (self, df):

        # feture engineering a new feature "TotalFS"

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        df['YrBltAndRemod'] = df['YearBuilt']+df['YearRemodAdd']



        df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +

                                         df['1stFlrSF'] + df['2ndFlrSF'])



        df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +

                                       df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))



        df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +

                                      df['EnclosedPorch'] + df['ScreenPorch'] +

                                      df['WoodDeckSF'])



        df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

        df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

        df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

        df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

        df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        

        return df

    

    def fit (self):

        return self

    

    def transform (self, df):

        return self.add_features(df)


class Separate (TransformerMixin):

    def __init__ (self, y):

        self.y =y

    

    def fit (self):

        return self

    

    def transform (self, X):

        return {'train': X.iloc[:len(self.y),:], 'test': X.iloc[len(train_y):,:]}

    

    

class Numerificy (TransformerMixin):

    def __init__ (self):

        pass

    

    def fit (self):

        return self

    

    def transform (self, X):

        train_num = X['train'].select_dtypes(exclude="object")

        test_num = X['test'].select_dtypes(exclude='object')

        return {'train': train_num, 'test': test_num}





# preprocessing data

model_1st_without_nans_and_strings = Pipeline([('without_nans', Get_rid_of_Nans()),  

                                               ('dummyficeted', Create_dummies()),

                                ('overfit_reduced', Overfit_reducer()),

                                ('separated', Separate(train_y)),

                                ('numyficyed', Numerificy())])



all_data = allyficy(train, test)

train_X, test_X = model_1st_without_nans_and_strings.transform(all_data).values()


# preprocessing data + RobustScaler

r = RobustScaler().fit(train_X)

train_X_robust, test_X_robust = r.transform(train_X), r.transform(test_X)
#my inplementation of rmsle metrc

def rmsle(clf, X, y):

    y_ = clf.predict(X)

    return -np.sqrt(mean_squared_error(np.log1p(y),np.log1p(y_)))
def plot_functions_from_params(all_params, scores):

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([i['alpha'] for i in all_params if i['max_iter']==10], [i['l1_ratio'] for i in all_params[::2]], scores[::2], c=scores[::2])
predict_LM = Pipeline([('model', LinearRegression())])

predict_model_Elastic = Pipeline([('model', ElasticNet())])





class Tune_XGB:

    def __init__(self, train_X, train_y, test_X, base_params=None):

        self.train_X, self.train_y, self.test_X = train_X.copy(), train_y.copy(), test_X.copy()

        FOLDS = 5

        EARLY_STOP = 50

        MAX_ROUNDS = 5

        if base_params == None:

            self.base_params =  {

                'n_estimators': 100,

                

                'max_depth': -1,

                'min_child_weight': 0,

                

                'gamma': 10,

                

                'subsample': 0.8,

                'colsample_bytree': 0.5,

                

                'reg_alpha': 0.0,

                'reg_lambda': 1,

                

                'learning_rate': 0.01,



            

            'objective': 'reg:linear',

            'silent': 1,

            'seed': 42,      

            'verbosity':0}

            

        else:

            self.base_params = base_params

            



    def update(self, base_dict, update_copy):

        for key in update_copy.keys():

            base_dict[key] = update_copy[key]

            

    def Step_0_find_n_estimators(self, n_estimators):

        print(n_estimators)

        test0 = {

            'n_estimators': n_estimators,

        }

        return self.grid_search(self.base_params, test0)

    

    def Step_05_find_n_estimators(self, test05):

        curent = self.base_params['n_estimators']

        

        test05 = {

            'n_estimators': [int(curent/1.3), int(curent/1.1), curent, int(curent/0.9), int(curent/0.7)]

        }

        return self.grid_search(self.base_params, test05)

    





    def Step_1_find_depth_and_child(self, test1):



        return self.grid_search(self.base_params, test1)



    def Step_2_narrow_depth(self, test2):

        max_depth = self.base_params['max_depth']

        test2 = {

            'max_depth': [max_depth-1,max_depth,max_depth+1]

        }

        return self.grid_search(self.base_params, test2)



    def Step_3_gamma(self, test3):

        return self.grid_search(self.base_params, test3)



    def Step_4_sample(self, test4):

        return self.grid_search(self.base_params, test4)



    def Step_5_reg1(self, test5):

        return self.grid_search(self.base_params, test5)

    

    def Step_6_eta_nround(self, test6):

        return self.grid_search(self.base_params, test6)

    

    



    def train_test(self, X, y):

        train_X_v, test_X_v, train_y_v, test_y_v = train_test_split(X, y, test_size=0.2)

        return train_X_v, test_X_v, train_y_v, test_y_v

    

    def validate(self, model):

        X,x, Y,y = self.train_test(self.train_X, self.train_y)

        model.fit(X, Y, 

                eval_set=[(X, Y), (x, y)],

                eval_metric='rmse',

                verbose=False)

        train_pred = model.predict(X)

        test_pred = model.predict(x)

        

        train_score = mean_squared_error(train_pred, Y)

        test_score = mean_squared_error(test_pred, y)

        

        error = model.evals_result()

        eval_score = error['validation_0']['rmse'][-1]

        

#         self.draw(error)

        return test_score, train_score, eval_score

            

 

    def grid_search(self, base_params, grid):

        print("Starn Tuning: ", grid)

        if self.mode==1:

            return self.grid_search_parallel(base_params, grid)

        else:

            keys = set(grid.keys())

            l = [grid[x] for x in keys]

            perm = list(itertools.product(*l))

            jobs = []

            for i in perm:

                jobs.append({k:v for k,v in zip(keys,i)})



            test_score = []

            train_score = []

            eval_score = []



            for i, job in enumerate(jobs):

                base_params = self.base_params.copy()

                print(job)

                self.update(base_params, job)

                model = xgb.XGBRegressor(**base_params)

                score = self.validate(model)

                test_score.append((job, score[0]))

                train_score.append((job, score[1]))

                eval_score.append((job, score[2]))

            return {'train_scores':train_score, 'test_scores':test_score, 'eval_scores':eval_score}

        

    def grid_search_parallel(self, base_params, grid):

        keys = set(grid.keys())

        l = [grid[x] for x in keys]

        perm = list(itertools.product(*l))

        jobs = []

        for i in perm:

            jobs.append({k:v for k,v in zip(keys,i)})



        test_score = []

        train_score = []

        eval_score = []



        e = ProcessPoolExecutor()





        score = list(e.map(self.build_val ,jobs))

        for i, scor in enumerate(score):

            test_score.append((jobs[i], scor[0]))

            train_score.append((jobs[i], scor[1]))

            eval_score.append((jobs[i], scor[2]))



        return {'train_scores':train_score, 'test_scores':test_score, 'eval_scores':eval_score}



    def build_val(self, job):

        base_params = self.base_params.copy()

        self.update(base_params, job)

        model = xgb.XGBRegressor(**base_params)

        score = self.validate(model)

        return(score)

    

    def draw(self, errors=None):

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(range(len(errors['validation_0']['rmse'])), errors['validation_0']['rmse'])

        plt.show()

            

    def draw_all(self, errors):

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(range(len(errors['train_scores'])),[i[1] for i in errors['train_scores']], label='train')

        ax.plot(range(len(errors['test_scores'])),[i[1] for i in errors['test_scores']], label='test')

        ax.plot(range(len(errors['eval_scores'])),[i[1] for i in errors['eval_scores']], label='eval_scores')

        ax.legend()

        plt.show()

        

    def start_tuning(self, num, mode=0):

        self.mode=mode

        print("loading")

        start_time = time.time()

        base_params = dict(self.base_params)

        

        steps = {self.Step_0_find_n_estimators:[100, 200, 400, 800,1600,3200],

            self.Step_05_find_n_estimators:None,

            self.Step_1_find_depth_and_child:{'max_depth': [10, 14, 18, 22, -1], 

                                           'min_child_weight': list(range(1,10,2))},

            self.Step_2_narrow_depth: None,

            self.Step_3_gamma:{'gamma': list([i/10.0 for i in range(0,5)])},

            self.Step_4_sample: {'subsample': list([i/10.0 for i in range(6,10)]),

                              'colsample_bytree': list([i/10.0 for i in range(6,10)])},

            self.Step_5_reg1:{'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],

                             'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]},

            self.Step_6_eta_nround:{'learning_rate':[0.001, 0.005, 0.01, 0.05, 0.1, 0.3]},

                }

        

        for i,key in enumerate(steps.keys()):

            if i in range(num):

                e = ProcessPoolExecutor

                result = key(steps[key])

                self.draw_all(result)

                best_param = min(result['test_scores'], key=lambda x:x[1])

            

                print('best selected param:', best_param)

                self.update(self.base_params, best_param[0])

                print(f'params after step {i}:', self.base_params)





        print("FINAL RESUTL: {}".format(self.base_params))

        elapsed_time = time.time() - start_time

        

       

        

print('done')

tune = Tune_XGB(train_X_robust, train_y, test_X_robust)
class Tune_ligthGBM:

    def __init__(self, train_X, train_y, test_X, base_params=None):

        self.train_X, self.train_y, self.test_X = train_X.copy(), train_y.copy(), test_X.copy()

        if base_params == None:

            self.base_params =  {'n_estimators':100,

                

                                'max_depth':7,                                 

                                 'num_leaves': 31,

                                 'min_data_in_leaf':20,

                                 

                                'bagging_fraction':0.75,

                                'bagging_freq':5, 

                                'bagging_seed':7,                                 

                                'feature_fraction':0.8,

                                'feature_fraction_seed':7,

                                 

                                'reg_alpha': 0.0,

                                'reg_lambda': 1,

                                 

                                 'learning_rate':0.01,

                            

                                 'objective':'regression', 

                            'num_leaves':4,

                            'max_bin':200, 

                            

                            

                            'verbose':-1,}

        else:

            self.base_params = base_params

            



    def update(self, base_dict, update_copy):

        for key in update_copy.keys():

            base_dict[key] = update_copy[key]

            

    def Step_0_find_n_estimators(self, n_estimators):

        print(n_estimators)

        test0 = {

            'n_estimators': n_estimators,

        }

        return self.grid_search(self.base_params, test0)

    

    def Step_05_find_n_estimators(self, test05):

        curent = self.base_params['n_estimators']

        

        test05 = {

            'n_estimators': [int(curent/1.3), int(curent/1.1), curent, int(curent/0.9), int(curent/0.7)]

        }

        return self.grid_search(self.base_params, test05)

    

    def Step_06_min_child_weight(self, teset06):

        print(teset06)

        return self.grid_search(self.base_params, teset06)

    

    def Step_07_min_child_weight_closer(self, teset07):

        curent = self.base_params['min_child_weight']

        if curent>=4:

            teset07={

                'min_child_weight':[int(curent/1.3),curent, int(curent/0.7)]

            }

        else:

            teset07={'min_child_weight': [curent]}



        return self.grid_search(self.base_params, teset07)



    def Step_1_find_depth_and_child(self, test1):



        return self.grid_search(self.base_params, test1)



    def Step_2_narrow_depth(self, test2):

        max_depth = self.base_params['max_depth']

        test2 = {

            'max_depth': [max_depth-1,max_depth,max_depth+1]

        }

        return self.grid_search(self.base_params, test2)





    def Step_4_sample(self, test4):

        return self.grid_search(self.base_params, test4)



    def Step_5_reg1(self, test5):

        return self.grid_search(self.base_params, test5)

    

    def Step_6_eta_nround(self, test6):

        return self.grid_search(self.base_params, test6)



    def train_test(self, X, y):

        train_X_v, test_X_v, train_y_v, test_y_v = train_test_split(X, y, test_size=0.2)

        return train_X_v, test_X_v, train_y_v, test_y_v

    

    def validate(self, model):

        X,x, Y,y = self.train_test(self.train_X, self.train_y)

        model.fit(X, Y, 

                eval_set=[(X, Y), (x, y)],

                eval_metric='rmse',

                verbose=False)

        train_pred = model.predict(X)

        test_pred = model.predict(x)

        

        train_score = mean_squared_error(train_pred, Y)

        test_score = mean_squared_error(test_pred, y)

        

        

#         self.draw(error)

        return test_score, train_score

            

 

    def grid_search(self, base_params, grid):

        print("Starn Tuning: ", grid)

        if self.mode==1:

            return self.grid_search_parallel(base_params, grid)

        else:

            keys = set(grid.keys())

            l = [grid[x] for x in keys]

            perm = list(itertools.product(*l))

            jobs = []

            for i in perm:

                jobs.append({k:v for k,v in zip(keys,i)})



            test_score = []

            train_score = []



            for i, job in enumerate(jobs):

                base_params = self.base_params.copy()

                print(job)

                self.update(base_params, job)

                model = lgb.LGBMRegressor(**base_params)

                score = self.validate(model)

                test_score.append((job, score[0]))

                train_score.append((job, score[1]))

                eval_score.append((job, score[2]))

            return {'train_scores':train_score, 'test_scores':test_score}

        

    def grid_search_parallel(self, base_params, grid):

        keys = set(grid.keys())

        l = [grid[x] for x in keys]

        perm = list(itertools.product(*l))

        jobs = []

        for i in perm:

            jobs.append({k:v for k,v in zip(keys,i)})



        test_score = []

        train_score = []



        e = ProcessPoolExecutor()

        

        score = list(map(self.build_val ,jobs))

        for i, scor in enumerate(score):

            test_score.append((jobs[i], scor[0]))

            train_score.append((jobs[i], scor[1]))



        return {'train_scores':train_score, 'test_scores':test_score}



    def build_val(self, job):

        base_params = self.base_params.copy()

        self.update(base_params, job)

        model = lgb.LGBMRegressor(**base_params)

        score = self.validate(model)

        return(score)

    

    def draw(self, errors=None):

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(range(len(errors['validation_0']['rmse'])), errors['validation_0']['rmse'])

        plt.show()

            

    def draw_all(self, errors):

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(range(len(errors['train_scores'])),[i[1] for i in errors['train_scores']], label='train')

        ax.plot(range(len(errors['test_scores'])),[i[1] for i in errors['test_scores']], label='test')

        ax.legend()

        plt.show()

        

    def start_tuning(self, num, mode=0):

        self.mode=mode

        print("loading")

        start_time = time.time()

        base_params = dict(self.base_params)

        

        steps = {self.Step_0_find_n_estimators:[100, 200, 400, 800, 1600, 3200],

            self.Step_05_find_n_estimators:None,

            

            self.Step_1_find_depth_and_child:{'max_depth': list(range(10,20,2)) ,                                 

                                             'num_leaves': [20, 31, 40],

                                             'min_data_in_leaf':[10, 15, 20, 25, 30]},

            self.Step_2_narrow_depth: None,

                 

            self.Step_4_sample: {'bagging_fraction':[0.2, 0.4, 0.6, 0.8],

                                'feature_fraction': [0.2, 0.4, 0.6, 0.8]},

            self.Step_5_reg1:{'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],

                             'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]},

            self.Step_6_eta_nround:{'learning_rate':[0.001, 0.005, 0.01, 0.05, 0.1, 0.3]}

                }

        for i,key in enumerate(steps.keys()):

            if i in range(num):

                e = ProcessPoolExecutor

                result = key(steps[key])

                self.draw_all(result)

                best_param = min(result['test_scores'], key=lambda x:x[1])

            

                print('best selected param:', best_param)

                self.update(self.base_params, best_param[0])

                print(f'params after step {i}:', self.base_params)





        print("FINAL RESUTL: {}".format(self.base_params))

        elapsed_time = time.time() - start_time

        

       

        

print('done')

tune_l = Tune_ligthGBM(train_X_robust, train_y, test_X_robust)
%%script false --no-raise-error

tune_l.start_tuning(10, mode=1)
%%script false --no-raise-error

tune.start_tuning(10, mode=1)

# XGB params 



my_local1 = {'objective': 'reg:linear', 

          'learning_rate': 0.01, 

          'seed': 42, 

          'n_estimators': 3200, 

          'max_depth': 5, 

          'min_child_weight': 9, 

          'gamma': 0.0, 

          'subsample': 0.6, 

          'colsample_bytree': 0.6, 

          'reg_alpha': 0.01} # 0.13015



my_local_eval={'n_estimators': 4571, 'num_round': 50, 'max_depth': 14, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0.01, 

               'reg_lambda': 1e-05, 'learning_rate': 0.0005, 'objective': 'reg:linear',  'seed': 42, 'verbosity': 0} #  0.13015



my_local_test_1_log =  {'n_estimators': 4571, 'num_round': 50, 'max_depth': 6, 'min_child_weight': 11, 'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 0.01, 

                        'reg_lambda': 0.1, 'learning_rate': 0.01, 'objective': 'reg:linear', 'seed': 42, 'verbosity': 0} #0.12847









# lgbm models



my_local_l2 = {'n_estimators': 6400, 

               'max_depth': -1, 

               'num_leaves': 40, 

               'min_data_in_leaf': 15, 

               'bagging_fraction': 0.6,

               'bagging_freq': 5, 

               'bagging_seed': 7, 

               'feature_fraction': 0.8, 

               'feature_fraction_seed': 7, 

               'reg_alpha': 1e-05, 

               'reg_lambda': 1, 

               'learning_rate': 0.01,

               'objective': 'regression',

               'max_bin': 200,

               'verbose': -1} # 0.13121





my_local_l4 = {'n_estimators': 2909, 'max_depth': 8, 'num_leaves': 20, 'min_data_in_leaf': 20, 

               'bagging_fraction': 0.8, 'bagging_freq': 5, 'bagging_seed': 7, 

               'feature_fraction': 0.6, 'feature_fraction_seed': 7, 'reg_alpha': 0.1,

               'reg_lambda': 1, 'learning_rate': 0.01, 

               'objective': 'regression', 'max_bin': 200, 'verbose': -1} #  0.12883



my_local_l5 = {'n_estimators': 1777, 'max_depth': 10, 'num_leaves': 40, 'min_data_in_leaf': 10, 'bagging_fraction': 0.6, 'bagging_freq': 5, 'bagging_seed': 7, 

          'feature_fraction': 0.8, 'feature_fraction_seed': 7, 'reg_alpha': 1e-05, 'reg_lambda': 1e-05, 'learning_rate': 0.005, 'objective': 'regression',

               'max_bin': 200, 'verbose': -1} #0.12808



test_l = {'n_estimators': 6400, 'max_depth': 17, 'num_leaves': 40, 'min_data_in_leaf': 15, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'bagging_seed': 7,

          'feature_fraction': 0.4, 'feature_fraction_seed': 7, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'learning_rate': 0.05, 'objective': 'regression', 

          'max_bin': 200, 'verbose': -1} #0.12947
# elastic params

base = 0.39823



my_first = {'l1_ratio': 0.17, 'alpha': 0.01, 'max_iter': 12} # 0.14057

my_second = {'max_iter': 500.0, 'l1_ratio': 0.16, 'alpha': 0.01}#  0.13960

my_therd = {'max_iter': 50.0, 'l1_ratio': 0.16, 'alpha': 0.01} #0.13951

test_el = {'max_iter': 50.0, 'l1_ratio': 0.16, 'alpha': 0.01} #0.13266



%%script false --no-raise-error

# GreedSearch with elasticNet

parametersGrid = {"max_iter": [5, 10],

                      "alpha": np.arange(0.14, 0.18, 0.005),

                      "l1_ratio": np.arange(0.5, 0.62, 0.005)}

model = ElasticNet()

grid_search = GridSearchCV(model, parametersGrid, verbose=0)

grid_search.fit(train_X, train_y)





best_params = grid_search.best_params_

all_params = grid_search.cv_results_['params']

scores = grid_search.cv_results_['mean_test_score']





predict = grid_search.predict(test_X)

plot_functions_from_params(all_params, scores)

best_params
%%script false --no-raise-error

# GreedSearch with elasticNet + RobustScaler

parametersGrid = {"max_iter": [5, 10],

                      "alpha": np.arange(0.1, 0.2, 0.005),

                      "l1_ratio": np.arange(0.3, 0.45, 0.005)}

model = ElasticNet()

grid_search_robust = GridSearchCV(model, parametersGrid, verbose=0)

grid_search_robust.fit(train_X_robust, train_y)





best_params_rubost = grid_search_robust.best_params_

all_params_rubost = grid_search_robust.cv_results_['params']

scores_rubost = grid_search_robust.cv_results_['mean_test_score']





predict = grid_search_robust.predict(test_X_robust)



plot_functions_from_params(all_params_rubost, scores_rubost)

best_params_rubost
%%script false --no-raise-error

xg_model = xgb.XGBRegressor(**my_local_test_1_log, n_jobs = 4)

# xg_model.fit(train_X, np.log1p(train_y))

# predict_x = xg_model.predict(test_X)

# predict_x = np.expm1(predict_x)
# %%script false --no-raise-error

lightgbm = lgb.LGBMRegressor(**test_l, n_jobs = 4)

lightgbm.fit(train_X, np.log1p(train_y))

predict_l = lightgbm.predict(test_X)

predict_l = np.expm1(predict_l)
%%script false --no-raise-error

elastic = ElasticNet(**test_el)

elastic_pipe = make_pipeline(RobustScaler(), elastic)

# elastic_pipe.fit(train_X, np.log1p(train_y))

# predict_e = elastic_pipe.predict(test_X)

# predict_e = np.expm1(predict_e)
%%script false --no-raise-error

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def cv_rmse(model, X=train_X, y=np.log1p(train_y)):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



lasso_pipe = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds,n_jobs=4))



# lasso_pipe.fit(train_X, np.log1p(train_y))

# predict_las = lasso_pipe.predict(test_X)

# predict_las = np.expm1(predict_las)
%%script false --no-raise-error

meta_xgboost = xgb.XGBRegressor(n_jobs=4)

stack = StackingCVRegressor(regressors=[elastic_pipe, lightgbm, xg_model],meta_regressor=meta_xgboost)

stack.fit(np.array(train_X), np.log1p(train_y))

predict_s = stack.predict(np.array(test_X))

predict_s = np.expm1(predict_s)


predict = predict_l
sub = pd.concat([test_ids, pd.Series(predict)], axis=1)

sub = sub.rename(columns={0:'SalePrice'})

sub.to_csv('sumbission.csv', index=False, header=True)
print(sub)

print('FINISHED')