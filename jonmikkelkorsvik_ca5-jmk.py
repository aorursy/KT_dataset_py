

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Standard libraries

import numpy as np

import pandas as pd

from scipy import stats

from copy import copy

import pyprind

import pickle

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

import time





# Plots and visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from mlxtend.plotting import plot_decision_regions

sns.set()



# Regressors

# from sklearn.neighbors import

from sklearn.tree import  DecisionTreeRegressor

from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression, LinearRegression, RANSACRegressor, Lasso, ElasticNet, SGDRegressor, Ridge, LassoCV, RidgeCV

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor





# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, auc, confusion_matrix, mean_squared_error, f1_score, classification_report

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.decomposition import PCA, KernelPCA, NMF

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, StratifiedShuffleSplit, KFold, cross_val_predict, GroupKFold, cross_validate

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.metrics.pairwise import (chi2_kernel, laplacian_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel, cosine_similarity

, linear_kernel)

from sklearn.cross_decomposition import PLSRegression

from sklearn.feature_selection import SelectKBest, chi2





# Alternative modules

import umap

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np



class ShiftEmscTransformer(BaseEstimator, TransformerMixin):#Class Constructor

    def __init__(self, shift = None, reference = None, degree=7):

        if shift is None or reference is None:

            raise ValueError('Need to set shift (list of shifts) and reference (index)')

        self._shift = shift

        self._reference_idx = reference

        self._degree = degree

        self._reference = None

        self._begin = None

        self._end = None



    def EMSC(self, X):

        # Create polynomials up to chosen degree

        poly = [] 

        pvar = [1]

        for i in range(self._degree):

            poly.append( np.polyval(pvar, np.linspace(-1,1,len(self._reference))) ) 

            pvar.append(0)

        # Reference spectrum and polynomials

        emsc_basis = np.vstack([self._reference, np.vstack(poly)])

        # Estimate EMSC parameters

        (params,_,_,_) = np.linalg.lstsq(emsc_basis.T, X.T, rcond=None)

        # Correct and return

        return (X - params[1:,:].T @ emsc_basis[1:,:])/params[:1,:].T





    def cut_values_from_spectrum(self, X):

        res = []

        for spectre in X:

            temp = []

            for ind, val in enumerate(spectre):

                if self._begin < self._shift[0, ind] < self._end:

                    temp.append(val)

                

            temp = np.asarray(temp)

            res.append(temp)

        return np.asarray(res)

        

    def fit(self, X,  begin, end):

        self._begin = begin

        self._end = end

        cut_X = self.cut_values_from_spectrum(X)

        self._reference = cut_X[self._reference_idx]

        

        return self 

    

    #Custom transform method Using EMSC

    def transform(self, X, y = None):

        X = self.cut_values_from_spectrum(X)

        return self.EMSC(X)



    def fit_transform(self, X, begin, end, y=None):

        self.fit(X, begin, end)

        return self.transform(X)

"""def EMSC(X, reference, degree=7):

    # Create polynomials up to chosen degree

    poly = []; pvar = [1]

    for i in range(degree):

        poly.append( np.polyval(pvar,np.linspace(-1,1,len(reference))) )

        pvar.append(0)

    # Reference spectrum and polynomials

    emsc_basis = np.vstack([reference, np.vstack(poly)])

    # Estimate EMSC parameters

    (params,_,_,_) = np.linalg.lstsq(emsc_basis.T, X.T, rcond=None)

    # Correct and return

    return (X - params[1:,:].T @ emsc_basis[1:,:])/params[:1,:].T





def cut_values_from_spectrum(spectra, shift, begin, end):

    res = []

    for spectre in spectra:

        temp = []

        for ind, val in enumerate(spectre):

            if begin < shift[0, ind] < end:

                temp.append(val)

            

        temp = np.asarray(temp)

        res.append(temp)

    return np.asarray(res)

"""



# Simple code for saving result

def save_result(result, name="submission"):

    df = pd.DataFrame(result.astype("float"), columns=["label"])

    df.index.name = "ID"

    df = df.reset_index()

    id_ = df.ID

    df = df.drop("ID", axis=1)

    df.insert(1, "Id", id_)

    df = df.set_index("label")

    df.to_csv(f"{name}.csv")

    df.to_csv(f"{name}.csv")

    print(f"saved succesfully as  - {name}.csv")

    

# These to methods takes advantage of replicate groups for an more accurate prediction.

def take_avg_of_repl_and_save(predictions, rep_df, name="submission"):

    #initiating with pred and mean of each replicate group

    df = pd.DataFrame(predictions.astype("float"), columns=["label"])

    df["group"] = rep_df

    group_means = df.groupby('group').mean()



    label_means = []

    for i, row in df.iterrows():

        label_means.append(float(group_means.iloc[int(row.group)]))

    df["label"] = label_means

    

    #Cleaning up submission



    df= df.drop('group', axis=1)

    save_result(df, name)

    

    

def take_med_of_repl_and_save(predictions, rep_df, name="submission"):

    # Same as above with median

    df = pd.DataFrame(predictions.astype("float"), columns=["label"])

    df["group"] = rep_df

    group_median = df.groupby('group').median()



    label_median = []

    for i, row in df.iterrows():

        label_median.append(float(group_median.iloc[int(row.group)]))

    df["label"] = label_median

    

    

    df= df.drop('group', axis=1)

    save_result(df, name)

    

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

        

def count_series_unique_(series):

    print(f"Number of missing values by column: \n{list(series.isna().sum())}\n")

    print(f"Number of unique labels: \n{series.unique()}\n")

    return series.value_counts()



        

        

# Scorer function         

#scorer = make_scorer(mean_squared_error, greater_is_better=False)



    
train_in = open("/kaggle/input/dat200-ca5-2020/train.pkl","rb")

train = pickle.load(train_in)

#train = pd.DataFrame.from_dict(train)



test_in = open("/kaggle/input/dat200-ca5-2020/test.pkl","rb")

test = pickle.load(test_in)

#test = pd.DataFrame.from_dict(test)

train.keys(), test.keys()
# Shift values for cutting

shift = train['shifts']

# Initiating costum transformer.

transfo = ShiftEmscTransformer(shift, 1343)



spectra_train = train['RamanCal']

spectra_test = test['RamanVal']



pre_proc_spectra_train = transfo.fit_transform(spectra_train, 499, 3101)

pre_proc_spectra_test = transfo.transform(spectra_test)

# Plotting



x = range(500, 3101)

fig = plt.figure(figsize=(20,12))

ax1 = fig.gca()

ax1.set_xlabel('Raman shift [cm-1]'); ax1.set_ylabel('Relative intensity')

ax1.set_title('EMSC Adjusted And Clipped Spectra (Raman spectroscopy)')

ax1.set_yscale('linear')

for spectrum in pre_proc_spectra_train:

    ax1.plot(x, spectrum)

    

plt.show()
# Creating dataframes

train_df = pd.DataFrame.from_records(pre_proc_spectra_train, columns=[f'shift{x}' for x in range(500, 3101)])

test_df = pd.DataFrame.from_records(pre_proc_spectra_test, columns=[f'shift{x}' for x in range(500, 3101)])





X = train_df.copy()

y = train['IodineCal']



X_test = test_df.copy()





train_df.insert(0, "target", y, False) 



train_df.insert(0, "replicategroup", train['repCal'], False)   

test_df.insert(0, "replicategroup", test['repVal'], False)
clfs = []



clfs.append(("RANSACRegressorKernelPCA", 

             Pipeline([("PCA", KernelPCA(n_components=10)),

                       ("RANSACRegressor", RANSACRegressor(random_state=0))])))



clfs.append(("RANSACRegressorPCA", 

             Pipeline([("PCA", PCA(n_components=10)),

                       ("RANSACRegressor", RANSACRegressor(random_state=0))])))





clfs.append(("LinReg", 

             Pipeline([("LinReg", LinearRegression(n_jobs=-1))]))) 



clfs.append(("LinRegPCA", 

             Pipeline([("PCA", PCA(n_components=10)),

                       ("LinReg", LinearRegression(n_jobs=-1))])))



clfs.append(("LinRegKernelPCA", 

             Pipeline([("PCA", KernelPCA(kernel="rbf", n_components=10)),

                       ("LinReg", LinearRegression(n_jobs=-1))])))





clfs.append(("GradientBoostingRegressor", 

             Pipeline([("GradientBoosting", GradientBoostingRegressor(n_estimators=100,

                                                                       random_state=42))]))) 



clfs.append(("PLSRegression", 

             Pipeline([("PLSRegression", PLSRegression(n_components=10))])))



clfs.append(("RidgeCV", 

             Pipeline([("RidgeCV", RidgeCV())])))



#clfs.append(("LassoCV", 

#             Pipeline([("LassoCV", LassoCV())])))



#clfs.append(("MLP", 

#             Pipeline([("MLP", MLPRegressor(hidden_layer_sizes=(100,), solver='adam', random_state=42))])))



clfs.append(("AdaBoostRegressor", 

             Pipeline([("PCA", PCA(n_components=3)),

                       ("AdaBoostRegressor", AdaBoostRegressor())])))



clfs.append(("DecisionTreeRegressor", 

             Pipeline([("PCA", PCA(n_components=10)),

                       (" DecisionTreeRegressor",  DecisionTreeRegressor())])))



clfs.append(("RandomForestRegressor", 

             Pipeline([("PCA", PCA(n_components=10)),

                       (" RandomForestRegressor",  RandomForestRegressor(random_state=1, max_depth=10, n_jobs=-1, verbose=1))])))



clfs.append(("XGBRegressor", 

             Pipeline([(" XGBRegressor",  XGBRegressor(eval_metric = 'rmse',

                                                                eta = 0.1,

                                                                num_boost_round = 80,

                                                                max_depth = 5,

                                                                subsample = 0.8,

                                                                colsample_bytree = 1.0,

                                                                silent = 0,

                                                                ))])))



clfs.append(("SVR", 

             Pipeline([("SVR", SVR())]))) 



clfs.append(("SVRPCA", 

             Pipeline([("PCA", PCA(n_components=10)),

                       ("SVR", SVR())])))



clfs.append(("SVRKernelPCA", 

             Pipeline([("PCA", KernelPCA(kernel="rbf", n_components=10)),

                       ("SVR", SVR())])))





#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'neg_root_mean_squared_error'

results, names  = [], []

group = train_df.replicategroup





gkf = list(GroupKFold( n_splits=5).split(X,y,group))

group_kfold = GroupKFold(n_splits=5).split(X, y, group)
for name, model  in clfs:

    cv_results = cross_val_score(model, X, y,

                                 groups=group,

                                 scoring=scoring,

                                 cv=gkf,

                                 n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())

    print(msg,"\n")

    

   

    

# boxplot algorithm comparison

fig = plt.figure(figsize=(20,10))

fig.suptitle('Regressor Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names, y=results)

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("RMSE of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)



plt.show()
# Best result with median of replicates

simple_pls = PLSRegression(n_components=33)



simple_pls.fit(X, y)

train_pred = simple_pls.predict(X)

result_1 = simple_pls.predict(X_test)

take_avg_of_repl_and_save(result_1, test['repVal'], "sjokoladepls_mean")

take_med_of_repl_and_save(result_1, test['repVal'], "sjokoladepls_median")

#save_result(result_1, "sjokoladepls")
resid = y - train_pred # np.mean(np.abs(y - train_pred))

print(train_pred.shape, resid)

sns.residplot(train_pred, resid)

plt.xlabel("Fitted"); plt.ylabel("Residuals"); plt.title("Residual vs Fitted"); plt.show()
group_kfold = GroupKFold(n_splits=5).split(X, y, group)



n_components = range(10, 50)

#scale = [True, False]

max_iter = [500]



hyperparam = dict(n_components = n_components,

                 #scale = scale,

                 max_iter = max_iter)



gridPLS = GridSearchCV(PLSRegression(), hyperparam, cv = group_kfold, verbose = 1, scoring=scoring, 

                      n_jobs = -1)



best_PLS = gridPLS.fit(X, y)
print(f"Best parameters: {best_PLS.best_params_} \n \nBest Estimator Score: {best_PLS.best_score_}")
group_kfold = GroupKFold(n_splits=5).split(X, y, group)



pipe_GBR = Pipeline([

    # the reduce_dim stage is populated by the param_grid

    #('scaler', StandardScaler()),

    ('reduce_dim', 'passthrough'),

    ('clf', GradientBoostingRegressor())

])



N_FEATURES_OPTIONS = range(3, 15)

n_estimators = range(50, 100)

learning_rate = np.arange(0.1, 1, 6)

param_grid = [

    {

        'reduce_dim': [PCA(), umap.UMAP(n_neighbors=7), PLSRegression()],

        'reduce_dim__n_components': N_FEATURES_OPTIONS,

        'clf__n_estimators': n_estimators,

        'clf__learning_rate': learning_rate

    },

]

                              

random_pipe_GBR = RandomizedSearchCV(estimator=pipe_GBR, param_distributions=param_grid, scoring=scoring,

                                  refit=True, n_iter = 130, cv = group_kfold, verbose=True, random_state=1, n_jobs = -1)

best_GBR = random_pipe_GBR.fit(X, y)                              
print(f"Best parameters: {random_pipe_GBR.best_params_} \n \nBest Estimator Score: {random_pipe_GBR.best_score_}")
res_GBR = best_GBR.predict(X_test)

take_med_of_repl_and_save(res_GBR, test['repVal'], "GBR_median")
# Ensemble regressors, voting and stacking

from sklearn.ensemble import VotingRegressor, StackingRegressor



xgb = XGBRegressor(eval_metric = 'rmse',

                                        eta = 0.1,

                                        num_boost_round = 80,

                                        max_depth = 5,

                                        subsample = 0.8,

                                        colsample_bytree = 1.0,

                                        silent = 0,

                                        random_state=1

                    )

                    

gbr = GradientBoostingRegressor(random_state=41, n_estimators=100)



pls = PLSRegression(n_components=33)



er = VotingRegressor([

                      ('xgb', xgb),  

                      ("pls", pls), 

                      ("gbr", gbr)

                     ])

                     

xgb.fit(X, y)

gbr.fit(X, y)

pls.fit(X, y)

er.fit(X, y)







sr = StackingRegressor(estimators=[ 

                                    ("pls", pls), 

                                    ("gbr", gbr),

                                    ('xgb', xgb)

                                   ], 

                        final_estimator=RidgeCV())



sr.fit(X, y)

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):

    """Scatter plot of the predicted vs true targets."""

    ax.plot([y_true.min(), y_true.max()],

            [y_true.min(), y_true.max()],

            '--r', linewidth=2)

    ax.scatter(y_true, y_pred, alpha=0.2)



    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()

    ax.get_yaxis().tick_left()

    ax.spines['left'].set_position(('outward', 10))

    ax.spines['bottom'].set_position(('outward', 10))

    ax.set_xlim([y_true.min(), y_true.max()])

    ax.set_ylim([y_true.min(), y_true.max()])

    ax.set_xlabel('Measured')

    ax.set_ylabel('Predicted')

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,

                          edgecolor='none', linewidth=0)

    ax.legend([extra], [scores], loc='upper left')

    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)

    ax.set_title(title)
estimators = [ ("pls", pls), ("gbr", gbr), ('xgb', xgb) ]

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs = np.ravel(axs)



for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', # Rename and switch to: er, for votingregressor

                                               sr)]):    # Takes A really long time to calculate because of CV score and predict, and wide data. The estimators also are heavy. 

    start_time = time.time()

    score = cross_validate(est, X, y,

                           scoring=['r2', 'neg_mean_absolute_error'],

                           n_jobs=-1, verbose=0)

    elapsed_time = time.time() - start_time 



    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)

    plot_regression_results(

        ax, y, y_pred,

        name,

        (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')

        .format(np.mean(score['test_r2']),

                np.std(score['test_r2']),

                -np.mean(score['test_neg_mean_absolute_error']),

                np.std(score['test_neg_mean_absolute_error'])),

        elapsed_time)



plt.suptitle('Single predictors versus stacked predictors')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

plt.show()
n=1



result_sr = sr.predict(X_test)

take_avg_of_repl_and_save(result_sr, test['repVal'], f"{n}sr_mean")

take_med_of_repl_and_save(result_sr, test['repVal'], f"{n}sr_median")
# Install required libraries

!pip install --upgrade pip

!pip install kaggle --upgrade
# !kaggle competitions submit -c dat200-ca5-2020 -f 1sr_mean.csv -m "testing API"

%mkdir --parents /root/.kaggle/

%cp /kaggle/input/kagglejson/kaggle.json ~/kaggle.json /root/.kaggle



!kaggle competitions submit -c dat200-ca5-2020 -f 1sr_mean.csv -m "testing API"
# Loop used for finding best result for PLS

"""for n in range(10, 50):

    simple_pls = PLSRegression(n_components=n)



    simple_pls.fit(X, y)

    result_1 = simple_pls.predict(X_test)

    take_avg_of_repl_and_save(result_1, test['repVal'], f"{n}sjokoladepls_mean")

    take_med_of_repl_and_save(result_1, test['repVal'], f"{n}sjokoladepls_median")""" 