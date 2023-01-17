from IPython.display import HTML

from IPython.display import display

baseCodeHide="""

<style>

.button {

    background-color: #008CBA;;

    border: none;

    color: white;

    padding: 8px 22px;

    text-align: center;

    text-decoration: none;

    display: inline-block;

    font-size: 16px;

    margin: 4px 2px;

    cursor: pointer;

}

</style>

 <script>

   // Assume 3 input cells. Manage from here.

   var divTag0 = document.getElementsByClassName("input")[0]

   var displaySetting0 = divTag0.style.display;

   // Default display - set to 'none'.  To hide, set to 'block'.

   // divTag0.style.display = 'block';

   divTag0.style.display = 'none';

   

   var divTag1 = document.getElementsByClassName("input")[1]

   var displaySetting1 = divTag1.style.display;

   // Default display - set to 'none'.  To hide, set to 'block'.

      divTag1.style.display = 'block';

   //divTag1.style.display = 'none';

   

   var divTag2 = document.getElementsByClassName("input")[2]

   var displaySetting2 = divTag2.style.display;

   // Default display - set to 'none'.  To hide, set to 'none'.

   divTag2.style.display = 'block';

   //divTag2.style.display = 'none';

 

    function toggleInput(i) { 

      var divTag = document.getElementsByClassName("input")[i]

      var displaySetting = divTag.style.display;

     

      if (displaySetting == 'block') { 

         divTag.style.display = 'none';

       }

      else { 

         divTag.style.display = 'block';

       } 

  }  

  </script>

  <!-- <button onclick="javascript:toggleInput(0)" class="button">Show Code</button> -->

"""

h=HTML(baseCodeHide)





display(h)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')

import gc





# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import matplotlib.patches as patches

from scipy import stats

from scipy.stats import skew



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 100)



py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os,random, math, psutil, pickle

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')
long_df = pd.read_csv("/kaggle/input/mri-and-alzheimers/oasis_longitudinal.csv")

cross_df = pd.read_csv("/kaggle/input/mri-and-alzheimers/oasis_cross-sectional.csv")



def split(df, test_size=0.2):

    #df = df.sample(frac=1, random_state=6) # Shuffle the df.

    split_index = int(len(df)*test_size)

    return df[:-split_index], df[-split_index:]



long_train, long_val = split(long_df)

cross_train, cross_val = split(cross_df)
from math import isnan



# Preserving the NaNs so they can be imputed later.

def binarize_cdr(x):

    if isnan(x):

        return x

    else:

        return 1 if x > 0 else 0



# There are no NaNs in the Group column.

def binarize_group(x):

    return 1 if x != 'Nondemented' else 0



long_train['demented'] = long_train['Group'].apply(binarize_group)

cross_train['demented'] = cross_train['CDR'].apply(binarize_cdr)

long_val['demented'] = long_val['Group'].apply(binarize_group)

cross_val['demented'] = cross_val['CDR'].apply(binarize_cdr)
def combine_dfs(long, cross):

    

    # Renaming the education column to match between the two dataframes.

    long = long.rename(columns={'EDUC': 'Educ'})



    # Combining the rows of each person in the Longitudinal into a single one, taking the mean of each feature's values.

    mean_long = long.groupby('Subject ID').mean()



    # Adding the columns of strings back, as they were dropped by the mean() 

    mean_long = pd.merge(mean_long, long[['Subject ID', 'M/F', 'Group']].drop_duplicates(), on='Subject ID', how='right').reset_index(drop=True)



    # Then concatenating the new condensed dataframe with the Cross-sectional.

    total = pd.concat((cross, mean_long)).reset_index(drop=True)

    

    return total, mean_long



total_train, long_train = combine_dfs(long_train, cross_train)

total_val, _ = combine_dfs(long_val, cross_val)



# Dropping from the validation set the rows where the demented feature is NaN, as they cannot be used for validation.

total_val = total_val[~np.isnan(total_val['demented'])]
total_train = total_train.drop(['Visit', 'Hand', 'MR Delay', 'Delay', 'Subject ID', 'ID'], axis=1)

total_val = total_val.drop(['Visit', 'Hand', 'MR Delay', 'Delay', 'Subject ID', 'ID'], axis=1)
fig = plt.figure(figsize=(10,7))

sns.distplot(cross_train[['Age']], hist=False, label='Cross')

sns.distplot(long_train['Age'], hist=False, label='Long', color='g')

g = sns.distplot(total_train['Age'], hist=False, rug=True, kde_kws=dict(linewidth=3), label='Total')

_ = g.set_title('Distribution of ages in the two datasets and their combination.')
fig = plt.figure(figsize=(20,9))

ax = fig.add_subplot(1, 2, 1)

ax.set_title('Cross-sectional')

sns.catplot(x='CDR', y='Age', data=cross_train, kind='swarm', split=True, hue='M/F', ax=ax)

ax = fig.add_subplot(1, 2, 2)

ax.set_title('Longitudinal')

sns.catplot(x='Group', y='Age', data=long_train, kind='swarm', split=True, hue='M/F', ax=ax);

plt.close(2)

plt.close(3)



fig = plt.figure(figsize=(8,6))

g = sns.catplot(x='demented', y='Age', data=total_train, kind='violin', split=True, hue='M/F')

g.ax.set_title('Total data')
fig = plt.figure(figsize=(20,9))

ax = fig.add_subplot(1, 2, 1)

ax.set_title('Cross-sectional')

sns.catplot(x='CDR', y='MMSE', data=cross_train, kind='swarm', hue='M/F', split=True, ax=ax)

ax = fig.add_subplot(1, 2, 2)

ax.set_title('Longitudinal')

sns.catplot(x='Group', y='MMSE', data=long_train, kind='swarm', hue='M/F', split=True, ax=ax);

plt.close(2)

plt.close(3)



fig = plt.figure(figsize=(8,6))

g = sns.catplot(x='demented', y='MMSE', data=total_train, kind='violin', split=True, hue='M/F')

g.ax.set_title('Total data')
fig = plt.figure(figsize=(20,9))

ax = fig.add_subplot(1, 2, 1)

ax.set_title('Cross-sectional')

sns.catplot(x='CDR', y='SES', data=cross_train, kind='swarm', split=False, hue='M/F', ax=ax)

ax = fig.add_subplot(1, 2, 2)

ax.set_title('Longitudinal')

sns.catplot(x='Group', y='SES', data=long_train, kind='swarm', split=False, hue='M/F', ax=ax);

plt.close(2)

plt.close(3)



fig = plt.figure(figsize=(8,6))

g = sns.catplot(x='demented', y='SES', data=total_train, kind='violin', split=True, hue='M/F')

g.ax.set_title('Total data')
fig = plt.figure(figsize=(20,9))

ax = fig.add_subplot(1, 2, 1)

ax.set_title('Cross-sectional')

sns.catplot(x='CDR', y='Educ', data=cross_train, kind='violin', ax=ax)#, split=True, hue='M/F')

ax = fig.add_subplot(1, 2, 2)

ax.set_title('Longitudinal')

sns.catplot(x='Group', y='Educ', data=long_train, kind='violin', ax=ax, split=True, hue='M/F')

# These are needed to suppress the extra plots that seaborn tries to create.

plt.close(2)

plt.close(3)



fig = plt.figure(figsize=(8,6))

g = sns.catplot(x='demented', y='Educ', data=total_train, kind='violin', split=True, hue='M/F')

g.ax.set_title('Total data')
fig = plt.figure(figsize=(15,15))

fig.subplots_adjust(hspace=0.2)

#ax = fig.add_subplot(1, 2, 1)

#_ = sns.distplot(total_train['eTIV']).set_title('eTIV distribution in the whole training data.')

#ax = fig.add_subplot(1, 2, 2)



for i, x in enumerate(['eTIV', 'nWBV', 'ASF']):

    ax = fig.add_subplot(2, 2, i+1)

    sns.distplot(cross_train[x], hist=False, label='Cross', ax=ax)

    sns.distplot(long_train[x], hist=False, label='Long', color='g', ax=ax)

    g = sns.distplot(total_train[x], hist=False, rug=True, kde_kws=dict(linewidth=3), label='Total', ax=ax)

    _ = g.set_title('Distribution of {} in the two datasets and their combination.'.format(x))



# These are needed to suppress the extra plots that seaborn tries to create.

plt.close(2)

plt.close(3)



#fig = plt.figure(figsize=(8,6))

#_ = sns.distplot(total_train['ASF']).set_title('Atlas scaling factor in the whole training data.')
# Plot a correlation heatmap with all the dataset's features.

plt.figure(figsize=(14, 8))

corr = total_train.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, annot=True, cmap='bwr')
features_to_drop = ['Group', 'CDR', 'ASF', 'eTIV', 'nWBV']

total_train, total_val = total_train.drop(features_to_drop, axis=1), total_val.drop(features_to_drop, axis=1) 
total_train['M/F'], total_val['M/F'] = pd.get_dummies(total_train['M/F']), pd.get_dummies(total_val['M/F'])
import missingno as msno

msno.matrix(total_train)



###############################

# Can also use describe(data)

###############################
h1_train, h1_val = total_train.copy(), total_val.copy()





from sklearn_pandas import CategoricalImputer

ci = CategoricalImputer().fit(h1_train.to_numpy())

h1_train.loc[:,:] = ci.transform(h1_train.to_numpy())

h1_val.loc[:,:] = ci.transform(h1_val.to_numpy()) 



#i2_train.loc[:,:] = CategoricalImputer().fit(i2_train.to_numpy()).transform(i2_train.to_numpy())

#i2_val.loc[:,:] = CategoricalImputer().fit(i2_val.to_numpy()).transform(i2_val.to_numpy()) 
h2_train, h2_val = total_train.copy(), total_val.copy()





from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



imp = IterativeImputer(max_iter=10, random_state=0).fit(h2_train)

h2_train.loc[:,:] = imp.transform(h2_train)

h2_val.loc[:,:] = imp.transform(h2_val)
h2_train['demented'].unique()
def binarize_imputed(total, cross):

    # Find the ratio of people in the training set (excluding the ones with NaNs) that do not have dementia.

    nonnan_demented = cross['demented'].dropna()

    demented_ratio = nonnan_demented.value_counts()[1] / len(nonnan_demented)



    # Getting the imputed values of the target column and ordering them ascending.

    sorted_imputed = cross.loc[cross['demented'].isna(), 'demented'].sort_values()

    to_zeros, to_ones = split(sorted_imputed, demented_ratio)



    # Conducting the mapping.

    total.loc[to_zeros.index, 'demented'] = 0

    total.loc[to_ones.index, 'demented'] = 1

    return total['demented']



h2_train['demented'] = binarize_imputed(h2_train, cross_train)
# Copying the linear regression-imputed data from the previous method.

h3_train, h3_val = h2_train.copy(), h2_val.copy()





nan_rows = h3_train.loc[total_train['demented'].isna()]

non_nan_rows = h3_train.loc[~total_train['demented'].isna()]



y, X = non_nan_rows['demented'], non_nan_rows.drop('demented', axis=1)

X = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(X)



param_grid = [ {   

                'penalty': ['l1', 'l2'], 

                'solver': ['liblinear', 'saga'], 

                'fit_intercept': [True, False],

                'C': np.logspace(0, 4, 20)

             } ]

best_tree = GridSearchCV(LogisticRegression(), param_grid, cv=5)

h3_train.loc[nan_rows.index, 'demented'] = best_tree.fit(X, y).predict(nan_rows.drop('demented', axis=1))
h4_train, h4_val = h2_train.copy(), h2_val.copy()





param_grid = [ {

                'max_depth': [2,3,4,5,6,7,8,9,10,11,12], 

                'criterion': ['entropy', 'gini'],

                'splitter': ['best', 'random'],

             } ]

best_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)

h4_train.loc[nan_rows.index, 'demented'] = best_tree.fit(X, y).predict(nan_rows.drop('demented', axis=1))
h5_train = total_train.dropna()

h5_val = h1_val.copy()
def Xy(train, val):

    y_train, y_val = train['demented'], val['demented']

    X_train, X_val = train.drop('demented', axis=1), val.drop('demented', axis=1)

    

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train.drop('demented', axis=1))

    return y_train, y_val, scaler.transform(X_train), scaler.transform(X_val)
def logreg(X, y, bag=True):

    

    # Create a hyperparameter space. There are two dictionaries because 

    # newton-cg, lbfgs and sag cannot use l1.

    param_grid = [

                      {

                           'penalty': ['l1', 'l2'], 

                           'solver': ['liblinear', 'saga'], 

                           'fit_intercept': [True, False],

                           'C': np.logspace(0, 4, 20)

                    },{

                           

                           'penalty': ['l2'], 

                           'solver': ['newton-cg', 'lbfgs', 'sag'], 

                           'fit_intercept': [True, False],

                           'C': np.logspace(0, 4, 20)

                      },

                 ]

    

    # Adding a prefix to the dictionary keys if bagging is used,

    # as BaggingClassifier needs it to distinguish its base estimator's

    # parameters from its own.

    if bag: param_grid = [

                            {'base_estimator__' + k: v for k, v in param_grid[0].items()},

                            {'base_estimator__' + k: v for k, v in param_grid[1].items()}

                         ]

    

    # The two alternative models.

    lr = LogisticRegression()

    bc = BaggingClassifier(base_estimator=lr)



    # Use stratified instead of shuffle split so that each split would contain a sufficient amount of each category.

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    

    # Conduct the search.

    best_model = GridSearchCV(estimator = (bc if bag else lr), 

                              param_grid = param_grid, 

                              cv = cv, 

                              scoring = 'recall',#'neg_log_loss', 

                              n_jobs = -1

                             ).fit(X,y)

    

    # Outputing the optimal point in the hyperparameter space.

    keys = param_grid[0].keys()

    grid_optimum = [best_model.best_estimator_.get_params()[p] for p in keys]

    for p, x in zip(keys, grid_optimum):

        print('Best {}: {}'.format(p,x))



    return best_model
imputations = [(h1_train, h1_val), (h2_train, h2_val), (h3_train, h3_val), (h4_train, h4_val), (h5_train, h5_val)]



def validate_logreg(bag):

    for i, (train, val) in enumerate(imputations):

        print("Handling method {}:".format(i+1))

        y_train, y_val, X_train, X_val = Xy(train, val)

        model = logreg(X_train, y_train, bag=bag)

        pred = model.predict(X_val)

        print(classification_report(y_val, pred))

        

validate_logreg(bag=False)
validate_logreg(bag=True)
def boosted_logreg(X, y):

    

    # Create a hyperparameter space. There are two dictionaries because 

    # newton-cg, lbfgs and sag cannot use l1.

    

    # The base estimator's subspace of the parameter space.

    logreg_param_grid = [

                              {

                                   'penalty': ['l1', 'l2'], 

                                   'solver': ['liblinear', 'saga'], 

                                   'fit_intercept': [True, False],

                                   'C': np.logspace(0, 4, 10)

                            },{



                                   'penalty': ['l2'], 

                                   'solver': ['newton-cg', 'lbfgs', 'sag'], 

                                   'fit_intercept': [True, False],

                                   'C': np.logspace(0, 4, 10)

                              },

                        ]

    

    # The booster's subspace of the parameter space.

    adaboost_param_grid = { 

                              'learning_rate': [0.1, 0.5, 1],

                              'n_estimators': np.arange(100, 200, 15)

                          }

    

    

    # Add a prefix to the logreg_param_grid keys,

    # as AdaBoostClassifier needs it to distinguish its base estimator's

    # parameters from its own.

    param_grid = [

                    {'base_estimator__' + k: v for k, v in logreg_param_grid[0].items()},#.update(adaboost_param_grid),

                    {'base_estimator__' + k: v for k, v in logreg_param_grid[1].items()}#.update(adaboost_param_grid)

                 ]

    

    # Combine the above two subspaces. Cannot use list comprehension with update() as it modifies in place.

    param_grid[0].update(adaboost_param_grid), param_grid[1].update(adaboost_param_grid)

    

    # The model.

    ab = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=LogisticRegression())

                            #learning_rate=1.0, 

                            #n_estimators=50,)

                            #random_state=None),



    # Use stratified instead of shuffle split so that each split would contain a sufficient amount of each category.

    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

    

    # Conduct the search.

    best_model = GridSearchCV(estimator = ab, 

                              param_grid = param_grid, 

                              cv = cv, 

                              scoring = 'recall',#'neg_log_loss', 

                              n_jobs = -1,

                              verbose = -1,

                             ).fit(X,y)

    

    # Outputing the optimal point in the hyperparameter space.

    keys = param_grid[0].keys()

    grid_optimum = [best_model.best_estimator_.get_params()[p] for p in keys]

    for p, x in zip(keys, grid_optimum):

        print('Best {}: {}'.format(p,x))



    return best_model
imputations = [(h1_train, h1_val), (h2_train, h2_val), (h3_train, h3_val), (h4_train, h4_val), (h5_train, h5_val)]



def validate_boosted():

    for i, (train, val) in enumerate(imputations):

        print("Handling method {}:".format(i+1))

        y_train, y_val, X_train, X_val = Xy(train, val)

        model = boosted_logreg(X_train, y_train)

        pred = model.predict(X_val)

        print(classification_report(y_val, pred))



validate_boosted()
from IPython.display import HTML

from IPython.display import display



cellNum=2

cellDisp='none'  # Other option is 'block'

cell="""

<script>

   var divTag = document.getElementsByClassName("input")[%s]

   var displaySetting = divTag.style.display;

   // Default display - set to 'none'.  To hide, set to 'block'.

   // divTag.style.display = 'block';

   divTag.style.display = '%s';

<script>

<!-- <button onclick="javascript:toggleInput(%s)" class="button">Toggle Code</button> -->

""" % (cellNum,'none',cellNum)

h=HTML(cell)

display(h)