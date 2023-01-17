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
# set seed for sklearn, without this even if the inputs are identical the sklearn models will have different outputs

np.random.seed(0)
df = pd.read_csv(r'/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.drop('Unnamed: 32', axis=1, inplace=True)

df.describe()
df.set_index('id', inplace=True)

print(df.columns)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['Diagnosis_labels'] = encoder.fit_transform(df['diagnosis'])

df.drop('diagnosis', axis=1, inplace = True)

df = df.sample(frac=1)



o_df = df.copy() # stands for original df, basic cleaning, no distribution correction
decode_dict = {0: 'Benign', 1:'Malignant'}
df['Diagnosis_labels'].value_counts()
import plotly.express as px

import plotly.figure_factory as ff

columns_chunks = [df.columns[:6], df.columns[6:12], df.columns[12:18], df.columns[18:24], df.columns[24:]]

for columns in columns_chunks:

    data = [df[column] for column in columns]

    group_labels = [column for column in columns]

    fig = ff.create_distplot(data, group_labels)

    #fig.show()
for i in df.columns:

    if i != 'Diagnosis_labels':

        fig = px.histogram(df, x=i, color='Diagnosis_labels')

        fig.show()
def shift_list(_list):

    shifted_list = [None]*len(_list)

    for i in range(len(_list)):

        shifted_list[i] = _list[i-1]

    return shifted_list
#test shift_list function

class my_Tests:

    def __init__(self):

        self.test_1_expected = [6, 1, 2, 3, 4, 5]

        self.test_1_input = [1, 2, 3, 4, 5, 6]

        

        self.test_2_expected = [False, False, True, False, False]

        self.test_2_input = [False, True, False, False, False]

    def test_1(self):

        result = shift_list(self.test_1_input)

        if result == self.test_1_expected:

            print('Passed') 

        else:

            print('Expected', self.test_1_expected, 'but got', result)

    def test_2(self):

        result = shift_list(self.test_2_input)

        if  result == self.test_2_expected:

            print('Passed') 

        else: 

            print('Expected', self.test_2_expected, 'but got', result)



T = my_Tests()

T.test_1()

T.test_2()
import plotly.graph_objects as go

def multi_hist(_df):

    columns = list(_df.columns)

    columns.pop(columns.index('Diagnosis_labels'))

    active_list = [True]*len(columns) 

    my_buttons = [dict(

        args=[{'visible':active_list}],

        label='All',

        method='update'

        )]

    active_list = [True]+([False]*(len(columns)-1))

    fig = go.Figure()

    for i in columns:

        fig.add_trace(go.Histogram(x=_df[i], name=i))

        my_buttons.append(

            dict(args=[{'visible':active_list}],

                 label=i,

                 method='update'

        ))

        active_list = shift_list(active_list)



    fig.update_layout(

        updatemenus=[

            dict(

                buttons=my_buttons,

                direction='down',

                pad={'r':.1, 't':10},

                showactive=True,

                x=0.04,

                xanchor='left',

                y=1.2,

                yanchor='top'

            )])



    fig.update_layout(

        annotations=[

            dict(text='Column', x=0, xref='paper', y=1.15, yref='paper', align='left', showarrow=False)

        ])

    fig.show()
from sklearn.preprocessing import PowerTransformer

from scipy.stats import boxcox

from math import log

#boxcox = PowerTransformer(method='box-cox')



skew_before = {}

columns = list(o_df.columns)

columns.pop(columns.index('Diagnosis_labels'))

boxcox_columns = []

for col in columns:

    skew_before[col] = df[col].skew()



t_df = pd.DataFrame()

skew_after = {}

for col in columns:

    if skew_before[col] >.5 and df[col].min() >0:

        t_df[col] = boxcox(df[col])[0]

        skew_after[col] = t_df[col].skew()

        boxcox_columns.append(col)

        print(col, 'Skew Before', skew_before[col], 'and Skew After', skew_after[col])

    else:

        t_df[col] = list(df[col])

        

#Fix columns that behaved poorly with box-cox and re-add daignosis labels

t_df['fractal_dimension_mean'] = list(o_df['fractal_dimension_mean'])

t_df['fractal_dimension_worst'] = list(o_df['fractal_dimension_worst'])

boxcox_columns.pop(boxcox_columns.index('fractal_dimension_mean'))

boxcox_columns.pop(boxcox_columns.index('fractal_dimension_worst'))

t_df['Diagnosis_labels'] = list(o_df['Diagnosis_labels'])

t_df.head()
def mean_center(series):

    mean = series.mean()

    mean_centered_series = series.apply(lambda x: x-mean)

    return mean_centered_series

    

def rescale(series): 

    lowest = series.min()

    highest = series.max()

    scaler = lowest if abs(lowest)> highest else highest

    rescaled_series = series.apply(lambda x: x/scaler)

    return rescaled_series



rs_df = pd.DataFrame()

for col in columns:

    rs_df[col] = rescale(mean_center(o_df[col]))

rs_df['Diagnosis_labels'] = list(o_df['Diagnosis_labels'])

rs_df.head()
o_df.var()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss_df= ss.fit_transform(o_df[columns])

ss_df = pd.DataFrame(ss_df, columns=columns)

ss_df['Diagnosis_labels'] = list(o_df['Diagnosis_labels']) 

ss_df.head()
multi_hist(o_df)

multi_hist(t_df)

multi_hist(rs_df)

multi_hist(ss_df)
temp_df = ss_df.copy()

temp_df.describe()
df.describe()
from sklearn.model_selection import train_test_split



y = temp_df['Diagnosis_labels'].copy()

X  = temp_df.copy()

X.drop('Diagnosis_labels', axis=1, inplace = True)

X, test_X, y, test_y = train_test_split(X, y, train_size=.8, test_size=.2, random_state=0)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size = .7, test_size = .3, random_state=0)
from random import randint, seed

from sklearn.metrics import f1_score

seed(0)

all_malignant = pd.Series([1] * X.shape[0])

all_benign = pd.Series([0] * X.shape[0])

random_guess = [randint(0, 1) for i in range(X.shape[0])]

naive_f1_score = f1_score(all_benign, y, average='binary')

print('Naive f1 score for always guessing "Benign" is', naive_f1_score)

naive_f1_score = f1_score(all_malignant, y, average='binary')

print('Naive f1 score for always guessing "Malignant" is', naive_f1_score)

naive_f1_score = f1_score(random_guess, y, average='binary')

print('Naive f1 score for randomly guessing', naive_f1_score)



baseline_score = f1_score(all_malignant, y, average='binary')
swapped_y = y.apply(lambda x: 1 if x==0 else 0)

print(f1_score([1]*X.shape[0], swapped_y))
malignant_count = valid_y[valid_y == 1].shape[0]

benign_count = valid_y[valid_y == 0].shape[0]

print('Ratio of malignant to total:', malignant_count/(malignant_count+benign_count))

print('Ratio of benign to total:', benign_count/(malignant_count+benign_count))
#Cross validation

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

def get_cross_val_score(_model, folds=5):

    pipeline = Pipeline(steps=[('model', _model)])

    scores = cross_val_score(pipeline, X, y, cv=folds, scoring = 'f1')

    return (scores.mean())

get_cross_val_score(XGBClassifier(n_estimators = 1000, learning_rate = .05, n_jobs=4))
from sklearn.model_selection import GridSearchCV

#Use Grid when there is a low combination of parameters, and randomized when it starts to be too costly
# Light GBM Classifier

from lightgbm import LGBMClassifier

from statistics import mean



lgb = LGBMClassifier()

lgb_params = {'booster':['gbdt', 'dart']}



lgb_grid_search = GridSearchCV(lgb, lgb_params, scoring='f1', cv=5, refit=True)

lgb_grid_search.fit(train_X, train_y)

print('Best Score:', lgb_grid_search.best_score_)

print('Best Parameters:', lgb_grid_search.best_params_)

print('Train Score:', lgb_grid_search.score(train_X, train_y))

print('Valid Score:', lgb_grid_search.score(valid_X, valid_y))
lgb = LGBMClassifier()

lgb_params = {'n_estimators':[50, 100, 300, 500, 1000],

          'num_leaves':[i for i in range(50, 150, 10)],

          'min_data_in_leaf':[100, 500, 1000, 1500, 2000],

          'max_depth':[i for i in range(3, 8)]}



lgb_grid_search = GridSearchCV(lgb, lgb_params, scoring='f1', cv=5, refit=True)

lgb_grid_search.fit(train_X, train_y)

print('Best Score:', lgb_grid_search.best_score_)

print('Best Parameters:', lgb_grid_search.best_params_)

print('Train Score:', lgb_grid_search.score(train_X, train_y))

print('Valid Score:', lgb_grid_search.score(valid_X, valid_y))
lgb = LGBMClassifier()

lgb_params = {'n_estimators':[i for i in range(220, 290, 5)],

          'num_leaves':[i for i in range(50, 100, 5)],

          'min_data_in_leaf':[i for i in range(50, 70, 5)],

          'max_depth':[3, 4]}



lgb_grid_search = GridSearchCV(lgb, lgb_params, scoring='f1', cv=5, refit=True)

lgb_grid_search.fit(train_X, train_y)

print('Best Score:', lgb_grid_search.best_score_)

print('Best Parameters:', lgb_grid_search.best_params_)

print('Train Score:', lgb_grid_search.score(train_X, train_y))

print('Valid Score:', lgb_grid_search.score(valid_X, valid_y))
from sklearn.metrics import confusion_matrix

import seaborn as sns



lgb_params = {'n_estimators':275,

          'num_leaves':5,

          'min_data_in_leaf':50,

          'max_depth':3}



lgb_params = {'n_estimators':250,

          'num_leaves':50,

          'min_data_in_leaf':60,

          'max_depth':3}



lgb = LGBMClassifier(**lgb_params)

lgb.fit(X, y)

lgb_predictions = lgb.predict(test_X)

print('F1_score', f1_score(test_y, lgb_predictions))

lgb_cf_matrix = confusion_matrix(test_y, lgb_predictions)

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ['{0:0.0f}'.format(value) for value in lgb_cf_matrix.flatten()]

group_percentages = ['{0:.2%}'.format(value) for value in lgb_cf_matrix.flatten()/np.sum(lgb_cf_matrix)]

labels = [f'{_1}\n{_2}\n{_3}' for _1, _2, _3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(lgb_cf_matrix, annot=labels, fmt='')

lgb_features_importances = pd.DataFrame()

lgb_features_importances['Columns'] = train_X.columns

lgb_features_importances['Feature_Importances'] = lgb.booster_.feature_importance(importance_type='gain')

lgb_features_importances.sort_values(['Feature_Importances'], inplace=True)

fig_lgb_feature = px.bar(lgb_features_importances, x = 'Columns', y='Feature_Importances') 

fig_lgb_feature.show()
#Optimize XGBoost

xgb_params = {'max_depth':[i for i in range(3,11, 2)], 

              'n_estimators':[100, 300, 500, 1000, 5000, 10000], 

              'learning_rate':[.1], 

              'n_jobs':[4],

              }

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
#Optimize XGBoost

xgb_params = {'max_depth':[5], 

              'n_estimators':[i for i in range(150, 250, 10)], 

              'learning_rate':[.1], 

              'n_jobs':[4], 

              }

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
xgb_params = {'max_depth':[5], 

              'n_estimators':[190], 

              'learning_rate':[.1], 

              'n_jobs':[4], 

              'min_child_weight':[i for i in range(1, 6)]}

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
xgb_params = {'max_depth':[5], 

              'n_estimators':[190], 

              'learning_rate':[.1], 

              'n_jobs':[4], 

              'min_child_weight':[1],

              'gamma':[i/10.0 for i in range(5)]}

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
xgb_params = {'max_depth':[5], 

              'n_estimators':[190], 

              'learning_rate':[.1], 

              'n_jobs':[4], 

              'min_child_weight':[1],

              'gamma':[0],

              'subsample':[i/10.0 for i in range(6, 10)],

              'colsample_bytree':[i/10.0 for i in range(6, 10)]}

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
xgb_params = {'max_depth':[5], 

              'n_estimators':[190], 

              'learning_rate':[.1], 

              'n_jobs':[4], 

              'min_child_weight':[1],

              'gamma':[0],

              'subsample':[.7],

              'colsample_bytree':[.8],

              'reg_alpha':[0, 1e-5, 1e-2, .1]}

xgb = XGBClassifier()

xgb_rand_search  = GridSearchCV(xgb, xgb_params, scoring='f1', cv=5, refit=True)

xgb_rand_search.fit(train_X, train_y)

print('Best Score:', xgb_rand_search.best_score_)

print('Best Parameters:', xgb_rand_search.best_params_)

print('Train Score:', xgb_rand_search.score(train_X, train_y))

print('Valid Score:', xgb_rand_search.score(valid_X, valid_y))
xgb_params = {'max_depth':5, 

              'n_estimators':190, 

              'learning_rate':.1, 

              'n_jobs':4, 

              'min_child_weight':1,

              'gamma':0,

              'subsample':.7,

              'colsample_bytree':.8,

              'reg_alpha':0}

my_xgb = XGBClassifier(**xgb_params)

my_xgb.fit(X, y ,verbose=False)

xgb_predictions = my_xgb.predict(test_X)

print('F1 score', f1_score(test_y, xgb_predictions))

xgb_cf_matrix = confusion_matrix(test_y, xgb_predictions)

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ['{0:0.0f}'.format(value) for value in xgb_cf_matrix.flatten()]

group_percentages = ['{0:.2%}'.format(value) for value in xgb_cf_matrix.flatten()/np.sum(xgb_cf_matrix)]

labels = [f'{_1}\n{_2}\n{_3}' for _1, _2, _3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(xgb_cf_matrix, annot=labels, fmt='')
xgb_features_importances = pd.DataFrame(my_xgb.get_booster().get_score(importance_type='gain').items(), columns = ['Columns', 'Feature_Importances'])

xgb_features_importances.sort_values(['Feature_Importances'], inplace=True)

fig_xgb_feature = px.bar(xgb_features_importances, x = 'Columns', y='Feature_Importances') 

fig_xgb_feature.show()
# SVC Model 



#TODO change all the ms to cs

from sklearn.svm import SVC

def my_svc(_train_X, _train_y, _valid_X, _valid_y, _kernel, _degree=3):

    svc_model = SVC(kernel=_kernel, degree=_degree)

    svc_model.fit(_train_X, _train_y)

    svc_predictions  = svc_model.predict(_valid_X)

    svc_score = f1_score(svc_predictions, _valid_y)

    print('F1 score for SVC model is', svc_score)
#Optimize SVC

kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:

    print(kernel)

    my_svc(train_X, train_y, valid_X, valid_y, kernel)



#Linear looks the best, but it's worth exploring polynomial to make sure that it isn't equally viable



for i in range(2, 5):

    print('For "poly" kernelm of degree',i)

    my_svc(train_X, train_y, valid_X, valid_y, 'poly', _degree=i)



#Linear still seems like the best option.



svc_params = {'kernel':['linear', 'rbf'],'C':[0.01,0.1,1,0.001],'gamma':[0.1,0.01,0.2,0.4]}

svc = SVC(probability = True)

svc_grid_search = GridSearchCV(svc, svc_params, scoring='f1',cv = 5,refit = True)



svc_grid_search.fit(train_X, train_y)

print()

print("Best Score ==> ", svc_grid_search.best_score_)

print("Tuned Paramerers ==> ",svc_grid_search.best_params_)

print("Accuracy on Train set ==> ", svc_grid_search.score(train_X,train_y))

print("Accuracy on Test set ==> ", svc_grid_search.score(valid_X, valid_y))



svc_params = {'kernel':'linear',

              'C':0.1,

              'gamma':0.1}

svc = SVC(**svc_params)

svc.fit(X, y)

svc_predictions = svc.predict(test_X)

print('F1 score', f1_score(test_y, svc_predictions))

svc_cf_matrix = confusion_matrix(test_y, svc_predictions)

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ['{0:0.0f}'.format(value) for value in svc_cf_matrix.flatten()]

group_percentages = ['{0:.2%}'.format(value) for value in svc_cf_matrix.flatten()/np.sum(svc_cf_matrix)]

labels = [f'{_1}\n{_2}\n{_3}' for _1, _2, _3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(svc_cf_matrix, annot=labels, fmt='')
svc_features_importances = pd.DataFrame()

svc_features_importances['Feature_Importances'] = svc.coef_[0]

svc_features_importances['Columns'] = train_X.columns

svc_features_importances.sort_values(['Feature_Importances'], inplace=True)

fig_svc_feature = px.bar(svc_features_importances, x = 'Columns', y='Feature_Importances') 

fig_svc_feature.show()