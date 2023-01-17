import numpy as np                  # Mathetimatical Operations

import pandas as pd                 # Data manipulation



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt     

%matplotlib inline



# Sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, classification_report, mean_squared_error, confusion_matrix, f1_score, precision_recall_curve, r2_score 

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor



# Scipy

from scipy.stats import stats

from scipy.stats import ttest_ind, ttest_ind_from_stats



# XGBoost

from xgboost import XGBClassifier

from xgboost import XGBRegressor

import xgboost as xgb



# LightGBM

import lightgbm as lgb



# Datetime

import datetime 

import time

from datetime import datetime



# Folium

import folium 

from folium import plugins

from folium.plugins import HeatMap



# Image

from IPython.display import Image



# Bayesian Optimizer

from skopt import BayesSearchCV



# Itertools

import itertools



# Remove warnings

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/Apply_Rate_2019.csv')
df.head()
# Check the total number of observations in the dataset



print('Total number of observations in the dataset are:',df.shape[0])
df.info()
df.drop(['apply'],axis=1).describe()
# Lets check the distribution for classes who applied and did not apply



count_classes = pd.value_counts(df['apply'], sort = True)

count_classes.plot(kind = 'bar')



plt.title("Apply Rate")

plt.xticks(range(2))

plt.xlabel("Class")

plt.ylabel("Frequency");



print('Number of customers who didnt apply:',df['apply'].value_counts()[0])

print('Number of customers who applied:',df['apply'].value_counts()[1])

print('Percentage of apply to non apply',df['apply'].value_counts()[0]/df['apply'].value_counts()[1],'%')
# Lets check the correlation between the features



sns.heatmap(df.corr())
l = ['title_proximity_tfidf', 'description_proximity_tfidf',

       'main_query_tfidf', 'query_jl_score', 'query_title_score',

       'city_match', 'job_age_days']

number_of_columns=7

number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(number_of_columns,5*number_of_rows))

for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.set_style('whitegrid')

    sns.boxplot(df[l[i]],color='green',orient='v')

    plt.tight_layout()
# Check the distribution



# Now to check the linearity of the variables it is a good practice to plot distribution graph and look for skewness 

# of features. Kernel density estimate (kde) is a quite useful tool for plotting the shape of a distribution.



for feature in df.columns[:-3]:

    ax = plt.subplot()

    sns.distplot(df[df['apply'] == 1][feature], bins=50, label='Anormal')

    sns.distplot(df[df['apply'] == 0][feature], bins=50, label='Normal')

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(feature))

    plt.legend(loc='best')

    plt.show()
print(df.shape)

df = df.drop_duplicates(keep = 'first')

df.shape
df.isnull().sum()
# Lets check the value counts for the three columns

df['title_proximity_tfidf'].value_counts().head()
df['description_proximity_tfidf'].value_counts().head()
df['city_match'].value_counts().head()
df['title_proximity_tfidf'].fillna(0,inplace=True)

df['description_proximity_tfidf'].fillna(0,inplace=True)

df.dropna(subset=['city_match'],inplace=True)
# From the correlation graph, we observed that title_proximity_tfidf and main_query_tfidf are quite correlated, 

# lets merge them and get a single feature by multiplying both of them



df['main_title_tfidf'] = df['title_proximity_tfidf']*df['main_query_tfidf']
df = df.drop(['title_proximity_tfidf','main_query_tfidf'], axis=1)
# Splitting the dataset by date

train = df.loc[df['search_date_pacific']<'2018-01-27']

test = df.loc[df['search_date_pacific'] == '2018-01-27']
# Drop the unnecessary columns

train.drop(['search_date_pacific','class_id'],axis=1,inplace = True)

test.drop(['search_date_pacific','class_id'],axis=1,inplace = True)
# Drop irrelevant features

X = df.drop(['search_date_pacific','class_id','apply'],axis=1)

y = df['apply']
# Reset the index

X = X.reset_index(drop='index')

y = y.reset_index(drop='index')
X_train = train.drop(['apply'],axis=1)

y_train = train['apply']

X_test = test.drop(['apply'],axis=1)

y_test = test['apply']
# Define a function to plot confusion matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`

    """

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
# Define a function which will be used to get the important parameters like AUC, Classification report



def report(test_set, predictions,labels,title):

    print('F1 score is:', f1_score(test_set,predictions))

    print("AUC-ROC is: %3.2f" % (roc_auc_score(test_set, predictions)))

    plot_confusion_matrix(confusion_matrix(test_set, predictions),labels,title)

    

    #plot the curve

    fpr, tpr, threshold = roc_curve(test_set,predictions)

    auc = roc_auc_score(test_set,predictions)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b',label='Model - AUC = %0.3f'% auc)

    ax.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--', label='Chance')

    ax.legend()

    ax.set_xlim([-0.1,1.0])

    ax.set_ylim([-0.1,1.01])

    ax.set_ylabel('True Positive Rate')

    ax.set_xlabel('False Positive Rate')

    plt.show()
# Define a function to print the status during bayesian hyperparameter search



def status_print(optim_result):

    """Status callback durring bayesian hyperparameter search"""

    

    # Get all the models tested so far in DataFrame format

    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    

    

    # Get current parameters and the best parameters    

    best_params = pd.Series(bayes_cv_tuner.best_params_)

    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(

        len(all_models),

        np.round(bayes_cv_tuner.best_score_, 4),

        bayes_cv_tuner.best_params_

    ))



# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL

ITERATIONS = 10

TRAINING_SIZE = 100000 

TEST_SIZE = 25000





# Classifier

bayes_cv_tuner = BayesSearchCV(

    estimator = xgb.XGBClassifier(

        n_jobs = 1,

        objective = 'binary:logistic',

        eval_metric = 'auc',

        silent=1,

        tree_method='approx'

    ),

    search_spaces = {

        'learning_rate': (0.01, 1.0, 'log-uniform'),

        'min_child_weight': (0, 10),

        'max_depth': (0, 50),

        'max_delta_step': (0, 20),

        'subsample': (0.01, 1.0, 'uniform'),

        'colsample_bytree': (0.01, 1.0, 'uniform'),

        'colsample_bylevel': (0.01, 1.0, 'uniform'),

        'reg_lambda': (1e-9, 1000, 'log-uniform'),

        'reg_alpha': (1e-9, 1.0, 'log-uniform'),

        'gamma': (1e-9, 0.5, 'log-uniform'),

        'min_child_weight': (0, 5),

        'n_estimators': (50, 100),

        'scale_pos_weight': (1e-6, 500, 'log-uniform')

    },    

    scoring = 'roc_auc',

    cv = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=42

    ),

    n_jobs = 3,

    n_iter = ITERATIONS,   

    verbose = 0,

    refit = True,

    random_state = 42

)

result = bayes_cv_tuner.fit(X, y, callback=status_print)
xgb = XGBClassifier(colsample_bylevel= 0.8390144719977516, colsample_bytree= 0.8844821246070537, 

                    gamma= 4.358684608480795e-07, learning_rate= 0.7988179462781242, max_delta_step= 17, 

                    max_depth= 3, min_child_weight= 1, n_estimators= 68, reg_alpha= 0.0005266983003701547, 

                    reg_lambda= 276.5424475574225, scale_pos_weight= 0.3016410771843142, subsample= 0.9923710598637134)

xgb.fit(X_train, y_train)

preds_xgb = xgb.predict_proba(X_test)[:, 1]

labels = ['No Apply', 'Apply']

#report(y_test, preds_xgb,labels, 'Confusion Matrix')

auc = roc_auc_score(y_test, preds_xgb)



print('The baseline score on the test set is {:.4f}.'.format(auc))
# ITERATIONS = 10 # 1000

# TRAINING_SIZE = 100000 # 20000000

# TEST_SIZE = 25000

# # Classifier

# bayes_cv_tuner = BayesSearchCV(

#     estimator = RandomForestClassifier(

#         n_jobs = -1

#     ),

#     search_spaces = {

#     'min_samples_split': [3, 5, 8, 10, 20], 

#     'n_estimators' : [100, 500],

#     'max_depth': [3, 5, 8, 10, 15],

#     'max_features': [3, 5, 6]

# },    

#     scoring = 'roc_auc',

#     cv = StratifiedKFold(

#         n_splits=3,

#         shuffle=True,

#         random_state=42

#     ),

#     n_jobs = 3,

#     n_iter = ITERATIONS,   

#     verbose = 0,

#     refit = True,

#     random_state = 42

# )
# result = bayes_cv_tuner.fit(X, y, callback=status_print)
# rf = RandomForestClassifier(

#     n_estimators=421, 

#     max_depth=15,

#     max_features=3,

#     min_samples_split=8, 

#     class_weight="balanced",

#     bootstrap=True,

#     criterion='entropy',

#     random_state=100

#     )



# rf.fit(X_train, y_train)

# preds_rf = rf.predict_proba(X_test)[:,1]

# #labels = ['No Apply', 'Apply']

# #report(y_test, preds_rf,labels, 'Confusion Matrix')

# auc = roc_auc_score(y_test, preds_rf)



# print('The baseline score on the test set is {:.4f}.'.format(auc))
# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL

ITERATIONS = 10

TRAINING_SIZE = 100000 

TEST_SIZE = 25000





# Classifier

bayes_cv_tuner = BayesSearchCV(

    estimator = lgb.LGBMClassifier(

        n_jobs = 1,

        objective = 'binary',

        eval_metric = 'auc',

        silent=1,

        tree_method='approx'

    ),

    search_spaces = {

        'learning_rate': (0.01, 1.0, 'log-uniform'),

        'min_child_weight': (0, 10),

        'max_depth': (0, 50),

        'subsample': (0.01, 1.0, 'uniform'),

        'colsample_bytree': (0.01, 1.0, 'uniform'),

        'reg_lambda': (1e-9, 1000, 'log-uniform'),

        'reg_alpha': (1e-9, 1.0, 'log-uniform'),

        'min_child_weight': (0, 5),

        'n_estimators': (50, 100)

    },    

    scoring = 'roc_auc',

    cv = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=42

    ),

    n_jobs = 3,

    n_iter = ITERATIONS,   

    verbose = 0,

    refit = True,

    random_state = 42

)

result = bayes_cv_tuner.fit(X, y, callback=status_print)
model = lgb.LGBMClassifier(colsample_bytree=0.8015579071911014, learning_rate=0.07517239253342656, 

                           max_depth=26, min_child_weight=4, n_estimators=95, reg_alpha=0.002839751649223172, 

                           reg_lambda=0.0001230656555713626, subsample=0.653781260730285)



model.fit(X_train, y_train)





preds_lgb = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, preds_lgb)



print('The baseline score on the test set is {:.4f}.'.format(auc))