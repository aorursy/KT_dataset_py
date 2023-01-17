# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np                # For arithmatic, linear algebra operations
import pandas as pd               # For data processing, CSV file I/O (e.g. pd.read_csv)
#import sklearn                    # For modeling
import matplotlib.pyplot as plt   # For visualization
import seaborn as sns             # For visualization
% matplotlib notebook
% matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Read and explore data
path = '../input/'
data_test = pd.read_csv(path + 'ml_case_test_data.csv')
print('data_test: {} rows, {} columns'.format(data_test.shape[0],data_test.shape[1] ))
data_test_hist = pd.read_csv(path + 'ml_case_test_hist_data.csv')
data_test_output = pd.read_csv(path + 'ml_case_test_output_template.csv')
assert( len(data_test.id.unique()) == len(data_test_hist.id.unique()))
assert( len(data_test.id.unique()) == len(data_test_output.id.unique()))

data_train = pd.read_csv(path + 'ml_case_training_data.csv')
print('data_train: {} rows, {} columns'.format(data_train.shape[0],data_train.shape[1] ))
data_train_hist  = pd.read_csv(path + 'ml_case_training_hist_data.csv')
data_train_output = pd.read_csv(path + 'ml_case_training_output.csv')
assert( len(data_train.id.unique()) == len(data_train_hist.id.unique()))
assert( len(data_train.id.unique()) == len(data_train_output.id.unique()))

data_train = data_train.merge(data_train_output, how='left', on = 'id')
print('data_train: {} rows, {} columns "after merging the output variable for target variable churn"'.format(data_train.shape[0],data_train.shape[1]))
## Creat Data Manager / Data Frame to store the required dataset
data_manager = pd.DataFrame( index = data_train.columns, columns = ['dtype', 'miss_rate', 'group', 'note', 'function'])
data_manager['dtype'] = data_train.dtypes
data_manager['dtype'].replace({'object': 'str'}, inplace = True)
data_manager['miss_rate'] = data_train.isnull().mean()
data_manager['group'] = 'raw'
data_manager
## Remove High Correlated Variable
corr_matrix = data_train.corr().abs()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print('The variables whose the correlations  > 0.95: {}'.format(to_drop) )
data_train[to_drop].corr().style.background_gradient().set_precision(2)

#'forecast_base_bill_year','forecast_bill_ele' 'forecast_cons', 'forecast_cons_year' are high correlated with 'imp_cons'
# Since, 'imp_cons' has less missing values, so that we keep only 'imp_cons'
list_remove = ['forecast_base_bill_year','forecast_base_bill_ele', 'forecast_cons', 'forecast_cons_year']

# 'forecast_bill_12m' and 'forecast_cons_12m' are strongly correlated
# Since 'forecast_cons_12m' has less missing values, so that we discard 'forecast_bill_12m'
list_remove.append('forecast_bill_12m')
data_manager.loc[list_remove, 'group'] = 'removed'
## Missing rate

# The graphs below show that:

# 1 Variable 'campagain_disc_ele' is totally missed. So that we remove this variable

# 2 The variables:
# a)'date_first_active', 'forecast_base_bill_ele', 'forecast_base_bill_year', 'forecast_base_bill_12m', 'forecast_cons', are perfectly correlated.
# b) 'forecast_discount_energy', 'forecast_price_energy_p1', 'forecast_price_energy_p2', 'forecast_price_pow_p1' are perfectly correlated.
## Missing rate
import missingno as msno
msno.matrix(data_train)

# Delete 'campaign_disc_ele'
data_manager.loc['campaign_disc_ele', 'group'] = 'removed'

# Impute the missing values.
map_imputation =  {}

# We impute below numeric variables by respective mean values
map_imputation['forecast_discount_energy'] = data_train.forecast_discount_energy.value_counts().index[0]
map_imputation['forecast_price_energy_p1'] = data_train.forecast_price_energy_p1.mean()
map_imputation['forecast_price_energy_p2'] = data_train.forecast_price_energy_p2.mean()
map_imputation['forecast_price_pow_p1'] = data_train.forecast_price_pow_p1.mean()
map_imputation['margin_gross_pow_ele'] = data_train.margin_gross_pow_ele.mean()
map_imputation['margin_net_pow_ele'] = data_train.margin_net_pow_ele.mean()
map_imputation['net_margin'] = data_train.net_margin.mean()
map_imputation['pow_max'] = data_train.pow_max.mean()

# We impute datetime  by most frenquency value
map_imputation['date_activ'] = data_train.date_activ.value_counts().index[0]
map_imputation['date_renewal'] = data_train.date_renewal.value_counts().index[0]
map_imputation['date_modif_prod'] = data_train.date_modif_prod.value_counts().index[0]
map_imputation['date_end'] = data_train.date_end.value_counts().index[0]
 
data_train.fillna( map_imputation, inplace = True)

# We impute 'NaN' for categorical vars
map_imputation['channel_sales']= 'NaN'
map_imputation['has_gas']= 'NaN'
map_imputation['origin_up']= 'NaN'
## Treatment for categorical variable

# Treatement the var 'activity_new'
print('Number of different category of company activity: {}'.format( len(data_train.activity_new.unique())) )
# There are 420 cats , so that we'll replace it by the new variable which is an average rate of churning 

# Calculate average rate of churning by cat
map_activity_churn = dict(data_train.groupby('activity_new')['churn'].mean())
map_activity_churn['NaN'] = data_train.loc[data_train['activity_new'].isnull(), 'churn'].mean()

# Creat the new variable 
#data_train['activity_churn'] = data_train.activity_new.fillna('NaN').replace(map_activity_churn)
def Activity_churn(df):
    df['activity_churn']= df.activity_new.fillna('NaN').replace(map_activity_churn)

## For other vars, there are not so much categories so that we'll transform them into one hot vector 

from sklearn.preprocessing import LabelBinarizer

# One-hot encode data - for below 3 variables
Channel_sales = LabelBinarizer()
Channel_sales.fit( data_train.channel_sales.fillna('NaN'))

Has_gas = LabelBinarizer()
Has_gas.fit( data_train.has_gas.fillna('NaN'))

Origin_up = LabelBinarizer()
Origin_up.fit(data_train.origin_up.fillna('NaN'))


# Update data manager
data_manager.loc['activity_new', 'group'] = 'not used'
data_manager.loc['activity_churn',: ] = ['float64', 0, 'derived', 'Average rate of churning by activity.', Activity_churn]
data_manager.loc['channel_sales', ['group', 'function'] ]= ['one hot', Channel_sales]
data_manager.loc['has_gas', ['group', 'function'] ]= ['one hot', Has_gas]
data_manager.loc['origin_up', ['group', 'function'] ]= ['one hot', Origin_up]

# Now, we look at the data manager to see how they stack up
#data_manager
#data_manager.loc['channel_sales'].value_counts().plot(kind='bar')
## Treatment for DATETIME variables

data_train[['date_activ', 'date_first_activ', 'date_modif_prod', 'date_renewal' , 'date_end']].head(10)
# We remove date_first_activ
data_manager.loc['date_first_activ', 'group']= 'removed'


def Year_activ(df):
    df.date_activ = df.date_activ.astype('datetime64[ns]')
    df['year_activ'] = df['date_activ'].dt.year
    
def Activ_to_end(df):
    df.date_activ = df.date_activ.astype('datetime64[ns]')
    df.date_end = df.date_end.astype('datetime64[ns]')
    df['activ_to_end'] = (df.date_end - df.date_activ)/ np.timedelta64(1,'Y')
    
    
def Activ_to_modif(df):
    df.date_activ = df.date_activ.astype('datetime64[ns]')
    df.date_modif_prod = df.date_modif_prod.astype('datetime64[ns]')
    df['activ_to_modif'] = (df.date_modif_prod - df.date_activ)/ np.timedelta64(1,'Y')

def Renewal_to_end(df):
    df.date_renewal = df.date_renewal.astype('datetime64[ns]')
    df.date_end = df.date_end.astype('datetime64[ns]')
    df['renewal_to_end'] = (df.date_end - df.date_renewal)/ np.timedelta64(1,'Y')
   
# Update data_manager
data_manager.loc[['date_activ', 'date_modif_prod', 'date_renewal' , 'date_end'], 'group'] = 'not used'
data_manager.loc['year_active', :] = ['int64', 0,  'derived', 'Year when the contract was activated', Year_activ]
data_manager.loc['activ_to_end', :] = ['float64', 0,  'derived', 'Time from active date to end date', Activ_to_end]
data_manager.loc['activ_to_modif', :] = ['float64', 0,  'derived', 'Time from active date to modification date', Activ_to_modif]
data_manager.loc['renewal_to_end', :] = ['float64', 0,  'derived', 'Time from renewal date to end date', Renewal_to_end]
## Treament for Price Variables

# Creat new variables to describe variation of forecast prices compared with prices in 2015.

def Forecast_price_eneregy_p1_delta(df, df_hist):
    price_mean_2015 = df_hist.groupby('id')['price_p1_var'].mean()
    price_mean_2015 = price_mean_2015.loc[ df.id, ].values
    price_forecast = df['forecast_price_energy_p1']
    df['forecast_price_energy_p1_delta'] = [(p1-p2)/p2*100 if p2!= 0 else 0 for (p1, p2) in zip(price_forecast, price_mean_2015) ]
    df['forecast_price_energy_p1_delta'].fillna(0, inplace = True)

    
def Forecast_price_eneregy_p2_delta(df, df_hist):
    price_mean_2015 = df_hist.groupby('id')['price_p2_var'].mean()
    price_mean_2015 = price_mean_2015.loc[ df.id, ].values
    price_forecast = df['forecast_price_energy_p2']
    df['forecast_price_energy_p2_delta'] = [(p1-p2)/p2*100 if p2!= 0 else 0 for (p1, p2) in zip(price_forecast, price_mean_2015) ]
    df['forecast_price_energy_p2_delta'].fillna(0, inplace = True)

def Forecast_price_pow_p1_delta(df, df_hist):
    price_mean_2015 = df_hist.groupby('id')['price_p1_fix'].mean()
    price_mean_2015 = price_mean_2015.loc[ df.id, ].values
    price_forecast = df['forecast_price_pow_p1']
    df['forecast_price_pow_p1_delta'] = [(p1-p2)/p2*100 if p2!= 0 else 0 for (p1, p2) in zip(price_forecast, price_mean_2015) ]
    df['forecast_price_pow_p1_delta'].fillna(0, inplace = True)

# Creat new variables to describe dynamic of 2015 prices.

def mean_derivate( prices):
    return(np.gradient(prices).mean())

def Price2015_energy_p1_derivate(df, df_hist):
    price_p1_var_derivate = df_hist.groupby('id')['price_p1_var'].apply( mean_derivate)
    price_p1_var_derivate = price_p1_var_derivate.loc[ df.id, ].values
    df['price2015_energy_p1_derivate'] = price_p1_var_derivate
    df['price2015_energy_p1_derivate'].fillna(0, inplace = True)


def Price2015_energy_p2_derivate(df, df_hist):
    price_p2_var_derivate = df_hist.groupby('id')['price_p2_var'].apply( mean_derivate)
    price_p2_var_derivate = price_p2_var_derivate.loc[ df.id, ].values
    df['price2015_energy_p2_derivate'] = price_p2_var_derivate
    df['price2015_energy_p2_derivate'].fillna(0, inplace = True)

def Price2015_pow_p1_derivate(df, df_hist):
    price_p1_fix_derivate = df_hist.groupby('id')['price_p1_fix'].apply( mean_derivate)
    price_p1_fix_derivate = price_p1_fix_derivate.loc[ df.id, ].values
    df['price2015_pow_p1_derivate'] = price_p1_fix_derivate
    df['price2015_pow_p1_derivate'].fillna(0, inplace = True)

## Update data_manager

data_manager.loc['forecast_price_energy_p1_delta', :] = ['float64', 0 ,  'derived', 'Variation ( percentage of price in 2015) of forecast energy price p1', Forecast_price_eneregy_p1_delta]
data_manager.loc['forecast_price_energy_p2_delta', :] = ['float64', 0,  'derived', 'Variation ( percentage of price in 2015) of forecast energy price p2', Forecast_price_eneregy_p2_delta]
data_manager.loc['forecast_price_pow_p1_delta', :] = ['float64',0,  'derived', 'Variation ( percentage of price in 2015) of forecast power price p1', Forecast_price_pow_p1_delta]

data_manager.loc['price2015_energy_p1_derivate', :] = ['float64', 0,  'derived', 'Mean derivate of 2015 energy price p1', Price2015_energy_p1_derivate]
data_manager.loc['price2015_energy_p2_derivate', :] = ['float64', 0,  'derived', 'Mean derivate of 2015 energy price p2', Price2015_energy_p2_derivate]
data_manager.loc['price2015_pow_p1_derivate', :] = ['float64', 0,  'derived', 'Mean derivate of 2015 power price p1', Price2015_pow_p1_derivate]
## Definition of Preprocessing Data Function

def Preprocessing_Data(df, df_hist, data_manager, map_imputation): 
    df_new = df.copy()
    
    # Drop the removed variables
    df_new.drop(list(data_manager.index[data_manager.group == 'removed']), axis = 1, inplace = True)
    
    # Imputation missing value
    df_new.fillna(map_imputation, inplace = True)
    
    # Create derived variables
    for var in list(data_manager.index[data_manager.group == 'derived']):
        if 'price' in var: 
            data_manager.loc[var, 'function'](df_new, df_hist)
        else:
            data_manager.loc[var, 'function'](df_new)        
    
    # Remove the 'not used' vars
    df_new.drop(list(data_manager.index[data_manager.group == 'not used']), axis = 1, inplace = True)
    
    # Creat one hot vector
    for var in list(data_manager.index[data_manager.group == 'one hot']): 
       # print(var)
        onehot = data_manager.loc[var, 'function']
        name_var = [var +'_'+ classe for classe in onehot.classes_ ]
        if(len(onehot.classes_) ==2):
            onehot_matrix = pd.DataFrame(onehot.transform( df_new[var]), columns = [name_var[1]])
        else:
            onehot_matrix = pd.DataFrame(onehot.transform( df_new[var]), columns = name_var)
        df_new = pd.concat([df_new, onehot_matrix], axis = 1)
        # Drop the original var 
        df_new.drop(var, axis = 1, inplace = True)
    
    return df_new
## Get the updated data frame
data_manager
## Get the data preprocessing for train dataset

#data_train.drop(list(data_manager.index[data_manager.group == 'removed']), axis = 1, inplace = True)
#del data_manager['campagain_disc_ele']
df_train = Preprocessing_Data( data_train, data_train_hist, data_manager, map_imputation)
df_train.index = df_train.id
df_train.drop('id',axis = 1, inplace = True)
label_train = df_train.churn
df_train.drop('churn', axis = 1, inplace= True)

df_test = Preprocessing_Data( data_test, data_test_hist, data_manager, map_imputation)
df_test.index = df_test.id
df_test.drop( 'id', axis = 1, inplace = True)
df_test.loc[(list(map(lambda x: type(x) == str, df_test.activity_churn))), 'activity_churn']  = 0

df_train.head(10)
df_test.head(10)
## Model Development
# Definition of models
# Load libraries
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA 
  
# Create standardizer
standardizer = StandardScaler()
# Reduction dimension
pca = PCA(copy=True,
          iterated_power='auto',
          n_components=5,
          random_state=0
          )
# Create logistic regression
logit = LogisticRegression(
    #penalty='l1',
    #C=1,
    class_weight='balanced')
# Create Support Vector Classification
svc = SVC( 
    kernel='rbf',
    # When C is large, the classifier is heavily penalized for misclassified data
    #C= 1,
    #class_weight='balanced',
    probability=True,
    random_state=0)                        
# Create adaboost
ada = AdaBoostClassifier(#n_estimators=50,
                         # the contribution of each model to the weights.
                         # Setting values less makes a less correction for each tree added to the model.
                         #learning_rate=1,
                         random_state=0
                         )
# Create a pipeline 
pipeline_logit = make_pipeline(standardizer, logit)
pipeline_svc = make_pipeline(standardizer,svc)
pipeline_ada = make_pipeline( standardizer, ada)
# Create space of candidate hyper-parameter values
search_space_logit = [{'logisticregression__C': [ 0.5, 1, 5],
                 'logisticregression__penalty': ['l1', 'l2'],
                }]
search_space_svc = [{'svc__C': [ 0.5, 1, 5],
                     #'svc__kernel': ['rbf', 'linear']
                     'svc__gamma': [0.001, 0.01, 1]
                }]
search_space_ada = [{'adaboostclassifier__n_estimators': [ 100, 500, 1000],
                 'adaboostclassifier__learning_rate': [0.1, 0.5, 1, 2],
                }]
## Feature Selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel

name_features = df_train.columns

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=2)
fvalue_selector.fit(df_train, label_train)
score_f = fvalue_selector.scores_

#Select Important Features In Random Forest

# Create random forest classifier
clf = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1, max_depth = 10)
clf.fit(df_train, label_train)
score_rf = clf.feature_importances_
score_rf
## Plot feature importance by F-test

feature_f = name_features[(-score_f).argsort()][:10]
print( feature_f)
plt.plot( np.arange(42),-np.sort(-score_f))
plt.title('Feature Importance by F-test')
plt.xlabel('Rank Feature')
plt.ylabel('Score')

# Brier score
def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)
scorer = metrics.make_scorer(ProbaScoreProxy, greater_is_better=False, needs_proba=True, class_idx=1, proxied_func=metrics.brier_score_loss)

## Plot feature importance by Random Forest

feature_rf = name_features[(-score_rf).argsort()][:30]
print( feature_rf)
plt.plot( np.arange(42),-np.sort(-score_rf))
plt.title('Feature Importance by RF')
plt.xlabel('Rank Feature')
plt.ylabel('Score')
## Correlation between Target variable churn and net margin
data_train[['margin_net_pow_ele', 'churn']].corr()
## Plot the Histogram
data_train.groupby('churn')['margin_net_pow_ele'].hist()
feature_import = list(set.union( set(feature_rf), set(feature_f)))
feature_import
## Forecast discount percentage counts
data_train.forecast_discount_energy.value_counts()
## HYPER PARAMETER TUNING PROCESS

# Create grid search 
#model, param = pipeline_logit,search_space_logit
#model, param = pipeline_svc,search_space_svc
#model, param = pipeline_ada,search_space_ada 

#clf = GridSearchCV(model, param, cv=5, verbose=0, n_jobs=-1 )
## Fit grid search
#best_model = clf.fit(df_train, label_train)

#best_model.best_params_
## Model Development / Training the models
# We are using 3 algorithms - Logistic Regression, Support Vector Machine, AdaBoost

from sklearn.model_selection import train_test_split
df_train_, df_valid, label_train_, label_valid = train_test_split(df_train, 
                                                    label_train, 
                                                    test_size=0.2, 
                                                    random_state=0)
# Training model with best hyper-parameters

pipeline_logit = make_pipeline(standardizer, LogisticRegression( C = 0.5, penalty = 'l1',   random_state = 0, class_weight='balanced'))
pipeline_logit.fit(df_train_[feature_import],label_train_)

pipeline_svc = make_pipeline( standardizer, SVC( C= 1, kernel = 'rbf', probability = True,   random_state = 0, class_weight='balanced'))
pipeline_svc.fit(df_train_[feature_import],label_train_)
                 

pipeline_ada = make_pipeline(standardizer, AdaBoostClassifier(learning_rate = 0.1,n_estimators= 1000, random_state = 0))
pipeline_ada.fit( df_train_[feature_import],
                 label_train_ )
# Model Evaluation
from sklearn.metrics import classification_report
proba_predict = pipeline_ada.predict_proba( df_train_[feature_import])[:, 1]
threshold = np.percentile( proba_predict,90)
label_predict = 1*(proba_predict > threshold)
# Create a classification report
class_names = ['not churn', 'churn']
print(classification_report(label_train_.values, label_predict, target_names=class_names))
# Plotting of the ROC curve and Area Under Curve (AUC)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(label_train_.values,proba_predict , pos_label=1)
auc =  metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, lw= lw, color='darkorange',
          label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - ROC')
plt.legend(loc="lower right")
plt.show()
# Minimum of net margin on power subscription
min_margin = data_train.margin_net_pow_ele.min()
min_margin
# Maximum of net margin on power subscription
max_margin = data_train.margin_net_pow_ele.max()
max_margin
import sklearn
sklearn.__version__
y= LogisticRegression().fit(df_train, label_train)
(y.coef_[0][23]), df_train.columns[22]
# Prediction for Test data

proba_ada = pipeline_ada.predict_proba(df_test[feature_import])[:, 1]
threshold = np.percentile( proba_ada,90)
label_ada = 1*(proba_ada > threshold)

proba_svc = pipeline_svc.predict_proba(df_test[feature_import])[:, 1]
threshold = np.percentile( proba_svc,90)
label_svc = 1*(proba_svc > threshold)

proba_logit = pipeline_logit.predict_proba(df_test[feature_import])[:, 1]
threshold = np.percentile( proba_logit,90)
label_logit = 1*(proba_logit > threshold)
data_test_output.Churn_prediction = label_ada
data_test_output.Churn_probability = proba_ada
# Save output:
data_test_output.to_csv('./ml_case_test_output.csv', index = False)