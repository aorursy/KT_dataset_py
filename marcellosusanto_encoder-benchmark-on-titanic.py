# Setting package umum 
import pandas as pd
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
%matplotlib inline

from matplotlib.pylab import rcParams
# For every plotting cell use this
# grid = gridspec.GridSpec(n_row,n_col)
# ax = plt.subplot(grid[i])
# fig, axes = plt.subplots()
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 150)
pd.options.display.float_format = '{:.4f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
### Load dataset
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
### Overview dataset
df_train.head(11)
#Function to give quick general information about the dataset
def dataset_summary(df) :
    # Make summary dataframe
    summary_df = pd.DataFrame()

    # Input the characteristic of the dataset
    summary_df['Var'] = df.columns
    summary_df['Dtypes'] = df.dtypes.values
    summary_df['Total Missing'] = df.isnull().sum().values
    summary_df['Missing%'] = summary_df['Total Missing'] / len(df) * 100
    summary_df['Total Unique'] = df.nunique().values
    summary_df['Unique%'] = summary_df['Total Unique'] / len(df) * 100

    # Dataset dimension
    print('Dataset dimension :',df.shape)

    return summary_df
### Summary of dataset
dataset_summary(df_train)
### Summary of dataset test
dataset_summary(df_test)
### Proportion of target variables
df_train['Survived'].value_counts() / len(df_train)
### Check distribution of age
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Plot
sns.distplot(df_train['Age']) ;
### Impute missing value of age using median
df_train['Age'] = df_train['Age'].fillna(np.median(df_train['Age'].dropna()))
df_test['Age'] = df_test['Age'].fillna(np.median(df_train['Age'].dropna()))
### Check proportion of embarked
df_train['Embarked'].value_counts() / len(df_train)
### Impute missing value of embarked using modus
df_train['Embarked'] = df_train['Embarked'].fillna('S')
### Check distribution of fare
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Plot
sns.distplot(df_train['Fare']) ;
### Impute missing value of fare using Q1
df_test['Fare'] = df_test['Fare'].fillna(np.percentile(df_train['Fare'], 25))
### Feature engineering
def feature_engineering(df) :
    # Creating a categorical variable for Ages
    df['AgeCat'] = ''
    df['AgeCat'].loc[(df['Age'] < 18)] = 'young'
    df['AgeCat'].loc[(df['Age'] >= 18) & (df['Age'] < 56)] = 'mature'
    df['AgeCat'].loc[(df['Age'] >= 56)] = 'senior'


    # Creating a categorical variable for Family Sizes
    df['FamilySize'] = ''
    df['FamilySize'].loc[(df['SibSp'] <= 2)] = 'small'
    df['FamilySize'].loc[(df['SibSp'] > 2) & (df['SibSp'] <= 5 )] = 'medium'
    df['FamilySize'].loc[(df['SibSp'] > 5)] = 'large'


    # Creating a categorical variable to tell if the passenger is alone
    df['IsAlone'] = ''
    df['IsAlone'].loc[((df['SibSp'] + df['Parch']) > 0)] = 'no'
    df['IsAlone'].loc[((df['SibSp'] + df['Parch']) == 0)] = 'yes'


    # Creating a categorical variable to tell if the passenger is a Young/Mature/Senior male or a Young/Mature/Senior female
    df['SexCat'] = ''
    df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'youngmale'
    df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age']) < 50] = 'maturemale'
    df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'maturemale'
    df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'youngfemale'
    df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age']) < 50] = 'maturefemale'
    df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'maturefemale'
    
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
### Drop unnecessary columns and get the list of col
def get_feature_names(df):

    # Dropping unused columns from the feature set
    df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

    # Splitting categorical and numerical column dataframes
    categorical_df = df.select_dtypes(include=['object'])
    numeric_df = df.select_dtypes(exclude=['object'])

    # And then, storing the names of categorical and numerical columns.
    categorical_columns = list(categorical_df.columns)
    numeric_columns = list(numeric_df.columns)
    
    print("Categorical columns:\n", categorical_columns)
    print("\nNumeric columns:\n", numeric_columns)

    return categorical_columns, numeric_columns

cat_var, cont_var = get_feature_names(df_train)
_, _ = get_feature_names(df_test)
### Overview of dataset
df_train.head()
### Use to close H2O
# Type "Y"
h2o.cluster().shutdown(prompt=True)
### Initialize h2o
import h2o
h2o.init()
h2o.no_progress()
### Class to do Bayesian Encoding
class mmotokiBetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)

        else:
            num = self.prior_mean
            dem = np.ones_like(N_prior)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
### Make all H2O baseline model
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
import time

def h2o_compare_models(train_data, val_data, X, Y, random_state, cv=5) :
    '''
    Function to compare H2O model
    Note that if you want to change the metrics you have to do it manually change the code
    In this notebook I use accuracy
    
    PARAMS
    train_data : H2OFrame, that contains the training dataset
    val_data : H2OFrame, that contains the validation dataset
    X : list, that contains the predictors used
    Y : string, the target variable
    random_state : int, for model seeding
    cv : int, for CV metrics
    
    OUTPUT
    result : pd.DataFrame, that contains the score for each model
    
    '''
    
    # Preprocess dataframe to H2O Dataset
    h2o_train = h2o.H2OFrame(train_data[X+[Y]])
    h2o_val = h2o.H2OFrame(val_data[X+[Y]])
    
    h2o_train[Y] = h2o_train[Y].asfactor()
    h2o_val[Y] = h2o_val[Y].asfactor()
    
    # Initialize all model (Ganti family/distributionnya)
    glm = H2OGeneralizedLinearEstimator(seed=random_state, family='binomial', nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    gbm = H2OGradientBoostingEstimator(seed=random_state, distribution='bernoulli', nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    xgb = H2OXGBoostEstimator(seed=random_state, distribution='bernoulli', nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    lgbm = H2OXGBoostEstimator(seed=random_state, distribution='bernoulli', tree_method="hist", grow_policy="lossguide",
                              nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    rf = H2ORandomForestEstimator(seed=random_state, distribution='bernoulli', nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    ext = H2ORandomForestEstimator(seed=random_state, distribution='bernoulli', histogram_type="Random",
                                  nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    
    # Train model
    glm.train(x=X, y=Y, training_frame=h2o_train)
    gbm.train(x=X, y=Y, training_frame=h2o_train)
    xgb.train(x=X, y=Y, training_frame=h2o_train)
    lgbm.train(x=X, y=Y, training_frame=h2o_train)
    rf.train(x=X, y=Y, training_frame=h2o_train)
    ext.train(x=X, y=Y, training_frame=h2o_train)
    
    # Calculate train metrics (Bisa diganti)
    from sklearn.metrics import accuracy_score
    train_glm = accuracy_score(h2o_train[Y].as_data_frame(), glm.predict(h2o_train)['predict'].as_data_frame())
    train_gbm = accuracy_score(h2o_train[Y].as_data_frame(), gbm.predict(h2o_train)['predict'].as_data_frame())
    train_xgb = accuracy_score(h2o_train[Y].as_data_frame(), xgb.predict(h2o_train)['predict'].as_data_frame())
    train_lgbm = accuracy_score(h2o_train[Y].as_data_frame(), lgbm.predict(h2o_train)['predict'].as_data_frame())
    train_rf = accuracy_score(h2o_train[Y].as_data_frame(), rf.predict(h2o_train)['predict'].as_data_frame())
    train_ext = accuracy_score(h2o_train[Y].as_data_frame(), ext.predict(h2o_train)['predict'].as_data_frame())

    # Calculate CV metrics for all model (Bisa diganti)
    met_glm = accuracy_score(h2o_train[Y].as_data_frame(), glm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_gbm = accuracy_score(h2o_train[Y].as_data_frame(), gbm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_xgb = accuracy_score(h2o_train[Y].as_data_frame(), xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_lgbm = accuracy_score(h2o_train[Y].as_data_frame(), lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_rf = accuracy_score(h2o_train[Y].as_data_frame(), rf.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_ext = accuracy_score(h2o_train[Y].as_data_frame(), ext.cross_validation_holdout_predictions()['predict'].as_data_frame())
    
    # Calculate holdout metrics
    from sklearn.metrics import accuracy_score
    hold_glm = accuracy_score(h2o_val[Y].as_data_frame(), glm.predict(h2o_val)['predict'].as_data_frame())
    hold_gbm = accuracy_score(h2o_val[Y].as_data_frame(), gbm.predict(h2o_val)['predict'].as_data_frame())
    hold_xgb = accuracy_score(h2o_val[Y].as_data_frame(), xgb.predict(h2o_val)['predict'].as_data_frame())
    hold_lgbm = accuracy_score(h2o_val[Y].as_data_frame(), lgbm.predict(h2o_val)['predict'].as_data_frame())
    hold_rf = accuracy_score(h2o_val[Y].as_data_frame(), rf.predict(h2o_val)['predict'].as_data_frame())
    hold_ext = accuracy_score(h2o_val[Y].as_data_frame(), ext.predict(h2o_val)['predict'].as_data_frame())
    
    # Make result dataframe
    result = pd.DataFrame({'Model':['GLM','GBM','XGB','LGBM','RF','ExtraTree'],
                          'Train Metrics':[train_glm,train_gbm,train_xgb,train_lgbm,train_rf,train_ext],
                          'CV Metrics':[met_glm,met_gbm,met_xgb,met_lgbm,met_rf,met_ext],
                          'Holdout Metrics':[hold_glm,hold_gbm,hold_xgb,hold_lgbm,hold_rf,hold_ext]})
    
    return result.sort_values('CV Metrics') 
# Define dict to store encoder
from category_encoders import JamesSteinEncoder, BackwardDifferenceEncoder, BaseNEncoder, BinaryEncoder
from category_encoders import CatBoostEncoder, CountEncoder, GLMMEncoder, HashingEncoder, HelmertEncoder
from category_encoders import LeaveOneOutEncoder, MEstimateEncoder, OneHotEncoder, OrdinalEncoder, SumEncoder
from category_encoders import PolynomialEncoder, TargetEncoder, WOEEncoder

dict_encoder = {'js':JamesSteinEncoder,
                'backd':BackwardDifferenceEncoder,
                'basen':BaseNEncoder,
                'bin':BinaryEncoder,
                'cat':CatBoostEncoder,
                'count':CountEncoder,
                'glm':GLMMEncoder,
                'hash':HashingEncoder,
                'helmert':HelmertEncoder,
                'leaveone':LeaveOneOutEncoder,
                'mest':MEstimateEncoder,
                'ohe':OneHotEncoder,
                'ord':OrdinalEncoder,
                'sum':SumEncoder,
                'poly':PolynomialEncoder,
                'target':TargetEncoder,
                'woe':WOEEncoder,
                
                # Additional
                'bayes':mmotokiBetaEncoder}
### Make function to benchmark encoder in category encoder package
def ce_benchmark_wrapper(dataset, repetition, list_enc, cat_var, Y) :
    '''
    Function to benchmark encoder using repetitive stratified split
    
    PARAMS
    dataset : pd.DataFrame, that contains the full dataset
    repetition : int, how many repetitive split should be used to benchmark
    list_enc : list, that contains the key names of encoder to benchmark
    cat_var : list, that contains the categorical variable to encode
    Y : string, the target variable
    
    RETURN
    final_result : pd.DataFrame, that contains the avg and std metric score for each encoder
    
    '''
    
    import random
    global dict_encoder
    import gc
    
    metric = []
    metric_std = []
    start = time.time()
    random_state = [random.choice(range(999999999)) for i in range(repetition)]
    
    print('There are',len(list_enc),'encoder for benchmark')
    print('Using',repetition,'repetition')
    
    for i_enc, enc_name in enumerate(list_enc) :
        
        print('')
        print('#'*33)
        print(i_enc+1,':',dict_encoder[enc_name].__name__)
        print('')
        enc_metric = []
        
        for i in range(repetition) :
            
            # Split dataframe
            from sklearn.model_selection import train_test_split
            train_data, val_data = train_test_split(dataset, stratify=dataset[Y], test_size = 0.2, random_state=random_state[i])
            
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
            
            # Encoding
            if enc_name != 'bayes' :
                enc = dict_encoder[enc_name](cols=cat_var)
                list_col = list(train_data.drop(columns=[Y]).columns)

                enc_train_data = enc.fit_transform(train_data[list_col], train_data[Y])
                enc_train_data[Y] = train_data[Y]

                enc_val_data = enc.transform(val_data[list_col])
                enc_val_data[Y] = val_data[Y]
                
            else :
                enc_train_data = train_data.copy().reset_index(drop=True)
                enc_val_data = val_data.copy().reset_index(drop=True)
                N_min = 11

                for var in cat_var :

                    # Fit
                    enc = dict_encoder[enc_name](var)
                    enc.fit(enc_train_data, Y)

                    # Transform
                    enc_train_data[var]  = enc.transform(enc_train_data,  'mean', N_min)
                    enc_val_data[var]  = enc.transform(enc_val_data,  'mean', N_min)
                
            
            # Make model
            list_col = list(enc_train_data.drop(columns=[Y]).columns)
            result = h2o_compare_models(enc_train_data, enc_val_data, list_col, Y, random_state=random_state[i])
            
            # Save metric score
            enc_metric.append(np.mean(result['Holdout Metrics']))
            
            print('Repetition',i+1,'completed')
            
        # Save encoder metric score
        metric.append(np.mean(enc_metric))
        metric_std.append(np.std(enc_metric))
        print('Score :',np.mean(enc_metric),'+-',np.std(enc_metric))
                      
        end = time.time()
        print('\nTime Used :',(end-start)/60)
        
        g = gc.collect()
        
    # Make final result dataframe
    final_result = pd.DataFrame({'Encoder':list_enc, 'Score':metric, 'Std':metric_std})
    
    return final_result.sort_values('Score')
### Try benchmark encoder
list_enc = list(dict_encoder.keys())
list_enc.remove('ohe')
result = ce_benchmark_wrapper(df_train, 5, list_enc, cat_var, 'Survived')
### See the result
result
### Encoding 
Y = 'Survived'
enc_name = 'basen'

enc = dict_encoder[enc_name](cols=cat_var)
list_col = list(df_train.drop(columns=[Y]).columns)

enc_train_data = enc.fit_transform(df_train[list_col], df_train[Y])
enc_train_data[Y] = df_train[Y]

enc_test_data = enc.transform(df_test[list_col])
### Build the model
X = list(enc_train_data.drop(columns=[Y]).columns)
random_state = 11
cv = 5

# Make H2OFrame
h2o_train = h2o.H2OFrame(enc_train_data[X+[Y]])
h2o_test = h2o.H2OFrame(enc_test_data[X])

h2o_train[Y] = h2o_train[Y].asfactor()

# Build LGBM
lgbm = H2OXGBoostEstimator(seed=random_state, distribution='bernoulli', tree_method="hist", grow_policy="lossguide",
                      nfolds=cv, keep_cross_validation_predictions=True, fold_assignment='Modulo')

# Fit LGBM
lgbm.train(x=X, y=Y, training_frame=h2o_train)
### Make prediction 
pred = lgbm.predict(h2o_test)['predict'].as_data_frame()
### Make submission
sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = pred

sub.to_csv('ce_benchmark_sub.csv', index=False)
