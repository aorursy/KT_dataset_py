import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# load data
cat_cols = [
    'region', 'city', 'parent_category_name', 'category_name',
    'param_1', 'param_2', 'param_3', 'image_top_1'
]

train = pd.read_csv('../input/train.csv', usecols=cat_cols+['deal_probability'])
test = pd.read_csv('../input/test.csv', usecols=cat_cols)
# fill in missing
train['image_top_1'] = train['image_top_1'].astype(str)
test['image_top_1'] = test['image_top_1'].astype(str)
train.fillna('', inplace=True)
test.fillna('', inplace=True)
# label encoding
for col in cat_cols:
    le = LabelEncoder()
    le.fit(np.concatenate([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
train.head()
test.head()
class BetaEncoder(object):
        
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
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
N_min = 1000
feature_cols = []    

# encode variables
for c in cat_cols:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(train, 'deal_probability')

    # mean
    feature_name = f'{c}_mean'
    train[feature_name] = be.transform(train, 'mean', N_min)
    test[feature_name]  = be.transform(test,  'mean', N_min)
    feature_cols.append(feature_name)

    # mode
    feature_name = f'{c}_mode'
    train[feature_name] = be.transform(train, 'mode', N_min)
    test[feature_name]  = be.transform(test,  'mode', N_min)
    feature_cols.append(feature_name)
    
    # median
    feature_name = f'{c}_median'
    train[feature_name] = be.transform(train, 'median', N_min)
    test[feature_name]  = be.transform(test,  'median', N_min)
    feature_cols.append(feature_name)    

    # var
    feature_name = f'{c}_var'
    train[feature_name] = be.transform(train, 'var', N_min)
    test[feature_name]  = be.transform(test,  'var', N_min)
    feature_cols.append(feature_name)        
    
    # skewness
    feature_name = f'{c}_skewness'
    train[feature_name] = be.transform(train, 'skewness', N_min)
    test[feature_name]  = be.transform(test,  'skewness', N_min)
    feature_cols.append(feature_name)    
    
    # kurtosis
    feature_name = f'{c}_kurtosis'
    train[feature_name] = be.transform(train, 'kurtosis', N_min)
    test[feature_name]  = be.transform(test,  'kurtosis', N_min)
    feature_cols.append(feature_name)    
    
train[['deal_probability']+feature_cols].head()
test[feature_cols].head()