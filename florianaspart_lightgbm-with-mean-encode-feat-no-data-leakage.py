import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
DATA_PATH = '/media/florian/8fd68a96-fc7e-47a1-a13e-4c4f910f6e51/ML_Data/avito/'

print(os.listdir(DATA_PATH))
data_tr = pd.read_csv(DATA_PATH+'/train.csv')
data_te = pd.read_csv(DATA_PATH+'/test.csv')
print('train data shape is :', data_tr.shape)
print('test data shape is :', data_te.shape)
data_tr, data_va = train_test_split(data_tr, shuffle=True, 
                              test_size=0.05, random_state=42)
def preprocessData(data):

    # Extract date info from the activate date
    data.activation_date    = pd.to_datetime(data.activation_date)

    data['day_of_month']    = data.activation_date.apply(lambda x: x.day)
    data['day_of_week']     = data.activation_date.apply(lambda x: x.weekday())

    # Extract info from the title
    data['char_len_title']  = data.title.apply(lambda x: len(str(x)))
    data['char_len_desc']   = data.description.apply(lambda x: len(str(x)))
preprocessData(data_te)
preprocessData(data_tr)
preprocessData(data_va)

# Encore the city, category_name and user_type labels
cate_cols = ['city',  'category_name', 'user_type']

for c in cate_cols:
    le = LabelEncoder()
    allvalues = np.unique(data_tr[c].values).tolist() \
                + np.unique(data_va[c].values).tolist() \
                + np.unique(data_te[c].values).tolist()
    le.fit(allvalues)
    
    for d in [data_tr, data_va, data_te]:
        d[c] = le.transform(d[c].values)
del d
class FeaturesStatistics():
    def __init__(self):
        self._stats = None
        self._agg_cols  = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_month','day_of_week']
    
    def fit(self,df):
        '''
        Compute the mean and std of some features from a given data frame
        '''
        self._stats             = {}
        
        # For each feature to be aggregated
        for c in tqdm(self._agg_cols,total=len(self._agg_cols)):
            # Compute the mean and std of the deal prob and the price.
            gp              = df.groupby(c)[['deal_probability','price']]
            desc            = gp.describe()
            self._stats[c]  = desc[ [('deal_probability','mean'),('deal_probability','std'),
                                     ('price','mean')] ]

    def transform(self,df):
        '''
        Add the mean features statistics computed from another dataset.
        '''
        # For each feature to be aggregated
        for c in tqdm(self._agg_cols,total=len(self._agg_cols)):
            # Add the deal proba and price statistics corrresponding to the feature
            df[c+'_deal_probability_mean']  = df[c].map(self._stats[c][('deal_probability','mean')])
            df[c+'_deal_probability_std']   = df[c].map(self._stats[c][('deal_probability','std')])
            df[c+'_price_mean']             = df[c].map(self._stats[c][('price','mean')])
            
        
    def fit_transform(self,df):
        '''
        First learn the feature statistics, then add them to the dataframe.
        '''
        self.fit(df)
        self.transform(df)
fStats = FeaturesStatistics()
fStats.fit_transform(data_tr)
fStats.transform(data_va)
fStats.transform(data_te)
col_to_drops = ['activation_date','user_id','description',
                'image','parent_category_name','region',
                'item_id','param_1','param_2','param_3','title']

y_tr = data_tr['deal_probability']
X_tr = data_tr.drop(col_to_drops+['deal_probability'],axis=1)

#y_te = data_te['deal_probability']
X_te = data_te.drop(col_to_drops,axis=1)

y_va = data_va['deal_probability']
X_va = data_va.drop(col_to_drops+['deal_probability'],axis=1)

print(X_tr.shape, X_va.shape, X_te.shape)
import gc
del data_tr, data_te, data_va
gc.collect()
# Create the LightGBM data containers
tr_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cate_cols)
va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cate_cols, reference=tr_data)
del X_tr
del X_va
del y_tr
del y_va
gc.collect()
# Train the model
parameters = {
    'task':             'train',
    'boosting_type':    'gbdt',
    'objective':        'regression',
    'metric':           'rmse',
    'num_leaves':       31,
    'learning_rate':    0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'verbose':          50
}


model = lgb.train(parameters,
                  tr_data,
                  valid_sets=va_data,
                  num_boost_round=2000,
                  early_stopping_rounds=120,
                  verbose_eval=50)
y_pred = model.predict(X_te)
sub = pd.read_csv(DATA_PATH+'/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('lgb_with_mean_encode.csv', index=False)
sub.head()
%matplotlib inline
lgb.plot_importance(model, importance_type='gain', figsize=(10,20))
