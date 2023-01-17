## Load Package
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('russian')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation
import gc

# Plotting Decision tree
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re

# Venn diagram
from matplotlib_venn import venn2

import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_per_train = pd.read_csv('../input/periods_train.csv')
df_test = pd.read_csv('../input/test.csv')
df_per_test      = pd.read_csv('../input/periods_test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
%%time
dtypes={
    'price': 'float32',
    'item_seq_number': 'uint16',
    'image_top_1':'float32',
    'deal_probabilty':'float32'
}
## train 
df_train = pd.read_csv('../input/train.csv',parse_dates=['activation_date'],dtype=dtypes)
df_per_train = pd.read_csv('../input/periods_train.csv',parse_dates=['activation_date','date_from','date_to'])
df_test = pd.read_csv('../input/test.csv',dtype=dtypes,parse_dates=['activation_date'])
%%time
dtypes1 = {
    'item_seq_number':'float16',
    'price':'float32'
}

df_act_trn = pd.read_csv('../input/train_active.csv',dtype=dtypes1,parse_dates=['activation_date'],
                         usecols=['item_id','user_id','city','activation_date']
                        )
df_train.info()
df_train.memory_usage(deep=True)*1e-6
def convert_columns_to_catg(df, column_list):
    for col in column_list:
        print("converting", col.ljust(30), "size: ", round(df[col].memory_usage(deep=True)*1e-6,2), end="\t")
        df[col] = df[col].astype("category")
        print("->\t", round(df[col].memory_usage(deep=True)*1e-6,2))
convert_columns_to_catg(df_train, ['city','region',"param_1","param_2","param_3","parent_category_name","category_name", "user_type"])
df_train.memory_usage(deep=True)/(2**20)
cat_cols=['city','region',"param_1","param_2","param_3","parent_category_name","category_name", "user_type"]
convert_columns_to_catg(df_test,cat_cols)
df_train.to_pickle("train.pkl")
df_test.to_pickle("test.pkl")

# size is shown in bytes again and needs to be converted to megabytes
print("train.csv:", os.stat('../input/train.csv').st_size * 1e-6)
print("train.pkl:", os.stat('train.pkl').st_size * 1e-6)

print("test.csv:", os.stat('../input/test.csv').st_size * 1e-6)
print("test.pkl:", os.stat('test.pkl').st_size * 1e-6)
%%time 
df_train = pd.read_pickle('train.pkl')
df_train.region.value_counts().tail()
df_train.user_id.value_counts().tail()
from sklearn.preprocessing import LabelEncoder
def create_label_encoding_with_min_count(df, column, min_count=50):
    column_counts = df.groupby([column])[column].transform("count").astype(int)
    column_values = np.where(column_counts >= min_count, df[column], "")
    df[column+"_label"] = LabelEncoder().fit_transform(column_values)
    
    return df[column+"_label"]
print("number of unique users      :", len(df_train["user_id"].unique()))
df_train.loc[df_train["city"]=="Светлый", "region"].value_counts().head()
df_train['region_city'] = df_train.loc[:,['region','city']].apply(lambda s: ' '.join(s),axis=1)
print("unique:", len(df_train["region_city"].unique()))
print("size:  ", df_train["region_city"].memory_usage(deep=True)*1e-6)
df_train['region_city2'] = df_train.groupby(['region','city'])['region'].transform(lambda x:np.random.random()) ## faster and encode it correctly!!
df_train.region_city2.value_counts().head()
print("unique:", len(df_train["region_city2"].unique()))
print("size:  ", df_train["region_city2"].memory_usage(deep=True)*1e-6)
df_train['region_city2_label']=create_label_encoding_with_min_count(df_train,'region_city2',min_count=50)
df_train.columns
gc.collect()
df_train['title'] = df_train.title.fillna(" ")
df_train['title_len'] = df_train.title.apply(lambda x:len(x.split())).astype('uint8')
df_train['title_char'] = df_train.title.apply(len).astype('uint8')
df_train.title_len.value_counts(sort=False).plot(kind='bar')
df_train.title_char.value_counts(sort=False).plot(kind='bar')
df_train['description'] = df_train.description.fillna(" ")
df_train['description_len'] = df_train.description.apply(lambda x:len(x.split())).astype('uint16')
df_train['description_char'] = df_train.description.apply(len).astype('uint16')
ax = df_train.description_len.value_counts(sort=False).plot(kind='bar',log=True)
ax.get_xaxis().set_visible(False)
df_train.description_char.value_counts().head().plot(kind='bar',log=True)
df_train.corr()
corr = df_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")

cmap = sns.diverging_palette(30,10,as_cmap=True)
sns.heatmap(corr,cmap=cmap,center=0,square=True,vmax=.3,linewidths=.1, cbar_kws={"shrink": .5});
import scipy.sparse as sp
def get_df_matrix_mappings(df, row_name, col_name):
    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid


    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid


    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid
rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = get_df_matrix_mappings(df_train,'user_id','item_seq_number')
df_trn_uidx = pd.DataFrame()
df_trn_uidx['uidx']= df_train.user_id.map(rid_to_idx)
df_trn_uidx['iidx']= df_train.item_seq_number.map(cid_to_idx)
df_trn_uidx['uid'] = df_train.user_id
df_trn_uidx['iid'] = df_train.item_seq_number
df_trn_uidx.head()
I = df_trn_uidx.uidx.as_matrix()
J = df_trn_uidx.iidx.as_matrix()
V = np.ones(df_trn_uidx.shape[0])

ui_trn_sp = sp.coo_matrix((V,(I,J)),dtype='uint8')
ui_trn_sp.shape
ui_trn_csr =ui_trn_sp.tocsr()
plt.spy(ui_trn_csr,markersize=0.5,aspect='auto')
plt.plot(np.array(ui_trn_csr.sum(axis=1)).flatten())
df_train['iidx'] = df_train.item_seq_number.map(cid_to_idx).astype('uint16')
df_train['uidx'] = df_train.user_id.map(rid_to_idx).astype('uint32')
df_train.groupby(['uidx','iidx']).size().value_counts()
df_train.columns
df_trn_uidx = df_trn_uidx.merge(df_train[['uidx','iidx','deal_probability']] , how ='left',on=['uidx','iidx'])
I = df_trn_uidx.uidx.as_matrix()
J = df_trn_uidx.iidx.as_matrix()
Vp = df_trn_uidx.deal_probability
ui_trn_deal = sp.coo_matrix((Vp,(I,J)),dtype='float32')
plt.spy(ui_trn_deal,markersize=0.5,aspect='auto')
data = ui_trn_deal.tocsc() # sparse operations are more efficient on csc
N, M = data.shape
s, t = 100, 1000           # decimation factors for y and x directions
T = sp.csc_matrix((np.ones((M,)), np.arange(M), np.r_[np.arange(0, M, t), M]), (M, (M-1) // t + 1))
S = sp.csr_matrix((np.ones((N,)), np.arange(N), np.r_[np.arange(0, N, s), N]), ((N-1) // s + 1, N))
result = S @ data @ T     # downsample by binning into s x t rectangles
result = result.todense() # ready for plotting
plt.imshow(result,cmap='gray_r',aspect='auto')
df_train.groupby('uidx')['deal_probability'].mean().rolling(1000).mean().plot()
tmp = df_train.groupby('uidx').size().to_frame().reset_index().rename(columns={0:'ads_cnt_by_uid'})
tmp.head()
print('doing add cnt by user_id...')
tmp = df_train.groupby('uidx').size().to_frame().reset_index().rename(columns={0:'ads_cnt_by_uid'})
tmp['ads_cnt_by_uid'] = tmp.ads_cnt_by_uid.astype('uint32')
df_train = df_train.merge(tmp,how='left' ,on='uidx')

print('doing add cnt by iidx(item_seq_number)...')
tmp = df_train.groupby('iidx').size().to_frame().reset_index().rename(columns={0:'ads_cnt_by_iid'})
tmp['ads_cnt_by_iid'] = tmp.ads_cnt_by_iid.astype('uint32')
df_train =  df_train.merge(tmp,how='left' ,on='iidx')
print('done')
del tmp; gc.collect()
usecols = ['uidx','iidx','ads_cnt_by_uid','ads_cnt_by_iid','image_top_1','deal_probability']

corr = df_train[usecols].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")

cmap = sns.diverging_palette(30,10,as_cmap=True)
sns.heatmap(corr,cmap=cmap,center=0,square=True,vmax=.3,linewidths=.1, cbar_kws={"shrink": .5});
df_train.groupby('ads_cnt_by_iid')['deal_probability'].mean().plot()
df_train.groupby('iidx')['deal_probability'].mean().rolling(100).mean().plot()
df_train.user_id.nunique()
df_test.user_id.nunique()
uid_in_trn_test = np.intersect1d(df_test.user_id,df_train.user_id) # overlap # of user 67,929
pd.Series(uid_in_trn_test).nunique()
test_uid = df_test.user_id.unique()
train_uid = df_train.user_id.unique()
train_itemid = df_train.item_id.unique()
test_itemid = df_test.item_id.unique()
train_itemid.size
test_itemid.size
itmid_in_trn_test = np.intersect1d(train_itemid,test_itemid)
itmid_in_trn_test.size
num_items_by_user = np.array(ui_trn_csr.sum(axis=1).flatten())[0]
pd.Series(num_items_by_user).value_counts().head()
print('max of df_train.activation_date',df_train.activation_date.max())
print('min of df_train.activation_date',df_train.activation_date.min())
df_train.activation_date.value_counts()
print('max of df_per_train.activation_date',df_per_train.activation_date.max())
print('min of df_per_train.activation_date',df_per_train.activation_date.min())

print('max of df_per_train.date_from',df_per_train.date_from.max())
print('min of df_per_train.date_from',df_per_train.date_from.min())

print('max of df_per_train.date_to',df_per_train.date_to.max())
print('min of df_per_train.date_to',df_per_train.date_to.min())
df_act_trn.activation_date.value_counts()
df_per_train.shape
df_act_trn.shape
df_per_train.activation_date.value_counts()
df_per_train.date_from.value_counts()
print('# of item_id : train active ',df_act_trn.item_id.nunique())
print('# of item_id : periods train', df_per_train.item_id.nunique())
print('# of item_id: train ',df_train.item_id.nunique())
df_per_train.head()
df_act_trn.head()
df_train.merge(df_per_train,how='inner',on=['item_id'])
df_train[df_train.user_id.isin(df_act_trn.user_id.head())]
act_trn_itemid = df_act_trn.item_id.head(10)
df_per_train[df_per_train.item_id.isin(act_trn_itemid)]
df_act_trn.head(20).merge(df_per_train,left_on=['item_id','activation_date'],right_on=['item_id','date_from'],how='inner')
df_per_train.head(10).merge(df_act_trn,on='item_id',how='inner')
df_per_train.columns
df_act_trn.columns
merged_trn_sup = df_act_trn[['item_id','activation_date','user_id']].merge(df_per_train,
                                                                                           how='inner',
                                                                                           left_on=['item_id','activation_date'],
                                                                                           right_on=['item_id','date_from'])
df_act_trn.shape
merged_trn_sup.shape
merged_trn_sup.head()
merged_trn_sup.date_from.max()
df_train.activation_date.value_counts()
df_act_trn.activation_date.value_counts()
merged_trn_sup.date_to.value_counts()
df_train.columns
df_train.deal_probability.mean()
df_train.user_type.value_counts()
df_train.parent_category_name.value_counts()
df_train.groupby('parent_category_name').deal_probability.mean()
df_train.groupby('category_name').deal_probability.mean()
df_train.shape[0] == np.sum(df_train.user_type.value_counts())
df_train.groupby('user_type').deal_probability.mean()
df_train.groupby('user_id').size().sort_values(ascending=False).head()
df_train.groupby('image_top_1').deal_probability.mean()
df_train.groupby('category_name').deal_probability.mean()
df_train.groupby('item_seq_number').deal_probability.mean()
title_text_raw = df_train.title.append(df_test.title)
title_text_raw.reset_index(drop=True,inplace=True)
title_text_raw.shape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
## tfidf 
tv = TfidfVectorizer(lowercase=False,ngram_range=(1,2),max_features=100000)
tv_feats = tv.fit_transform(title_text_raw)
print('shape of tfidf Vectorizer:{}'.format(tv_feats.shape))
svd = TruncatedSVD(n_components=5, random_state=0)
tv_svd_feats = svd.fit_transform(tv_feats)
print('shape of tv_svd_feats:',tv_svd_feats.shape)
print(svd.explained_variance_ratio_)
print(np.cumsum(svd.explained_variance_ratio_))
tv_svd_df = pd.DataFrame(tv_svd_feats).iloc[:df_train.shape[0]]
tv_svd_df['y'] = df_train.deal_probability

tv_svd_df.corr()['y']
sns.jointplot(x = tv_svd_df[0].values, y=tv_svd_df['y'].values)
sns.jointplot(x = tv_svd_df[1].values, y=tv_svd_df['y'].values)
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(ngram_range=(1, 2), lowercase=False)
hv_features = hv.fit_transform(title_text_raw).tocsr()
print('shape of hv features:{}'.format(hv_features.shape))

svd = TruncatedSVD(n_components=5, random_state=0)
hv_svd_features = svd.fit_transform(hv_features)
np.cumsum(svd.explained_variance_ratio_)
hv_svd_df = pd.DataFrame(hv_svd_features).iloc[:df_train.shape[0]]
hv_svd_df['y'] = df_train.deal_probability
hv_svd_df.corr().y
#desc_raw = df_train.description.append(df_test.description)
#desc_raw.fillna('',inplace=True)
#desc_raw.reset_index(drop=True,inplace=True)
## tfidf  + svd 
#tv = TfidfVectorizer(lowercase=False,ngram_range=(1,2),max_features=100000)
#tv_feats = tv.fit_transform(desc_raw)

#print('shape of tfidf Vectorizer:{}'.format(tv_feats.shape))

#svd = TruncatedSVD(n_components=5, random_state=0)
#tv_svd_feats1 = svd.fit_transform(tv_feats)
#print('shape of tv_svd_feats:',tv_svd_feats.shape)
#svd.explained_variance_ratio_
## hashing + svd 

#hv = HashingVectorizer(ngram_range=(1, 2), lowercase=False)
#hv_features = hv.fit_transform(desc_raw).tocsr()
#print('shape of hv features:{}'.format(hv_features.shape))

#svd = TruncatedSVD(n_components=5, random_state=0)
#hv_svd_features1 = svd.fit_transform(hv_features)
#hv_svd_features1.shape
#tv_svd_feats1.shape
def init_seeds(seed):
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return sess
k_latent = 2
embedding_reg = 0.0002
kernel_reg = 0.1

def get_embed(x_input, x_size, k_latent):
    if x_size > 0: #category
        embed = Embedding(x_size, k_latent, input_length=1, 
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed

def build_model_1(X, f_size):
    dim_input = len(f_size)
    
    input_x = [Input(shape=(1,)) for i in range(dim_input)] 
     
    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]
    
    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]
    
    s = Add()(factors)
    
    diffs = [Subtract()([s, x]) for x in factors]
    
    dots = [Dot(axes=1)([d, x]) for d,x in zip(diffs, factors)]
    
    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    opt = Adam(clipnorm=0.5)
    model.compile(optimizer=opt, loss='mean_squared_error')
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)
    return model, model_features
%%time 
df_train = pd.read_pickle('train.pkl')
print('build id->idx map ...')
rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = get_df_matrix_mappings(df_train,'user_id','item_seq_number')


df_trn_uidx = pd.DataFrame()
df_trn_uidx['uidx']= df_train.user_id.map(rid_to_idx)
df_trn_uidx['iidx']= df_train.item_seq_number.map(cid_to_idx)
df_trn_uidx['uid'] = df_train.user_id
df_trn_uidx['iid'] = df_train.item_seq_number

print('build iidx, uidx col')
df_train['iidx'] = df_train.item_seq_number.map(cid_to_idx).astype('uint16')
df_train['uidx'] = df_train.user_id.map(rid_to_idx).astype('uint32')
feats = ['uidx','iidx']

target = ['deal_probability']
fm_data = df_train[feats].copy()
fm_data.head()
fm_data.info()
f_size  = [int(fm_data[f].max()) + 1 for f in feats]
f_size
fm_data.dtypes
fm_data = fm_data.merge(df_train[['uidx','iidx','deal_probability']], how='left',on=['uidx','iidx'])
fm_data.head()