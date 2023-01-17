import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import json
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import re
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result
PATH_TO_RAW_DATA = 'raw_data/'
PATH_TO_PROCESSED_DATA = 'processed_data/'
def preprocess(path_to_inp_json_file, path_to_out_txt_file):
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file, \
        open(path_to_out_txt_file, 'w', encoding='utf-8') as out_file:
        for line in tqdm_notebook(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            out_file.write(content_no_html_tags + '\n')
%%time
preprocess(path_to_inp_json_file=os.path.join(PATH_TO_RAW_DATA, 'train.json'),
           path_to_out_txt_file=os.path.join(PATH_TO_PROCESSED_DATA, 'train_raw_content.txt'))
%%time
preprocess(path_to_inp_json_file=os.path.join(PATH_TO_RAW_DATA, 'test.json'),
           path_to_out_txt_file=os.path.join(PATH_TO_PROCESSED_DATA, 'test_raw_content.txt'))
cv = CountVectorizer(ngram_range=(1, 2), max_features=100000)
%%time
with open(os.path.join(PATH_TO_PROCESSED_DATA, 'train_raw_content.txt'), encoding='utf-8') as input_train_file:
    X_train_raw_content = cv.fit_transform(input_train_file)
%%time
with open(os.path.join(PATH_TO_PROCESSED_DATA, 'test_raw_content.txt'), encoding='utf-8') as input_test_file:
    X_test_raw_content = cv.transform(input_test_file)
#prev
X_train_raw_content.shape, X_test_raw_content.shape
data = []
def if_exists(feature):
    if type(feature) is None:
        return 0
    else:
        return 1

def extract_features_and_write(path_to_data,
                               inp_filename, is_train=True):
    
    features = ['content', 'published', 'title', 'author']
    prefix = 'train' if is_train else 'test'
    feature_files = [open(os.path.join(path_to_data,
                                       '{}_{}.txt'.format(prefix, feat)),
                          'w', encoding='utf-8')
                     for feat in features]
    with open(os.path.join(path_to_data, inp_filename), 
              encoding='utf-8') as inp_json_file:
        
        for line in tqdm_notebook(inp_json_file):
            json_data = read_json_line(line)
            df = {}
            #df['id']  = json_data['_id']
            #df['timestamp'] = json_data['_timestamp']
            #df['spider'] = json_data['_spider']
            df['domain'] = json_data['domain']
            df['published'] = json_data['published']['$date']
            df['title'] = json_data['title']
            df['content'] = strip_tags(json_data['content'])
            #df['author_name'] = json_data['author']['name']
            #df['img'] = if_exists(json_data['image_url'])
            #df['tags'] = len(json_data['tags'])
            df['description'] = json_data['meta_tags']['description']
            df['author'] = json_data['meta_tags']['article:author']
            df['robots'] = json_data['meta_tags']['robots'].split(',')[0] #lambda x: 1 if 'noindex' not in json_data['meta_tags']['robots'] else 0
            df['min_to_read'] = re.split('\s', json_data['meta_tags']['twitter:data1'])[0]
            data.append(df)
        print(json_data)
%%time
extract_features_and_write(PATH_TO_RAW_DATA, 'train.json', is_train=True)
df_train = pd.DataFrame(data=data)
df_train.head(1)
feat_train = pd.read_csv(os.path.join(PATH_TO_RAW_DATA, 'train_log1p_recommends.csv'))
y_train = feat_train['log_recommends'].values
feat_train.drop('log_recommends', axis = 1).head(1)
df_train[df_train.columns] = df_train[df_train.columns].fillna(0).astype('O')
df_train.published = df_train.published.astype('datetime64[ns]')
df_train.min_to_read = df_train.min_to_read.astype(int)
df_train.dtypes
# sequence of indices
feat_flatten = feat_train.values.flatten()

# and the matrix we are looking for
feat_train_sparse = csr_matrix(([1] * feat_flatten.shape[0],
                                feat_flatten,
                                range(0, feat_flatten.shape[0]  + 10, 10)))[:, 1:]
def fit_feature(lmdb, feature, reason, axis = 0):
    feat_train[feature] = df_train[reason].apply(lmbd, axis)
    #feat_test[feature] = test_df[reason].apply(lmbd, axis)
    if axis != 0:
        feat_train[feature].values.reshape(len(df_train[reason]), 1)
        #feat_test[feature].values.reshape(len(test_df[reason]), 1)
    scaler = StandardScaler()
    scaled_feature = feature + '_scaled'
    feat_train[scaled_feature] = scaler.fit_transform(feat_train[feature].values.reshape(-1, 1))
    feat_test[scaled_feature] = scaler.transform(feat_test[feature].values.reshape(-1, 1))
lmbd = lambda ts: ts.date().weekday()
feature = 'weekday'
column = 'published'
fit_feature(lmbd,feature,column)

lmbd = lambda x: 1 if x.date().weekday() in (5, 6) else 0
feature = 'is_weekend'
column = 'published'
fit_feature(lmbd,feature,column)
feat_train['time_to_read'] = df_train['min_to_read']

lmbd = lambda ts: len(ts)
feature = 'text_length'
reason = 'content'
fit_feature(lmbd,feature,reason)
lmbd = lambda ts: ts.hour
feature = 'hour'
reason = 'published'
fit_feature(lmbd,feature,reason)

lmbd = lambda ts: int(ts.hour > 6 and ts.hour <= 12)
feature = 'morning'
reason = 'published'
fit_feature(lmbd,feature,reason)

lmbd = lambda ts: int(ts.hour > 12 and ts.hour <= 18)
feature = 'afternoon'
reason = 'published'
fit_feature(lmbd,feature,reason)

lmbd = lambda ts: int(ts.hour > 18 and ts.hour <= 23)
feature = 'evening'
reason = 'published'
fit_feature(lmbd,feature,reason)

lmbd = lambda ts: int(ts.hour > 23 and ts.hour <= 6)
feature = 'night'
reason = 'published'
fit_feature(lmbd,feature,reason)
feat_train['morning'] = feat_train['morning'].astype(int)
feat_train['afternoon'] = feat_train['afternoon'].astype(int)
feat_train['evening'] = feat_train['evening'].astype(int)
feat_train['night'] = feat_train['night'].astype(int)
feat_train['weekday'] = feat_train['weekday'].astype(int)
feat_train['hour'] = feat_train['hour'].astype(int)
feat_train['is_weekend'] = feat_train['is_weekend'].astype(int)
feat_train['text_length'] = feat_train['text_length'].astype(int)
feat_train.dtypes
#the partition migh be more interesting and correct
train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid =  X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]
tmp = csr_matrix(feat_train['morning'])
X_train = csr_matrix(hstack([X_train_raw_content[:], tmp[:].T]))
def tokens(x):
    return x.split(',')

tfidf_vect= TfidfVectorizer( tokenizer=tokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
X_train_tfidf =tfidf_vect.fit_transform(X_train)
X_train_tfidf.shape
count_vect.fit_transform(X_train_tfidf)
%%time
ridge = Ridge(random_state=17)
ridge.fit(X=X_train_title_tfidf,y=y_train_part)
ridge_pred = ridge.predict(X_valid)
plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
plt.legend();
valid_mae = mean_absolute_error(y_valid, ridge_pred)
valid_mae, np.expm1(valid_mae)
%%time
ridge.fit(X_train, y_train);
%%time
ridge_test_pred = ridge.predict(X_test)
def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_RAW_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)
write_submission_file(prediction=ridge_test_pred, 
                      filename='result_ridge.csv')