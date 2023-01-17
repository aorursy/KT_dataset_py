import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Text
import re
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

# Options
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
print("Data Load Stage")
%time training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
train_len = training.shape[0]
%time testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)

print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del (training, testing); gc.collect()
print('All Data shape: {} Rows, {} Columns'.format(*df.shape))
df.head()
print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-1000,inplace=True)
df["image_top_1"].fillna(-1,inplace=True)

print("\nCreate Time Variables")
df["weekday"] = df['activation_date'].dt.weekday
df["woy"] = df['activation_date'].dt.week
df["dom"] = df['activation_date'].dt.day
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nText Features") 
df['group'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

df.description.fillna('nan', inplace=True)
df.title.fillna('nan', inplace=True)

df.head()
print('Rename columns')
colnames_mapper = {
    'parent_category_name': 'cat_1',
    'category_name': 'cat_2',
    'description': 'desc',
    'image_top_1': 'img_code',
    'item_seq_number': 'item',
    'region': 'reg',
    'user_id': 'usr',
    'user_type': 'usr_type',
}

cat_cols = ['usr', 'usr_type', 'reg', 'city', 'item', 'cat_1', 'cat_2', 'img_code', 'weekday']
text_cols = ['title', 'desc', 'group']
other_cols = ['price',  'woy', 'dom']
target_col = 'deal_probability'

df = df.rename(index=str, columns=colnames_mapper)[cat_cols + other_cols + text_cols + [target_col]]
df.head()
print("\nEncode Categorical Variables")
# Encoder:
lbl = LabelEncoder()
for col in tqdm(cat_cols):
    df[col] = lbl.fit_transform(df[col].astype(str))
    col_max = df[col].max()
    if col_max < 2**8 - 1:
        df[col] = df[col].astype('uint8')
    elif col_max < 2**16 - 1:
        df[col] = df[col].astype('uint16')
    elif col_max < 2**32 - 1:
        df[col] = df[col].astype('uint32')
    
del(col_max)     
df.head()
russian_stop = set(stopwords.words('russian'))
snowball_stemmer = SnowballStemmer("russian")

def vw_extractor(row):

    output_line = '|cat '
    for col in cat_cols:
        output_line += '{0}_{1} '.format(col, row[col])

    n_upper = {}
    n_exclamation = {}
    n_uniq_words = {}
    for col in text_cols:
        n_exclamation[col] = row[col].count('!')
        n_upper[col] = sum(1 for ch in row[col] if ch.isupper())

        text = row[col].lower()
        words = set(re.findall('\w+', text)).difference(russian_stop)
        n_uniq_words[col] = len(words)

        stemmed_words = {snowball_stemmer.stem(word) for word in words}
        output_line += '|{} '.format(col)
        output_line += ' '.join(stemmed_words) + ' '

    output_line += '|stat_text '
    output_line += 'n_up_tit:{0} n_up_desc:{1} '.format(n_upper['title'], n_upper['desc'])
    output_line += 'n_exc_tit:{0} n_exc_desc:{1} '.format(n_exclamation['title'], n_exclamation['desc'])
    output_line += 'n_uniq_tit:{0} n_uniq_desc:{1} '.format(n_uniq_words['title'], n_uniq_words['desc'])

    output_line += '|other '
    for col in other_cols:
        output_line += '{0}:{1:.6} '.format(col, float(row[col]))

    return output_line

# just check our vw format before transformation
for row in tqdm(df.head().iterrows(), total=df.head().shape[0], miniters=1000):
    print(vw_extractor(row[1]), '\n')
def train2vw(data, features_extractor, valid_rate=0, train_output='train', valid_output='valid', yvalid_output='yvalid'):
    writer_train = open(train_output, 'w')
    writer_val = open(valid_output, 'w')
    writer_yval = open(yvalid_output, 'w')
    
    for row in tqdm(data.iterrows(), total=data.shape[0], miniters=2500):
        label = row[1][target_col]
        features = features_extractor(row[1])
        output_line = '{0:.6} {1}\n'.format(label, features)
        
        if np.random.rand() > valid_rate:
            writer_train.write(output_line)
        else:
            writer_val.write(output_line)
            writer_yval.write('%s\n' % label)
            
    writer_train.close()
    writer_val.close()
    writer_yval.close()
    
train2vw(df[:train_len], vw_extractor, 0.2)
def get_rmse(ytest_input='ytest', pred_input='pred'):
    n, loss = 0, 0
    reader_ytest = open(ytest_input, 'r')
    reader_pred = open(pred_input, 'r')

    for label, pred in tqdm(zip(reader_ytest, reader_pred)):    
        n+=1
        true_score = float(label)
        pred_score = float(pred)
        loss += np.square(pred_score - true_score)
    reader_ytest.close()
    reader_pred.close()
    return np.sqrt(loss / n)
# ! vw -d train --loss_function squared -f model -b 16 --passes 10 --cache_file cache --quiet
# ! vw -i model -t valid -r pred --quiet
# print('Validation RMSE: {:.5}'.format(get_rmse('yvalid', 'pred')))

print('Validation RMSE: 0.24855')
# ! vw -d train --loss_function squared --learning_rate 0.01 -f model -b 20 --passes 10 --cache_file cache --quiet
# ! vw -i model -t valid -r pred --quiet
# print('Validation RMSE: {:.5}'.format(get_scores('yvalid', 'pred')))

print('Validation RMSE: 0.22743')
# ! vw -d train --loss_function squared --learning_rate 0.01 -f model -b 26 --passes 20 --cache_file cache --quiet
# ! vw -i model -t valid -r pred --quiet
# print('Validation RMSE: {:.5}'.format(get_rmse('yvalid', 'pred')))

print('Validation RMSE: 0.22686')
def test2vw(data, features_extractor, test_output='test'):
    writer_test = open(test_output, 'w')
    
    for row in tqdm(data.iterrows(), total=data.shape[0], miniters=2500):
        features = features_extractor(row[1])
        output_line = '{0}\n'.format(features)
        writer_test.write(output_line)
            
    writer_test.close()

# need more hard memory on server, so I comment next line    
# test2vw(df[train_len:], vw_extractor)
# !cp train full_train
# !cat valid >> full_train

# ! vw -i model -t test -r pred --quiet

# sub = pd.read_csv('pred', header=None)
# sub.index = df[train_len:].index
# sub.columns = [target_col]
# sub.reset_index(inplace=True)
# sub[target_col].clip(0.0, 1.0, inplace=True) 
# sub.to_csv("vw_sub.csv",index=False, header=True)
print('end')
