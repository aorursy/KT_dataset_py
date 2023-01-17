import numpy as np
import pandas as pd
import gc

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GRU, Embedding, concatenate, Dropout, BatchNormalization, Flatten, LSTM
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
data = pd.read_csv("../input/avito-dataset/train_fe_3.csv")
data.drop(data.columns[0], axis = 1, inplace = True)


data['param_1'].fillna(value='missing', inplace=True)
data['param_2'].fillna(value='missing', inplace=True)
data['param_3'].fillna(value='missing', inplace=True)
    
data['param_1'] = data['param_1'].astype(str)
data['param_2'] = data['param_2'].astype(str)
data['param_3'] = data['param_3'].astype(str)

data['category_name'] = data['category_name'].astype('category')
data['parent_category_name'] = data['parent_category_name'].astype('category')
data['region'] = data['region'].astype('category')
data['city'] = data['city'].astype('category')

data['param123'] = (data['param_1']+'_'+data['param_2']+'_'+data['param_3']).astype(str)
del data['param_2'], data['param_3']
gc.collect()

data['title_description']= (data['title']+" "+data['description']).astype(str)
del data['description'], data['title']
gc.collect()

#data['title'] = data['title'].astype(str)
#data['description'] = data['description'].astype(str)

max_seq_title_description_length = 100
max_words_title_description = 200000

tokenizer = text.Tokenizer(num_words = max_words_title_description)
all_text = np.hstack([data['title_description'].str.lower()])
tokenizer.fit_on_texts(all_text)
del all_text
gc.collect()

#data['seq_title'] = tokenizer.texts_to_sequences(data.title.str.lower())
#data['seq_description'] = tokenizer.texts_to_sequences(data.description.str.lower())
data['seq_title_description']= tokenizer.texts_to_sequences(data.title_description.str.lower())
#del data['title']
#del data['description']
del data['title_description']
gc.collect()

le_region = LabelEncoder()
le_region.fit(data.region)

le_city = LabelEncoder()
le_city.fit(data.city)
    
le_category_name = LabelEncoder()
le_category_name.fit(data.category_name)
    
le_parent_category_name = LabelEncoder()
le_parent_category_name.fit(data.parent_category_name)
    
le_param_1 = LabelEncoder()
le_param_1.fit(data.param_1)
    
le_param123 = LabelEncoder()
le_param123.fit(data.param123)

data['region'] = le_region.transform(data['region'])
data['city'] = le_city.transform(data['city'])
data['category_name'] = le_category_name.transform(data['category_name'])
data['parent_category_name'] = le_parent_category_name.transform(data['parent_category_name'])
data['param_1'] = le_param_1.transform(data['param_1'])
data['param123'] = le_param123.transform(data['param123'])



"""
drop_cols = ['item_id', 'user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 
            'param_3', 'title', 'description', 'price', 'item_seq_number', 'activation_date', 'user_type', 'image', 
             'image_top_1', 'deal_class', 'deal_class_2', 'item_label']
data.drop(drop_cols, axis = 1, inplace = True)

drop_cols = []

for column in data:
    if(data[column].dtype == object):
        drop_cols.append(column)

data.drop(drop_cols, axis = 1, inplace = True)

train = data

#scaler = StandardScaler()
#scaler.fit(train)
X = train.iloc[:, 1:]
Y = train.iloc[:, 0]

from sklearn.preprocessing import StandardScaler

for column in X:
    if(X[column].max()>100000):
        X[column].fillna(value=0.5, inplace=True)
        scaler = StandardScaler()
        np_arr = X.as_matrix(columns = [column])
        np_arr = np.reshape(np_arr, (-1, 1))
        X[column] = scaler.fit_transform(np_arr)

for column in X:
    if(X[column].dtype == float or X[column].dtype == int):
        X[column].fillna(value=0.5, inplace=True)
        X[column] = X[column].abs()
        scaler = StandardScaler()
        np_arr = X.as_matrix(columns = [column])
        np_arr = np.reshape(np_arr, (-1, 1))
        X[column] = scaler.fit_transform(np_arr)

"""

price_cols = [
            'price',
            'category_name_mean_price',
            'category_name_min_price',
            'category_name_max_price',
            'region_mean_price',
            'parent_category_name_mean_price',
            'parent_category_name_min_price',
            'parent_category_name_max_price',
            'user_id_mean_price',
            'user_id_min_price',
            'user_id_max_price',
            'image_top_1_mean_price',
            'user_type_mean_price',
            'item_seq_number_mean_price',
            'activation_md_mean_price',
            'activation_wd_mean_price'
]

for column in price_cols:
    data[column].fillna(0.5)
    data[column] = np.log1p(data[column])

data.rename(columns = {'birght_max': 'bright_max'}, inplace = True)
data.info()
# EMBEDDINGS COMBINATION 
# FASTTEXT

EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+2
EMBEDDING_DIM1 = 300# this is from the pretrained vectors
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
print(embedding_matrix1.shape)
# Creating Embedding matrix 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in embeddings_index1:
        c +=1
        embedding_vector = embeddings_index1[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix1[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix1.shape)
del embeddings_index1
gc.collect()

print(" FAST TEXT DONE")

print(vocab_size)
data.info()
Y = data['deal_probability']
X = data.copy(deep = True)

del X['deal_probability']
gc.collect()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.01)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def rmse(y, y_pred):
    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 
"""

DF = pd.DataFrame()

for column in data:
    DF[column] = np.array(data[column])
    
"""

def get_dict(DF):

    data_dict = {}

    columns = list(DF.columns.values)
    
    text_col = ['seq_title_description', 'seq_title', 'seq_description']
    
    for column in columns:
        if(column in text_col):
            data_dict.update({column: pad_sequences(DF[column], maxlen=max_seq_title_description_length)})
        else:
            data_dict.update({column: np.array(DF[column])})
    
    return data_dict
#Inputs
seq_title_description = Input(shape=[100], name="seq_title_description")
#seq_title = Input(shape=[100], name="seq_title")
#seq_description = Input(shape=[100], name="seq_description")


region = Input(shape=[1], name="region")
city = Input(shape=[1], name="city")
category_name = Input(shape=[1], name="category_name")
parent_category_name = Input(shape=[1], name="parent_category_name")
param_1 = Input(shape=[1], name="param_1")
param123 = Input(shape=[1], name="param123")

image_top_1 = Input(shape=[1], name="image_top_1")
item_seq_number = Input(shape = [1], name = 'item_seq_number')


avg_days_up_user = Input(shape=[1], name="avg_days_up_user")
avg_times_up_user = Input(shape=[1], name="avg_times_up_user")
n_user_items = Input(shape=[1], name="n_user_items")


size = Input(shape=[1], name='size')
width = Input(shape=[1], name = 'width')
height = Input(shape=[1], name = 'height')
red_avg = Input(shape=[1], name = 'red_avg')
green_avg = Input(shape=[1], name='green_avg')
blue_avg = Input(shape=[1], name='blue_avg')
width_height_diff = Input(shape=[1], name='width_height_diff')
green_blue_diff = Input(shape=[1], name='green_blue_diff')
green_red_diff = Input(shape=[1], name='green_red_diff')
red_blue_diff = Input(shape=[1], name='red_blue_diff')
width_height_ratio = Input(shape=[1], name='width_height_ratio')
total_pixel = Input(shape=[1], name='total_pixel')
bright_avg = Input(shape=[1], name='bright_avg')
u_avg = Input(shape=[1], name='u_avg')
yuv_v_avg = Input(shape=[1], name='yuv_v_avg')
bright_std = Input(shape=[1], name='bright_std')
bright_diff = Input(shape=[1], name='bright_diff')
bright_min = Input(shape=[1], name='bright_min')
bright_max = Input(shape=[1], name='bright_max')
hue_avg = Input(shape=[1], name='hue_avg')
sst_avg = Input(shape=[1], name='sst_avg')
hsv_v_avg = Input(shape=[1], name='hsv_v_avg')
sat_std = Input(shape=[1], name='sat_std')
sat_diff = Input(shape=[1], name='sat_diff')
sat_min = Input(shape=[1], name='sat_min')
sat_max = Input(shape=[1], name='sat_max')
colorfull = Input(shape=[1], name='colorfull')
r_std = Input(shape=[1], name='r_std')
r_md = Input(shape=[1], name='r_md')
g_std = Input(shape=[1], name='g_std')
g_md = Input(shape=[1], name='g_md')
b_std = Input(shape=[1], name='b_std')
b_md = Input(shape=[1], name='b_md')
xception_prob = Input(shape=[1], name='xception_prob')
xception_var = Input(shape=[1], name='xception_var')
xception_nonzero = Input(shape=[1], name='xception_nonzero')
mean_nima = Input(shape=[1], name='mean_nima')
std_nima = Input(shape=[1], name='std_nima')
blurr = Input(shape=[1], name='blurr')
too_bright = Input(shape=[1], name='too_bright')
too_dark = Input(shape=[1], name='too_dark')

price = Input(shape=[1], name="price")
category_name_mean_price = Input(shape=[1], name='category_name_mean_price')
category_name_min_price = Input(shape=[1], name='category_name_min_price')
category_name_max_price = Input(shape=[1], name='category_name_max_price')
region_mean_price = Input(shape=[1], name='region_mean_price')
parent_category_name_mean_price = Input(shape=[1], name='parent_category_name_mean_price')
parent_category_name_min_price = Input(shape=[1], name='parent_category_name_min_price')
parent_category_name_max_price = Input(shape=[1], name='parent_category_name_max_price')
user_id_mean_price = Input(shape=[1], name='user_id_mean_price')
user_id_min_price = Input(shape=[1], name='user_id_min_price')
user_id_max_price = Input(shape=[1], name='user_id_max_price')
image_top_1_mean_price = Input(shape=[1], name='image_top_1_mean_price')
user_type_mean_price = Input(shape=[1], name='user_type_mean_price')
item_seq_number_mean_price = Input(shape=[1], name='item_seq_number_mean_price')
activation_md_mean_price = Input(shape=[1], name='activation_md_mean_price')
activation_wd_mean_price = Input(shape=[1], name='activation_wd_mean_price')

region_ind = Input(shape=[1], name='region_ind') 
city_ind = Input(shape=[1], name='city_ind') 
parent_category_name_ind = Input(shape=[1], name='parent_category_name_ind') 
category_name_ind = Input(shape=[1], name='category_name_ind') 
user_type_ind = Input(shape=[1], name='user_type_ind') 
param_1_ind = Input(shape=[1], name='param_1_ind') 
param_2_ind = Input(shape=[1], name='param_2_ind') 
param_3_ind = Input(shape=[1], name='param_3_ind') 

#Embeddings layers
emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
#emb_seq_title = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title)
#emb_seq_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_description)


emb_region = Embedding(vocab_size, 10)(region)
emb_city = Embedding(vocab_size, 10)(city)
emb_category_name = Embedding(vocab_size, 10)(category_name)
emb_parent_category_name = Embedding(vocab_size, 10)(parent_category_name)
emb_param_1 = Embedding(vocab_size, 10)(param_1)
emb_param123 = Embedding(vocab_size, 10)(param123)


rnn_layer1 = LSTM(100) (emb_seq_title_description)
#rnn_layer2 = GRU(50) (emb_seq_title)
#rnn_layer3 = GRU(50) (emb_seq_description)

#rnn_region = GRU(50) (emb_region)
#rnn_city = GRU(50) (emb_city)
#rnn_category_name = GRU(50) (emb_category_name)
#rnn_parent_category_name = GRU(50) (emb_parent_category_name)
#rnn_param_1 = GRU(50) (emb_param_1)
#rnn_param123 = GRU(50) (emb_param123)

cat_emb = concatenate([Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_param_1)
        , Flatten() (emb_param123)
                        ])
                        
cat_emb = Dropout(0.2)(Dense(1024,activation='relu') (cat_emb))
cat_emb = BatchNormalization() (cat_emb)
cat_emb = Dropout(0.1)(Dense(512, activation='relu') (cat_emb))
cat_emb = BatchNormalization() (cat_emb)
cat_emb = Dense(64, activation='relu') (cat_emb)
cat_emb = Dense(32, activation='relu') (cat_emb)
cat_emb = Dense(1, activation='sigmoid') (cat_emb)
    
    
img_layer = concatenate([
            image_top_1,
            size,
            width,
            height,
            red_avg,
            green_avg,
            blue_avg,
            width_height_diff,
            green_blue_diff,
            green_red_diff,
            red_blue_diff,
            width_height_ratio,
            total_pixel,
            bright_avg,
            u_avg,
            yuv_v_avg,
            bright_std,
            bright_diff,
            bright_min,
            bright_max,
            hue_avg,
            sst_avg,
            hsv_v_avg,
            sat_std,
            sat_diff,
            sat_min,
            sat_max,
            colorfull,
            r_std,
            r_md,
            g_std,
            g_md,
            b_std,
            b_md,
            xception_prob,
            xception_var,
            xception_nonzero,
            mean_nima,
            std_nima,
            blurr,
            too_bright,
            too_dark
            ])

img_layer = Dropout(0.2)(Dense(1024,activation='relu') (img_layer))
img_layer = BatchNormalization() (img_layer)
img_layer = Dropout(0.1)(Dense(512, activation='relu') (img_layer))
img_layer = BatchNormalization() (img_layer)
img_layer = Dense(64, activation='relu') (img_layer)
img_layer = Dense(32, activation='relu') (img_layer)
img_layer = Dense(1, activation='sigmoid') (img_layer)

price_layer = concatenate([
            price,
            category_name_mean_price,
            category_name_min_price,
            category_name_max_price,
            region_mean_price,
            parent_category_name_mean_price,
            parent_category_name_min_price,
            parent_category_name_max_price,
            user_id_mean_price,
            user_id_min_price,
            user_id_max_price,
            image_top_1_mean_price,
            user_type_mean_price,
            item_seq_number_mean_price,
            activation_md_mean_price,
            activation_wd_mean_price
])

price_layer = Dropout(0.2)(Dense(1024,activation='relu') (price_layer))
price_layer = BatchNormalization() (price_layer)
price_layer = Dropout(0.1)(Dense(512, activation='relu') (price_layer))
price_layer = BatchNormalization() (price_layer)
price_layer = Dense(64, activation='relu') (price_layer)
price_layer = Dense(32, activation='relu') (price_layer)
price_layer = Dense(1, activation='sigmoid') (price_layer)

label_layer = concatenate([
        region_ind,
        city_ind,
        parent_category_name_ind,
        category_name_ind,
        user_type_ind,
        param_1_ind,
        param_2_ind,
        param_3_ind
])

label_layer = Dropout(0.2)(Dense(1024,activation='relu') (label_layer))
label_layer = BatchNormalization() (label_layer)
label_layer = Dropout(0.1)(Dense(512, activation='relu') (label_layer))
label_layer = BatchNormalization() (label_layer)
label_layer = Dense(64, activation='relu') (label_layer)
label_layer = Dense(32, activation='relu') (label_layer)
label_layer = Dense(1, activation='sigmoid') (label_layer)
               
extra_layer = concatenate([
    item_seq_number,
    avg_days_up_user,
    avg_times_up_user,
    n_user_items
])

extra_layer = Dropout(0.2)(Dense(1024,activation='relu') (extra_layer))
extra_layer = BatchNormalization() (extra_layer)
extra_layer = Dropout(0.1)(Dense(512, activation='relu') (extra_layer))
extra_layer = BatchNormalization() (extra_layer)
extra_layer = Dense(64, activation='relu') (extra_layer)
extra_layer = Dense(32, activation='relu') (extra_layer)
extra_layer = Dense(1, activation='sigmoid') (extra_layer)
               
main_layer = concatenate([
    rnn_layer1,
    #rnn_layer2,
    #rnn_layer3,
    #rnn_region,
    #rnn_city,
    #rnn_category_name,
    #rnn_parent_category_name,
    #rnn_param_1,
    #rnn_param123,
    cat_emb,
    img_layer,
    price_layer,
    label_layer,
    extra_layer
])

main_layer = Dropout(0.2)(Dense(1024,activation='relu') (main_layer))
main_layer = BatchNormalization() (main_layer)
main_layer = Dropout(0.1)(Dense(512,activation='relu') (main_layer))
main_layer = BatchNormalization() (main_layer)
main_layer = Dropout(0.1)(Dense(64,activation='relu') (main_layer))
        
#output
output = Dense(1, activation="sigmoid") (main_layer)

model = Model([
            image_top_1,
            size,
            width,
            height,
            red_avg,
            green_avg,
            blue_avg,
            width_height_diff,
            green_blue_diff,
            green_red_diff,
            red_blue_diff,
            width_height_ratio,
            total_pixel,
            bright_avg,
            u_avg,
            yuv_v_avg,
            bright_std,
            bright_diff,
            bright_min,
            bright_max,
            hue_avg,
            sst_avg,
            hsv_v_avg,
            sat_std,
            sat_diff,
            sat_min,
            sat_max,
            colorfull,
            r_std,
            r_md,
            g_std,
            g_md,
            b_std,
            b_md,
            xception_prob,
            xception_var,
            xception_nonzero,
            mean_nima,
            std_nima,
            blurr,
            too_bright,
            too_dark,
            price,
            category_name_mean_price,
            category_name_min_price,
            category_name_max_price,
            region_mean_price,
            parent_category_name_mean_price,
            parent_category_name_min_price,
            parent_category_name_max_price,
            user_id_mean_price,
            user_id_min_price,
            user_id_max_price,
            image_top_1_mean_price,
            user_type_mean_price,
            item_seq_number_mean_price,
            activation_md_mean_price,
            activation_wd_mean_price,
            region_ind,
            city_ind,
            parent_category_name_ind,
            category_name_ind,
            user_type_ind,
            param_1_ind,
            param_2_ind,
            param_3_ind,
            item_seq_number,
            avg_days_up_user,
            avg_times_up_user,
            n_user_items,
            region,
            city,
            category_name,
            parent_category_name,
            param_1,
            param123,
            seq_title_description,
            #seq_title,
            #seq_description
], output)

model.compile(optimizer = 'rmsprop',
            #loss= root_mean_squared_error,
            loss = 'binary_crossentropy',
            metrics = [root_mean_squared_error])

model.fit(get_dict(X_train), np.array(Y_train), batch_size = 512*3, validation_split = 0.40, epochs = 1)
vals_preds = model.predict(get_dict(X_test))
Y_pred = vals_preds[:, 0]
rmse_score = rmse(np.array(Y_test), np.array(Y_pred))
print(rmse_score)