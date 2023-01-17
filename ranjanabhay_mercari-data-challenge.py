# Based on Bojan: https://www.kaggle.com/tunguz/wordbatch-ftrl-fm-lgb-lbl-0-42506



import re

import gc

import time

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix,hstack

from time import gmtime,strftime

import wordbatch

from wordbatch.extractors import WordBag,WordHash

from wordbatch.models import FTRL,FM_FTRL

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix,hstack

import lightgbm as lgb





# Function to handle the missing data.

def handle_missing_data(comb):

    comb['first_category'].fillna(value='missing_category',inplace=True)

    comb['second_category'].fillna(value='missing_category',inplace=True)

    comb['third_category'].fillna(value='missing_category',inplace=True)

    comb['item_description'].fillna(value='missing_description',inplace=True)

    comb['brand_name'].fillna(value='missing_brand',inplace=True)

    comb['name'].fillna(value='missing_name',inplace=True)

    #return comb



# Function to split the category into sub-categories

def custom_split(description):

    try:

        return description.split("/")

    except:

        return ["No Label","No Label","No Label"]



# Function to change the data type of the categorical variable.

def convert_to_categorical(comb):

    comb['first_category'] = comb['first_category'].astype('category')

    comb['second_category'] = comb['second_category'].astype('category')

    comb['third_category'] = comb['third_category'].astype('category')

    comb['item_condition_id'] = comb['item_condition_id'].astype('category')



    

# Function to filter the dataset to feed to the model.

def filtering_dataset(comb):

    popular_brand = comb['brand_name'].value_counts().loc[lambda x: x.index !='missing_brand'].index[:4500]

    comb.loc[~comb['brand_name'].isin(popular_brand),'brand_name'] = 'missing'

    popular_first_category = comb['first_category'].value_counts().loc[lambda x: x.index!='missing_category'].index[0:1250]

    comb.loc[~comb['first_category'].isin(popular_first_category),'first_category'] ='missing'

    popular_second_category = comb['second_category'].value_counts().loc[lambda x: x.index!='missing_category'].index[0:1250]

    comb.loc[~comb['second_category'].isin(popular_second_category),'third_category'] = 'missing'

    popular_third_category = comb['third_category'].value_counts().loc[lambda x: x.index!='missing_category'].index[0:1250]

    comb.loc[~comb['third_category'].isin(popular_third_category),'third_category'] = 'missing'



stopwords = {x:1 for x in stopwords.words('english')}

non_alphanums = re.compile(u'[^A-Za-z0-9]+')

# def normalize_text(text):

#     return u" ".join([x for x in [y for y in non_aplhanums.sub(' ',text).lower().strip().split(" ")]\

#                      if len(x) > 1 and x not in stopwords])



# Function to normalize the text

def normalize_text(text):

    return u" ".join(

        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \

         if len(x) > 1 and x not in stopwords])



# Function to compute the Root Mean Squared Logarithmic Error.

def rmsle(y_act,y_pred):

    assert len(y_act) == len(y_pred)

    return np.sqrt(np.mean(np.power(np.log1p(y_act) - np.log1p(y_pred),2)))





# Function for implementing the business logic.

def main_function():

    start_time = time.time()

    print(start_time)

    print(strftime("%Y-%m-%d %H:%M:%S",gmtime()))

    mercari_data_train = pd.read_table('../input/train.tsv',engine='c')

    mercari_data_test = pd.read_table('../input/test.tsv',engine='c')

    print('[{}] Finished to load the data'.format(time.time() - start_time))

    print('Shape of training Data',mercari_data_train.shape)

    print('Shape of testing Data',mercari_data_test.shape)

    nrow_test = mercari_data_train.shape[0]

    dftt = mercari_data_train[(mercari_data_train.price < 1.0)]

    mercari_data_train = mercari_data_train.drop(mercari_data_train[(mercari_data_train.price < 1.0)].index)

    del dftt['price']

    nrow_train = mercari_data_train.shape[0]

    y = np.log1p(mercari_data_train["price"])

    print('Fields of training dataset',mercari_data_train.columns)

    print('Fields of testing dataset',mercari_data_test.columns) 

    comb: pd.DataFrame = pd.concat([mercari_data_train,dftt,mercari_data_test])

    print('Shape of training data:',comb.shape)

    submission:pd.Dataframe = mercari_data_test[['test_id']]

    #comb = comb[comb['category_name'].notnull()]

    comb['first_category'],comb['second_category'],comb['third_category'] = zip(*comb['category_name'].apply(lambda x: custom_split(x)))

    handle_missing_data(comb)

    comb.drop(['category_name'],axis=1,inplace=True)

    print('[{}] Split categories complete and original dropped'.format(time.time() - start_time))

    filtering_dataset(comb)

    convert_to_categorical(comb)

    print(comb.dtypes)

    comb = comb[comb['name'].notnull()]

    print(comb['name'].head(n=500))

    word_batch = wordbatch.WordBatch(normalize_text,extractor=(WordBag,{"hash_ngrams":2,"hash_ngrams_weights":[1.5,1.0],"hash_size":2**29,"norm":None,"tf":"binary","idf":None}),procs=4)

    word_batch.dictionary_freeze=True

    #comb.head(n=5)

    #comb.columns.values

    X_name = word_batch.fit_transform(comb['name'])

    X_name = X_name[:,np.where(X_name.getnnz(axis=0) > 1)[0]]

    del(word_batch)

    word_batch = CountVectorizer()

    X_first_category = word_batch.fit_transform(comb['first_category'])

    X_second_category = word_batch.fit_transform(comb['second_category'])

    X_third_category = word_batch.fit_transform(comb['third_category'])

    print('[{}] Count Vectorize categories completed.'.format(time.time() - start_time))

    # Word Batch for item description

    word_batch = wordbatch.WordBatch(normalize_text,extractor=(WordBag,{"hash_ngrams":2,"hash_ngrams_weights":[1.5,1.0],"hash_size":2**29,"norm":"l2","tf":1.0,"idf":None,}),procs=8)

    word_batch.dictionary_freeze=True

    X_description = word_batch.fit_transform(comb['item_description'])

    del(word_batch)

    X_description = X_description[:,np.where(X_description.getnnz(axis=0)>1)[0]]

    print('[{}] Vectorize item_description completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)

    X_brand = lb.fit_transform(comb['brand_name'])

    print('[{}] Label Binarize brand name completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(comb[['item_condition_id','shipping']],sparse=True).values)

    print('[{}] Get dummies on item condition and shipping done.'.format(time.time() - start_time))

    print(X_dummies.shape,X_description.shape,X_brand.shape,X_first_category.shape,X_second_category.shape,X_third_category.shape,X_name)

    sparse_merge = hstack((X_dummies,X_description,X_brand,X_first_category,X_second_category,X_third_category,X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    print(sparse_merge.shape)

    sparse_merge = sparse_merge[:,np.where(sparse_merge.getnnz(axis=0) > 100)[0]]

    X = sparse_merge[:nrow_train]

    X_test = sparse_merge[nrow_test:]

    #print(sparse_merge.head(n=5))

    

    gc.collect()

    train_X,train_y = X,y

    train_X,valid_X,train_y,valid_y = train_test_split(X,y,test_size=0.05,random_state=100)

    model  = FTRL(alpha=0.01,beta=0.1,L1=0.00001,L2=1.0,D=sparse_merge.shape[1],iters=50,inv_link="identity",threads=1)

    model.fit(train_X,train_y)

    print('[{}] Train FTRL completed'.format(time.time() - start_time))

    preds = model.predict(X = valid_X)

    print("FTRL Dev MSLE:",rmsle(np.expm1(valid_y),np.expm1(preds)))

    

    predsF = model.predict(X_test)

    print('[{}] Predict FTRL completed'.format(time.time() - start_time))

    

    model = FM_FTRL(alpha=0.01,beta=0.01,L1=0.00001,L2=0.1,D=sparse_merge.shape[1],alpha_fm=0.01,L2_fm=0.0,init_fm=0.01,D_fm=200,e_noise=0.0001,iters=20,inv_link='identity',threads=4)

    model.fit(train_X,train_y)

    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))

    preds = model.predict(X=valid_X)

    print("FM_FTRL dev RMSLE:",rmsle(np.expm1(valid_y),np.expm1(preds)))

    

    predsFM = model.predict(X_test)

    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

    

    params = {'learning_rate':0.2,'application':'regression','max_depth':4,'num_leaves':15,'verbosity':-1,'metric':'RMSE','data_random_seed':1,'bagging_fraction':0.6,'bagging_freq':5,'feature_fraction':0.65,'nthread':4,'min_data_in_leaf':100,'max_bin':16}

    

    # Remove features with document frequency <=100

    print(sparse_merge.shape)

    sparse_merge = sparse_merge[:,np.where(sparse_merge.getnnz(axis=0) > 100)[0]]

    X = sparse_merge[:nrow_train]

    X_test = sparse_merge[nrow_test:]

    print(sparse_merge.shape)

    train_X,train_y = X,y

    train_X,valid_X,train_y,valid_y = train_test_split(X,y,test_size=0.05,random_state=100)

    d_train = lgb.Dataset(train_X,label=train_y)

    watch_list = [d_train]

    d_valid = lgb.Dataset(valid_X,label=valid_y)

    watch_list = [d_train,d_valid]

    model = lgb.train(params,train_set=d_train,num_boost_round=200,valid_sets=watch_list,early_stopping_rounds=50,verbose_eval=20)

    preds = model.predict(valid_X)

    print("LGB dev RMSLE:",rmsle(np.expm1(valid_y),np.expm1(preds)))

    predsL = model.predict(X_test)

    print('[{}] Predict LGB completed.'.format(time.time() - start_time))

    preds = (predsF*0.18 + predsL*0.27 + predsFM*0.55)

    submission['price'] = np.expm1(preds)

    submission.to_csv("wordbatch_ftrl_fm_lgb.csv",index=False)

    

main_function()



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.