import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.contrib import layers

from tensorflow.python.estimator.inputs import numpy_io

from tensorflow.contrib.learn import *

import pickle as pkl

import sys

import re
#from google.colab import drive

#drive.mount('/content/drive')

path_csv=   "../input/perfumes/reviews_table_with_common_words_waleed.csv" #'/content/drive/My Drive/share area/Perfumes_data/reviews_table_with_common_words_waleed.csv'

# path='/content/drive/My Drive/share area/Perfumes_data/user_data.csv'

# user_cols=[ "product_pid", "customer_id","product_rating"]

# df_raw_ratings = pd.read_csv(path_csv, usecols=user_cols) #Time_stamp not found in the data set

df_raw_ratings = pd.read_csv(path_csv) #Time_stamp not found in the data set

df_raw_ratings = df_raw_ratings.iloc[: 20471] # after this wrong data

df_raw_ratings.head(2)
#Gad's path:

path='../input/perfumes/sephora_products_notes_waleed-3.csv'

item_cols = ['product_pid', 'product_name', 'product_price','popularity']

#the following items('release_date', 'video_release_date', 'imdb_url') were replaced with ('product_price','product_brand','perfume_gender') along with 'popularity'



# Loading only 5 columns

df_raw_items = pd.read_csv(path, usecols=item_cols)

df_raw_items.head()
df_raw_items.drop_duplicates(subset=['product_pid'], keep='first', inplace=True)
df_raw_ratings = df_raw_ratings.dropna(how='any',subset=["product_pid", "customer_id","product_rating"],axis=0)

df_raw_items = df_raw_items.dropna(how='any',subset=["product_pid"],axis=0)
pid_keys=np.unique(df_raw_items['product_pid'].values)

pid_values=[i for i in range(len(pid_keys))]

pid_dict=dict(list(zip(pid_keys,pid_values)))



cid_keys=np.unique(df_raw_ratings['customer_id'].values)

cid_values=[i for i in range(len(cid_keys))]

cid_dict=dict(list(zip(cid_keys,cid_values)))
max_user_id=len(cid_values)

max_item_id=len(pid_values)
def replace_pid_with_good_one(pid_to_be_replaced):

  return pid_dict[pid_to_be_replaced]



def replace_cid_with_good_one(cid_to_be_replaced):

  return cid_dict[cid_to_be_replaced]
df_raw_ratings['product_pid']=df_raw_ratings['product_pid'].apply(replace_pid_with_good_one)

df_raw_ratings['customer_id']=df_raw_ratings['customer_id'].apply(replace_cid_with_good_one)

df_raw_items['product_pid']=df_raw_items['product_pid'].apply(replace_pid_with_good_one)
df_raw_ratings.head(2)
df_raw_items.head(5)
cols= [col for col in df_raw_ratings.columns]    #read the columns from the csv

first_word_index=0                         #initalize first french word index to zero



for ind,word in enumerate(cols):

  first_word_index+=1 #increment per column (To get the column number for first french word)

  if word=="review_text": #NOTE: THIS DEPENDS ON HAVING "review_text" as the last column before the french words.

    break 

    

word_cols=cols[first_word_index:]
# word_and_itemid=word_cols.copy()

# word_and_itemid.append('product_pid')

# word_and_itemid.append('review_id')



# df_raw_ratings[word_and_itemid]
#For ITEMS:

word_and_itemid=word_cols.copy()

word_and_itemid.append('product_pid')

word_and_itemid.append('review_id')



# review_words_itemid_only=pd.read_csv(path_csv,usecols=word_and_itemid)

review_words_itemid_only = df_raw_ratings[word_and_itemid]

p_id_cumulative_sum=review_words_itemid_only.drop(['review_id'], axis=1).groupby('product_pid').sum()

product_count_matrix=p_id_cumulative_sum.to_numpy()

print(product_count_matrix.shape)

print(f'This means that we have {p_id_cumulative_sum.to_numpy().shape[0]} products, and {p_id_cumulative_sum.to_numpy().shape[1]} vocab words.')



#should be 1303 products and 100 vocab words
review_words_itemid_only.drop(['review_id'], axis=1).groupby('product_pid').sum()
items_list=np.unique(review_words_itemid_only['product_pid'].tolist())

items_list.sort()
#For CUSTOMERS:



word_and_custid=word_cols.copy()

word_and_custid.append('customer_id')

word_and_custid.append('review_id')



# review_words_custid_only=pd.read_csv(path_csv,usecols=word_and_custid).sort_values(by='customer_id')

review_words_custid_only = df_raw_ratings[word_and_custid]

c_id_cumulative_sum=review_words_custid_only.drop(['review_id'], axis=1).groupby('customer_id').sum()

customer_count_matrix=c_id_cumulative_sum.to_numpy()

print(customer_count_matrix.shape)

print(f'This means that we have {customer_count_matrix.shape[0]} customers, and {customer_count_matrix.shape[1]} vocab words.')



#should be 17278 products and 100 vocab words
customers_list=np.unique(review_words_custid_only['customer_id'].tolist())

customers_list.sort()

print(f'There are, infact, {len(customers_list)} customers.')
#ITEMS:

dict_of_counts={}

for key_num,value_list in zip(items_list, product_count_matrix):

  #debug#print(value_list)

  dict_of_counts[key_num]=value_list



#CUSTOMERS:

dict_of_counts_custs={}

for key_num,value_list in zip(customers_list, customer_count_matrix):

  #debug#print(value_list)

  dict_of_counts_custs[key_num]=value_list

  
dict_of_item_embeddings={}

for key_num,value_list in zip(items_list, product_count_matrix):

  #debug#print(value_list)

  counts=value_list

  temp_embdng_list_for_this_key=np.zeros((1,len(word_embeddings_list[0])))

  for count,embdng in zip(counts,word_embeddings_list):

    temp_embdng_list_for_this_key+=count*np.array(embdng,dtype=np.float32)

  dict_of_item_embeddings[key_num]=temp_embdng_list_for_this_key
print(f'Dimensions of Word Embedding are: {len(dict_of_item_embeddings[items_list[0]][0])}') #should be equal to 635

print(f'Size of of the dict of word embeddings for items is : {sys.getsizeof(dict_of_item_embeddings)} bytes.') #equal to 36968 bytes
#ASSERT WE ARE WORKING FINE:

#That all elements in the summed word embeddings list per item have dimensions = 635

for i in range(len(dict_of_item_embeddings.keys())):

  assert len(dict_of_item_embeddings[items_list[i]][0]) == len(word_embeddings_list[0])
dict_of_customer_embeddings={} #key: customer_id #value:summerd embeddings weighted with count

for key_num,value_list in zip(customers_list, customer_count_matrix):

  #debug#print(value_list)

  counts=value_list

  temp_embdng_list_for_this_key=np.zeros((1,len(word_embeddings_list[0])))

  for count,embdng in zip(counts,word_embeddings_list): #count vector per word embedding, and word_embddings list. 

                                                            #they have corresponding order.

    temp_embdng_list_for_this_key+=count*np.array(embdng,dtype=np.float32)

  dict_of_customer_embeddings[key_num]=temp_embdng_list_for_this_key
print(f'Dimensions of Word Embedding are: {len(dict_of_customer_embeddings[customers_list[0]][0])}') #should be equal to 635

import sys

print(f'Size of of the dict of word embeddings for csutomers is : {sys.getsizeof(dict_of_customer_embeddings)}')
#ASSERT WE ARE WORKING FINE:

#That all elements in the summed word embeddings list per customer have dimensions = 635

for i in range(len(dict_of_customer_embeddings.keys())):

  assert len(dict_of_customer_embeddings[customers_list[i]][0]) == len(word_embeddings_list[0])
df_ratings_with_item_info = pd.merge(df_raw_items,df_raw_ratings[['customer_id','product_pid','product_rating']], how='inner', on='product_pid')

df_ratings_with_item_info.head(10)
dict_of_item_embeddings #checking it exists.

_=0
# #Create series from count per review for each item

# pid_list_to_form_embeddings=df_all_ratings['product_pid'].tolist()

# word_embedding_repeated_for_reviews=[]

# for iid in pid_list_to_form_embeddings:

#     word_embedding_repeated_for_reviews.append(list(dict_of_counts[iid])) #dict_of_item_embeddings

    

# columns_for_items_word_embeddings=["i_w_"+str(i) for i in range(len(word_cols))]



#Create series from count per review for each item

pid_list_to_form_embeddings=df_ratings_with_item_info['product_pid'].tolist()

word_embedding_repeated_for_reviews=[]

for iid in pid_list_to_form_embeddings:

    word_embedding_repeated_for_reviews.append(list(dict_of_item_embeddings[iid])[0]) #dict_of_counts

    

columns_for_items_word_embeddings=["i_w_"+str(i) for i in range(635)]
df_ratings_with_item_info_and_embbeddings=df_ratings_with_item_info.join(pd.DataFrame.from_records(word_embedding_repeated_for_reviews, columns=columns_for_items_word_embeddings))
df_ratings_with_item_info_and_embbeddings.head()
dict_of_customer_embeddings #checking it exists

_=0
#Create series from count per review for each item

cid_list_to_form_embeddings=df_ratings_with_item_info_and_embbeddings['customer_id'].tolist()

cid_list_to_form_embeddings=[str(x) for x in cid_list_to_form_embeddings ] ## CONVERT TO STRING TO AVOID KeyError

## CONVERT TO STRING TO AVOID KeyError

## CONVERT TO STRING TO AVOID KeyError

## CONVERT TO STRING TO AVOID KeyError

## CONVERT TO STRING TO AVOID KeyError



customer_word_embedding_repeated_for_reviews=[]

for cid in cid_list_to_form_embeddings:

    customer_word_embedding_repeated_for_reviews.append(list(dict_of_customer_embeddings[int(cid)])[0]) #dict_of_counts

    

columns_for_customers_word_embeddings=["c_w_"+str(i) for i in range(635)]
assert len(customer_word_embedding_repeated_for_reviews)== len(df_ratings_with_item_info_and_embbeddings)
df_ratings_with_item_info_and_embbeddings=df_ratings_with_item_info_and_embbeddings.join(pd.DataFrame.from_records(customer_word_embedding_repeated_for_reviews, columns=columns_for_customers_word_embeddings))
df_ratings_with_item_info_and_embbeddings.info()
df_ratings_with_item_info_and_embbeddings = df_ratings_with_item_info_and_embbeddings.dropna(how='any',axis=0)

df_ratings_with_item_info_and_embbeddings.info()
# Split All ratings into train_val and test

ratings_train_val, ratings_test = train_test_split(df_ratings_with_item_info_and_embbeddings, test_size=0.05, random_state=0)

# Split train_val into training and validation set

ratings_train, ratings_val = train_test_split(ratings_train_val, test_size=0.02, random_state=0)

print('Total rating rows count: {0} '.format(len(df_ratings_with_item_info_and_embbeddings))) #should be 40511

print('Total training rows count: {0} '.format(len(ratings_train_val))) #should be 32408

print('Total validation rows count: {0} '.format(len(ratings_val))) #should be 6482

print('Total test rows count: {0} '.format(len(ratings_test))) #should be 8103

from sklearn.preprocessing import QuantileTransformer



meta_columns = ['popularity', 'product_price']

meta_columns.extend(columns_for_items_word_embeddings)

meta_columns.extend(columns_for_customers_word_embeddings)



scaler = QuantileTransformer()

item_meta_train = scaler.fit_transform(ratings_train[meta_columns]).astype(np.float32)

item_meta_val = scaler.transform(ratings_val[meta_columns])

item_meta_test = scaler.transform(ratings_test[meta_columns])
np.count_nonzero(np.isnan(item_meta_train))
embedding_size = 450 #best: 300 )0.647 #tried: 250-700 with 50 steps #best_c: 450, 0.6216

reg_param = 0.01 #Changed from 0.01 #best: 0.005 #best_c: 0.01, 0.6216

learning_rate = 0.005 #Changed from 0.01 #best_c: 0.005, 0.6216

n_users = max_user_id + 1

n_items = max_item_id + 1

meta_size = 2 + len(columns_for_items_word_embeddings) + len(columns_for_customers_word_embeddings)





g = tf.Graph()

with g.as_default():

    

    tf.set_random_seed(1234)

    

    users = tf.placeholder(shape=[None,1], dtype=tf.int64, name='input_users')

    items = tf.placeholder(shape=[None,1], dtype=tf.int64, name='input_items')

    meta = tf.placeholder(shape=[None,meta_size], dtype=tf.float32, name='input_metadata')

    ratings = tf.placeholder(shape=[None,1], dtype=tf.float32, name='input_ratings')

    is_training = tf.placeholder(tf.bool, None, name='is_training')



    

    l2_loss = tf.constant(0.0)

    

    # embeddding layer

    with tf.variable_scope("embedding"):

        user_weights = tf.get_variable("user_w"

                                      , shape=[n_users, embedding_size]

                                      , dtype=tf.float32

                                      , initializer=layers.xavier_initializer())

        

        item_weights = tf.get_variable("item_w"

                                       , shape=[n_items, embedding_size]

                                       , dtype=tf.float32

                                       , initializer=layers.xavier_initializer())

        

        

        

        user_embedding = tf.squeeze(tf.nn.embedding_lookup(user_weights, users),axis=1, name='user_embedding')

        item_embedding = tf.squeeze(tf.nn.embedding_lookup(item_weights, items),axis=1, name='item_embedding')

        

        l2_loss += tf.nn.l2_loss(user_weights)

        l2_loss += tf.nn.l2_loss(item_weights)

        

        

        print(user_embedding)

        print(item_embedding)

        

    

    # combine inputs

    with tf.name_scope('concatenation'):

        input_vecs = tf.concat([user_embedding, item_embedding, meta], axis=1)

        print(input_vecs)

        

    # fc-1

    num_hidden = 16 #CHANGED FROM 64, best: 32 tried: 8,15,16,28,45,64 #best_c: 16, 0.6216

    with tf.name_scope("fc_1"):

        W_fc_1 = tf.get_variable(

            "W_hidden",

            shape=[2*embedding_size + meta_size, num_hidden],

            initializer=tf.contrib.layers.xavier_initializer())

        b_fc_1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")

        hidden_output = tf.nn.relu(tf.nn.xw_plus_b(input_vecs, W_fc_1, b_fc_1), name='hidden_output')

        l2_loss += tf.nn.l2_loss(W_fc_1)

        print(hidden_output)

    

    # dropout

    with tf.name_scope("dropout"):

        h_drop = tf.nn.dropout(hidden_output, 0.9999, name="hidden_output_drop") #was 0.99 #best_c: 0.9999 #tried: 0.98,0.9,0.99,0.999

        print(h_drop)

    

    # fc-2

    with tf.name_scope("fc_2"):

        W_fc_2 = tf.get_variable(

            "W_output",

            shape=[num_hidden,1],

            initializer=tf.contrib.layers.xavier_initializer())

        b_fc_2 = tf.Variable(tf.constant(0.1, shape=[1]), name="b")

        pred = tf.nn.xw_plus_b(h_drop, W_fc_2, b_fc_2, name='pred')

        l2_loss += tf.nn.l2_loss(W_fc_2)

        print(pred)



    # loss

    with tf.name_scope("loss"):    

        loss = tf.cond(is_training , lambda: tf.nn.l2_loss(pred - ratings) + reg_param * l2_loss, lambda: tf.nn.l2_loss(pred - pred))

#         loss = tf.nn.l2_loss(pred - ratings) + reg_param * l2_loss

        train_ops = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        rmse = tf.cond(is_training , lambda: tf.sqrt(tf.reduce_mean(tf.pow(pred - ratings, 2))), lambda: tf.sqrt(tf.reduce_mean(tf.pow(pred - pred, 2))))

#         rmse = tf.sqrt(tf.reduce_mean(tf.pow(pred - ratings, 2)))



    saver = tf.train.Saver()

        
def train_model_deep_meta():

    



    losses_train = []

    losses_val = []

    epochs = 40 #best_c: 80





    with tf.Session(graph=g) as sess:

        sess.run(tf.global_variables_initializer())

        

        train_input_dict = {users: ratings_train['customer_id'].values.reshape([-1,1])

            , items: ratings_train['product_pid'].values.reshape([-1,1])

            , ratings: ratings_train['product_rating'].values.reshape([-1,1])

                           ,meta: item_meta_train

                           ,is_training : True} #was || meta: item_meta_train || only. no .value.reshape



        val_input_dict = {users: ratings_val['customer_id'].values.reshape([-1,1])

            , items: ratings_val['product_pid'].values.reshape([-1,1])

            , ratings: ratings_val['product_rating'].values.reshape([-1,1])

                         ,meta : item_meta_val

                         ,is_training : True} #was || meta: item_meta_train || only. no .value.reshape



        test_input_dict = {users: ratings_test['customer_id'].values.reshape([-1,1])

            , items: ratings_test['product_pid'].values.reshape([-1,1])

            , ratings: ratings_test['product_rating'].values.reshape([-1,1])

                          ,meta : item_meta_test

                          ,is_training : True} #was || meta: item_meta_train || only. no .value.reshape

        #debug#print(train_input_dict)

        



        for i in range(epochs):

            print

            sess.run([train_ops], feed_dict=train_input_dict)

            if i % 10 == 0:

                loss_train = sess.run(loss, feed_dict=train_input_dict)

                loss_val = sess.run(loss, feed_dict=val_input_dict)

                losses_train.append(loss_train)

                losses_val.append(loss_val)

                print("iteration : %d train loss: %.3f , valid loss %.3f" % (i,loss_train, loss_val))

        

        

        # plot train and validation loss

        plt.plot(losses_train, label='train')

        plt.plot(losses_val, label='validation')

        plt.legend(loc='best')

        plt.title('Loss');

        

         # calculate RMSE on the test dataset

        print('RMSE on test dataset : {0:.4f}'.format(sess.run(rmse, feed_dict=test_input_dict)))

        save_path = tf.train.Saver().save(sess, "/content/drive/My Drive/share area/pfembeddings/model.ckpt")

        print("Model saved in file: %s" % save_path)

train_model_deep_meta()
def pred_rat(userrr,itemmm,mettt):

    #loding the model again and predicting

    with tf.Session(graph=g) as sess:



        saver = tf.train.import_meta_graph('/content/drive/My Drive/share area/pfembeddings/model.ckpt.meta')

        saver.restore(sess,tf.train.latest_checkpoint('/content/drive/My Drive/share area/pfembeddings/'))



        dicttt = {users: userrr.reshape([-1,1])

            , items: itemmm.reshape([-1,1])

                          ,meta : mettt

                          ,is_training : False} #was || meta: item_meta_train || only. no .value.reshape



    #     print('RMSE on test dataset : {0:.4f}'.format(sess.run(rmse, feed_dict=test_input_dict)))



        predicted = sess.run(pred, feed_dict=dicttt)

    return predicted
# print(pred_rat(ratings_test['customer_id'].values,ratings_test['product_pid'].values,item_meta_test))
def get_data_for_model_from_cust_id(old_cust_id,items_list,df_raw_items):

  

  columns_for_items_word_embeddings=["i_w_"+str(i) for i in range(635)]

  columns_for_customer_word_embeddings=["c_w_"+str(i) for i in range(635)]



  this_customer_list=[old_cust_id for x in items_list]

  this_customer_word_embedding_rows=[dict_of_customer_embeddings[old_cust_id] for x in items_list]

  all_items_word_embedding_rows=[dict_of_item_embeddings[x] for x in items_list]



  data_frame_cols=['customer_id','product_pid']

  # data_frame_cols.extend(columns_for_items_word_embeddings)

  # data_frame_cols.extend(columns_for_customer_word_embeddings)







  #datframe with columns customer_id, product_id, item_word_embeddings, customer_word_embeddings:

  this_cust_with_all_items_df=pd.DataFrame(list(zip(this_customer_list,items_list)),columns=data_frame_cols)

  all_items_word_embedding_df=pd.DataFrame(np.squeeze(np.array(all_items_word_embedding_rows)),columns=columns_for_items_word_embeddings)

  this_cust_word_embeddings_df=pd.DataFrame(np.squeeze(np.array(this_customer_word_embedding_rows)),columns=columns_for_customer_word_embeddings)



  our_merged_df=this_cust_with_all_items_df.join([all_items_word_embedding_df,this_cust_word_embeddings_df])





  item_cols = ['product_pid', 'product_name', 'product_price','product_brand','perfume_gender','popularity']

  # Loading only 5 columns

  df_items_info = df_raw_items







  our_final_merged_df = pd.merge(our_merged_df, df_items_info,on='product_pid')





  from sklearn.preprocessing import QuantileTransformer



  meta_columns = ['popularity', 'product_price']

  meta_columns.extend(columns_for_items_word_embeddings)

  meta_columns.extend(columns_for_customers_word_embeddings)



  # scaler = QuantileTransformer()

  item_meta_predict = scaler.transform(our_final_merged_df[meta_columns]).astype(np.float32)

  

  



  return our_final_merged_df['customer_id'].values,our_final_merged_df['product_pid'].values,item_meta_predict

def easy_pred(useroldid):

  inverted_pid_dict = dict([[v,k] for k,v in pid_dict.items()])

  inverted_cid_dict = dict([[v,k] for k,v in cid_dict.items()])

  # print(inverted_cid_dict[useroldid])

  cc,pp,mm = get_data_for_model_from_cust_id(cid_dict[useroldid],items_list,df_raw_items)

  a = pred_rat(cc,pp,mm)



  cco =[inverted_cid_dict[idc] for idc in cc]

  ccp =[inverted_pid_dict[idc] for idc in pp]

  comp_out_df = pd.DataFrame({'customer_id':cco,'product_pid':ccp,'expected_rating':np.squeeze(a)})

  comp_out_df.dropna(inplace=True)

  comp_out_df = comp_out_df.sort_values(by='expected_rating')

  return comp_out_df[-5:]

# easy_pred('71723071')
path='/content/drive/My Drive/share area/Perfumes_data/item_data.xlsx'

item_cols = ['product_pid', 'product_name', 'notes_de_tete','note_de_coeur','note_de_fond']

df_notes = pd.read_excel(path, usecols=item_cols)



df_notes.sort_values("product_pid", inplace = True) 

  

# dropping ALL duplicte values 

df_notes.drop_duplicates(subset ="product_pid", 

                     keep = "first", inplace = True) 
def splitter(notes):

#A regex based function that splits the notes cell into an array of separate notes.

  Notes=[]

  for note in notes:

    note=str(note).strip('.;')

    note_list=re.split(r'et\s*|,\s*',note)

    for n in range(len(note_list)):

      note_list[n]=note_list[n].strip()

      if n=='':

        note_list.remove(note_list[n])

      



    Notes.append(note_list)

  return Notes
def most_frequent(List):

#takes a list and returns the most common element

    return max(set(List), key = List.count) 

def top_finder(PIDs,df_notes,notes,top_note): 

#takes as an input the PIDs of top rated perfumes,a data frame of notes data, a list of all note lists, and the top note if already extracted, empty if not.

#It returns the top component

  top_notes=[]

  pids=[]

  for PID in PIDs:

    idx=pd.Index(df_notes['product_pid']).get_loc(PID)

    if notes[idx][0]=='nan':

      continue

    



    if top_note and top_note[0] in notes[idx]:

      

      notes[idx]=list(filter(lambda a: a !=top_note[0], notes[idx])) #remove the top note from list

      #print(top_note)

      #print(notes[idx])

      top_notes=top_notes+notes[idx]

      pids.append(PID)

    elif not top_note:

      top_notes=top_notes+notes[idx]

      

  top_note=most_frequent(sorted(top_notes)) #If there are no duplicates, it returns the first element

  return top_note,pids

  
def top_comp(notes,top_note,Idxs):

  idxs=[]

  top_notes=[]

  for idx in Idxs:

    if notes[idx][0]=='nan':

      continue

    



    if top_note and top_note[0] in notes[idx]:

      

      notes[idx]=list(filter(lambda a: a !=top_note[0], notes[idx])) #remove the top note from list

      top_notes=top_notes+notes[idx]

      idxs.append(idx)

    elif not top_note:

      top_notes=top_notes+notes[idx]

      

  top_note=most_frequent(sorted(top_notes)) #If there are no duplicates, it returns the first element

  return top_note,idxs
def full_note_finder(PIDs):

    tete=splitter(df_notes['notes_de_tete'])

    coeur=splitter(df_notes['note_de_coeur'])

    fond=splitter(df_notes['note_de_fond'])

    notes=[]

    for i in range(len(tete)):

      note=tete[i]+coeur[i]+fond[i]

      notes.append(note)





    top_note=[]

    try:

      first,dumb=top_finder(PIDs,df_notes,notes,top_note)

      # print('The top single component is',first)

      top_note.append(first)

      second,PIDs=top_finder(PIDs,df_notes,notes,top_note)

      # print('The top double components are',first,second)

      top_note[0]=second

      third,dumb=top_finder(PIDs,df_notes,notes,top_note)

      # print('The top triple components are',first,second,third)

      return first,second,third

    except:

      print('The inputted perfumes have no components! Outptting popular components in all perfumes:')

      Idxs=range(len(notes))

      first,dumb=top_comp(notes,top_note,Idxs)

      # print('The top single component is',first)

      top_note.append(first)

      second,Idxs=top_comp(notes,top_note,Idxs)

      # print('The top double components are',first,second)

      top_note[0]=second

      third,dumb=top_comp(notes,top_note,Idxs)

      # print('The top triple components are',first,second,third)

      return first,second,third



# PIDs=['P1626001','P1626007','P1626008','P1626009','P1626017'] # a PIDs list of perfumes with components

# a,b,c = full_note_finder(PIDs)

# print(a,b,c)
def final_note_predict(useID): 

  """

  userID: string with original user ID (before changing or anything)

  output: predicted notes

  """

  mydf = easy_pred(useID)

  return full_note_finder(mydf['product_pid'].values)
final_note_predict('70837926')