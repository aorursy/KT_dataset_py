import os
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy import sparse
import numpy as np
import random
import tensorflow as tf
import time
import gc
import matplotlib.pyplot as plt
from itertools import product
test_rate=1
test_lu=500
LEARNING_RATE_BASE=0.01
LEARING_RATE_DECAY=0.1
REGULARAZTION_RATE=0.000001
BATCH=32768
epoches=50
TOP_K=10
def data_process():
    donations = pd.read_csv('../input/Donations.csv',usecols=[0,2],low_memory=False)
    donors = pd.read_csv('../input/Donors.csv',low_memory=False)
    projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,low_memory=False)
    num_users = len(donors)
    num_items = len(projects)
    num_ratings = len(donations)
    ratio=0.001
    chunck_size=500
    chunck_num=int(num_ratings/chunck_size)+1
    mask=lil_matrix((num_users, num_items), dtype=np.float32)
    R=lil_matrix((num_users, 1), dtype=np.float32)
    #mask_1=csr_matrix((num_users, num_items), dtype=np.float32)
    train_user_input,train_item_input,train_ratings=[],[],[]
    test_user_input,test_item_input,test_ratings=[],[],[]
    users=donors['Donor ID'].values
    items=projects['Project ID'].values
    user_id=dict(zip(users,range(num_users)))
    item_id=dict(zip(items,range(num_items)))
    row_m,col_m,data_m=[],[],[]
    num_test=0
    rating_list=set()
    #计算一阶原点矩
    o_num=0
    test_positive={}
    train=[]
    #construct rating matrix
    print('constructing the rating matrix...')     
    datas=donations.values
    for data in datas:
        try:
            row=user_id[data[1]]
            col=item_id[data[0]]
        except:
            continue
        if R[row,0]==0:
            train.append((row,col))
            train_user_input.append(row)
            train_item_input.append(col)
            mask[row,col]=1
            o_num+=1
            R[row,0]=1
        rating_list.add((row,col))     
    #print('processing the positive data...')
    total_size=len(rating_list)
    remain_rating_list = list(rating_list - set(train))
    random.shuffle(remain_rating_list)
    num_addition = int((1 - ratio) * total_size - len(train))
    print(total_size)
    print(len(train))
    print(len(set(train)))
    if num_addition < 0:
        print('this ratio cannot be handled')
        sys.exit()
    else:
        rows,cols=zip(*remain_rating_list[:num_addition])
        train.extend(remain_rating_list[:num_addition])
        train_user_input.extend(list(rows))
        train_item_input.extend(list(cols))
        o_num+=num_addition
        mask[list(rows),list(cols)]=1
        test = remain_rating_list[num_addition:]
    
    #print(remain_rating_list[:num_addition])
    #print(remain_rating_list[num_addition:])   
    for user,item in test:
        if user not in test_positive.keys():
            test_positive[user]=[item]          
        else:
            test_positive[user].append(item)
        mask[user,item]=1
    print('test positive number is %d'%len(test_positive.keys()))
    #print('processing the test negative data...')
    for key in test_positive:
        l=len(test_positive[key])
        num=test_lu-l
        test_ratings_tmp=[1]*l
        test_ratings_tmp.extend([0]*num)
        test_user_input_tmp=[key]*l
        test_item_input_tmp=test_positive[key]
        for i in range(num):
            col=np.random.randint(num_items)
            while(mask[key,col]==1):
                col=np.random.randint(num_items)
            test_user_input_tmp.append(key)
            test_item_input_tmp.append(col)
            test.extend([key,col]) 
            mask[key,col]=1
        seed=np.random.randint(20)
        random.seed(seed)
        random.shuffle(test_user_input_tmp)
        random.seed(seed)
        random.shuffle(test_item_input_tmp)
        random.seed(seed)
        random.shuffle(test_ratings_tmp)
        test_user_input.append(test_user_input_tmp)
        test_item_input.append(test_item_input_tmp)
        test_ratings.append(test_ratings_tmp)
    train_ratings.extend([1]*len(train_user_input))
    #print('processing the train negative data...')
    time1=time.time()
    remain_negative=list(set(zip(range(num_users),range(num_items)))-set(train)-set(test))
    rows,cols=zip(*remain_negative)
    train_user_input.extend(rows)
    train_item_input.extend(cols)
    train_ratings.extend([0]*len(remain_negative))
    time2=time.time()
    #print('processing the train negative data time: %f'%(time2-time1))
    random.seed(2018)
    random.shuffle(train_user_input)
    random.seed(2018)
    random.shuffle(train_item_input)
    random.seed(2018)
    random.shuffle(train_ratings)
    print('number of user is %d,number of items is %d'%(num_users,num_items))   
    return o_num,num_users,num_items,train_user_input,train_item_input,train_ratings,test_user_input,test_item_input,test_ratings

def reference(X,Y,user_num,item_num,train,regularizer): 
    with tf.variable_scope('user-embedding'):
        user_embedding=tf.get_variable('user-embedding',[item_num,50],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        emebdding_user_out=tf.nn.embedding_lookup(user_embedding, X)       
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(user_embedding))
    with tf.variable_scope('item-embedding'):
        item_embedding=tf.get_variable('item-embedding',[user_num,50],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        emebdding_item_out=tf.nn.embedding_lookup(item_embedding, Y)
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(item_embedding))

    gmf_output=tf.multiply(emebdding_user_out, emebdding_item_out)
    embedding_out=tf.concat([emebdding_user_out,emebdding_item_out],1)
    with tf.variable_scope('mlp_layer1'):
        mlp1_weights=tf.get_variable('mlp1_weights',[100,80],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        mlp1_biase=tf.get_variable('mlp1_biase',[80],
            initializer=tf.truncated_normal_initializer(stddev=0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(mlp1_weights))
            tf.add_to_collection('losses',regularizer(mlp1_biase))
    mlp1_output=tf.nn.relu(tf.add(tf.matmul(embedding_out,mlp1_weights),mlp1_biase))
    with tf.variable_scope('mlp_layer2'):
        mlp2_weights=tf.get_variable('mlp2_weights',[80,50],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        mlp2_biase=tf.get_variable('mlp2_biase',[50],
            initializer=tf.truncated_normal_initializer(stddev=0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(mlp2_weights))
            tf.add_to_collection('losses',regularizer(mlp2_biase))
    mlp2_output=tf.nn.relu(tf.add(tf.matmul(mlp1_output,mlp2_weights),mlp2_biase))
    with tf.variable_scope('h'):
        h=tf.get_variable(name='h', shape=[100],
        initializer=tf.truncated_normal_initializer(stddev=0.01)) 
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(h))
    pre_out_tmp=tf.reduce_sum(tf.multiply(tf.concat([gmf_output,mlp2_output],1), h),axis=1)
    prediction=tf.sigmoid(pre_out_tmp)
    return prediction
def train():
    o_num,num_users,num_items,train_user_input,train_item_input,train_ratings,test_user_input,test_item_input,test_ratings=data_process()
    num_ratings=len(train_user_input)
    test_epoch=int(len(test_user_input))
    batch_num=int(num_ratings/BATCH)+1
    X=tf.placeholder(tf.int32)
    Y=tf.placeholder(tf.int32)
    rating=tf.placeholder(tf.float32,[None])
    regularizer=None
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    prediction=reference(X,Y,num_users,num_items,True,regularizer)
    #pre_neg=1-prediction
    #result=tf.stack([prediction,pre_neg], axis=1)
    global_step=tf.Variable(0,trainable=False)
    #predict loss
    #pred_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rating,logits=result))
    pred_loss=-1*tf.reduce_mean(rating*tf.log(prediction)+(1-rating)*tf.log(1-prediction))
    #total loss
    loss=tf.add_n(tf.get_collection('losses'))+pred_loss
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,batch_num,LEARING_RATE_DECAY)
    learning_rate = tf.cond(tf.less(learning_rate,0.0000001), lambda: 0.0000001, lambda: learning_rate) 
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    loss_y=[]
    accuracy_y=[]
    #RS defination
    rank_output=tf.nn.top_k(prediction,TOP_K).indices
    with tf.Session() as sess:
    
        tf.global_variables_initializer().run()
        for i in range (epoches): 
            #test...
            total_num=0            
            for k in range(test_epoch):
                X_feed=test_user_input[k]   
                Y_feed=test_item_input[k]
                rating_feed=test_ratings[k]
                pre,rank,r=sess.run([prediction,rank_output,rating],feed_dict={X:X_feed,Y:Y_feed,rating:rating_feed})
                index=np.argwhere(r==1)
                tmp=0
                for ind in index:
                    if ind in rank:
                        tmp=1
                        break
                total_num+=tmp
            hits=float(total_num)/test_epoch
            accuracy_y.append(hits)
            print('After %d epoches,hits on test data is %f'%(i+1,hits))
            loss_tmp=0.0
            for j in range(batch_num):
                start=j*BATCH
                end=min(j*BATCH+BATCH,num_ratings)
                X_feed=train_user_input[start:end]
                Y_feed=train_item_input[start:end]
                #user_rating,item_rating
                rating_feed=train_ratings[start:end]
                _,pre_loss,total_loss,step=sess.run([train_step,pred_loss,loss,global_step],feed_dict={X:X_feed,Y:Y_feed,rating:rating_feed})
                loss_tmp+=total_loss
                if j%100==0:
                    print('After %d steps,training loss is %f,prediction loss is %f'%(step,total_loss,pre_loss))
            loss_tmp= loss_tmp/batch_num
            loss_y.append(loss_tmp)
    x1=range(0,epoches)       
    plt.plot(x1, accuracy_y, label='hits')
    plt.show()
    
train()
   