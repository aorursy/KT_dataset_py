#coding=utf-8
import os
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy import sparse
import numpy as np
import random
from sklearn import preprocessing
import tensorflow as tf
import time
import gc
import re
import matplotlib.pyplot as plt
from itertools import product
test_rate=1
test_lu=500
NOISE=0.3
LEARNING_RATE_BASE=0.0001
LEARING_RATE_DECAY=0.1

REGULARAZTION_RATE=0.0005
ALPHA=0.01
BETA=0.8
BATCH=2048
epoches=10
TOP_K=20
def numeric_to_one_hot(arr_of_numbers, num_bins=200):
    
    x = arr_of_numbers
    minx = np.min(x)
    maxx = np.max(x)
    step = (maxx - minx)/num_bins
    bins = np.arange(minx+step, maxx, step, dtype=float) #create num_bins-1 thresholds
    bin_indxs = np.digitize(x, bins)#返回所在bin的下标
    one_hots = preprocessing.OneHotEncoder(sparse=True).fit_transform(bin_indxs.reshape(-1,1))
    return one_hots
def string_to_one_hot(arr_of_strings,sparse):
    """
    Turn categorical string variable into one-hot.
    """

    X_int = preprocessing.LabelEncoder().fit_transform(arr_of_strings.astype(str))
    one_hots = preprocessing.OneHotEncoder(sparse=sparse).fit_transform(X_int.reshape(-1,1))
    return one_hots
def data_process():
    donations = pd.read_csv('../input/Donations.csv',usecols=[0,2],low_memory=False,iterator=True)
    donors = pd.read_csv('../input/Donors.csv',low_memory=False)
    projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,low_memory=False)
    num_users = len(donors)
    num_items = len(projects)
    num_ratings = 4687884
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
    for j in range(chunck_num):        
        if (j+1)*chunck_size>num_ratings:
            chunck_size=num_ratings-j*chunck_size
        datas=donations.get_chunk(chunck_size).values    
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
        
    print('processing the positive data...')
    total_size=len(rating_list)
    remain_rating_list = list(rating_list - set(train))
    random.shuffle(remain_rating_list)
    num_addition = int((1 - ratio) * total_size - len(train))
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
    print(num_addition)
    print(len(remain_rating_list))
    #print(remain_rating_list[:num_addition])
    #print(remain_rating_list[num_addition:])   
    for user,item in test:
        if user not in test_positive.keys():
            test_positive[user]=[item]
            
        else:
            test_positive[user].append(item)
        mask[user,item]=1
    print('processing the test negative data...')
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
        seed=np.random.randint(2000)
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
    print('processing the train negative data...')
    time1=time.time()
    remain_negative=list(set(zip(range(num_users),range(num_items)))-set(train)-set(test))
    rows,cols=zip(*remain_negative)
    train_user_input.extend(rows)
    train_item_input.extend(cols)
    train_ratings.extend([0]*len(remain_negative))
    time2=time.time()
    print('processing the train negative data time: %f'%(time2-time1))
    random.seed(2018)
    random.shuffle(train_user_input)
    random.seed(2018)
    random.shuffle(train_item_input)
    random.seed(2018)
    random.shuffle(train_ratings)
    print('processing data ends,number of user is %d,number of items is %d'%(num_users,num_items))
    
    #处理用户side information
    print('processing user side information...')
    city=np.array(donors['Donor City'].values)       
    city_one_hot=string_to_one_hot(city,True)  
    is_teacher=np.zeros(shape=(num_users,2),dtype=float)
    tmp=np.array(donors['Donor Is Teacher'].values)
    for i in range(num_users):
        if tmp[i]=='Yes':
            is_teacher[i,1]=1
        elif tmp[i]=='No':
            is_teacher[i,0]=1
    all_user_info_features=sparse.hstack((city_one_hot,is_teacher),'csr')
    #处理item的side information
    #Project Subject Category Tree,Project Subject Subcategory Tree,
    project_type=np.array(projects['Project Type'].values)
    project_type_one_hot=string_to_one_hot(project_type,True)
    project_resource=np.array(projects['Project Resource Category'].values)
    project_resource_one_hot=string_to_one_hot(project_resource,True)
    project_current=np.array(projects['Project Current Status'].values)
    project_current_one_hot=string_to_one_hot(project_current,True)
    project_grade=np.array(projects['Project Grade Level Category'].values)
    project_grade_one_hot=string_to_one_hot(project_grade,True)

    project_subject=np.array(projects['Project Subject Category Tree'].values)
    category=[]
    project_subject_list=[]
    for su in project_subject:
        lis=re.split('[&, ]',str(su))
        category.extend(lis)
        project_subject_list.append(lis)
    temp=list(set(category))
    l=len(temp)
    category=np.array(temp)
    project_subject_one_hot=string_to_one_hot(category,False)
    project_subject_one_hot_dict=dict(zip(category, project_subject_one_hot))
    length=project_subject_one_hot.shape[1]
    project_subject_one_hot_result=[]
    for su in project_subject_list:
        vec=np.zeros(length,dtype=np.float32)
        for li in su:
            vec+=project_subject_one_hot_dict[li]
        project_subject_one_hot_result.append(vec)
    project_subject_one_hot_result=csr_matrix(project_subject_one_hot_result)
    
    project_cost=np.array(projects['Project Cost'].values)
    project_cost_one_hot=numeric_to_one_hot(project_cost,num_bins=200)
    all_item_info_features=sparse.hstack((project_type_one_hot,project_resource_one_hot,project_current_one_hot,project_grade_one_hot,project_subject_one_hot_result,project_cost_one_hot),'csr')
    print('processing data ends,number of user is %d,number of items is %d'%(num_users,num_items))
    print('length of user side information is %d,length of item side information is %d'%(all_user_info_features.shape[1],all_item_info_features.shape[1]))
    
    return num_users,num_items,train_user_input,train_item_input,train_ratings,test_user_input,test_item_input,test_ratings,all_user_info_features,all_item_info_features

def get_variables(size,regularizer):
    weights=tf.get_variable('weight',size,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases=tf.get_variable('biases',size[-1],
            initializer=tf.truncated_normal_initializer(stddev=0.0))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
        tf.add_to_collection('losses',regularizer(biases))
    return weights,biases
def reference(X,Y,s_u,s_v,user_info_num,item_info_num,user_num,item_num,train,regularizer):
    #corrupt_s_u=tf.layers.dropout(s_u,rate=NOISE,training=train)
    #corrupt_s_v=tf.layers.dropout(s_v,rate=NOISE,training=train)
    #user part
    with tf.variable_scope('user_information_layer1'):
        #encoding
        weights,biases=get_variables([user_info_num,50],regularizer)
        user_layer1_tmp=tf.add(tf.matmul(s_u,weights),biases)
        user_layer1=tf.nn.relu(user_layer1_tmp) 
    with tf.variable_scope('item_information_layer1'):
        #encoding
        weights,biases=get_variables([item_info_num,50],regularizer)
        item_layer1_tmp=tf.add(tf.matmul(s_v,weights),biases)
        item_layer1=tf.nn.relu(item_layer1_tmp)
    
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
    user_feature=user_layer1
    item_feature=item_layer1
    sdae_out=tf.sigmoid(tf.reduce_sum(tf.multiply(user_feature, item_feature),axis=1))
    gmf_output=tf.multiply(emebdding_user_out, emebdding_item_out)#50
    embedding_out=tf.concat([emebdding_user_out, emebdding_item_out],1)
    with tf.variable_scope('mlp_layer1'):
        mlp1_weights=tf.get_variable('mlp1_weights',[100,50],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        mlp1_biase=tf.get_variable('mlp1_biase',[50],
            initializer=tf.truncated_normal_initializer(stddev=0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(mlp1_weights))
            tf.add_to_collection('losses',regularizer(mlp1_biase))
    mlp1_output=tf.nn.relu(tf.add(tf.matmul(embedding_out,mlp1_weights),mlp1_biase))
    
    with tf.variable_scope('h'):
        h=tf.get_variable(name='h', shape=[100],
        initializer=tf.truncated_normal_initializer(stddev=0.01)) 
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(h))
    pre_out_tmp=tf.reduce_sum(tf.multiply(tf.concat([gmf_output,mlp1_output],1), h),axis=1)
    prediction=0.5*tf.sigmoid(pre_out_tmp)+0.5*sdae_out
    return prediction
def train():
    num_users,num_items,train_user_input,train_item_input,train_ratings,test_user_input,test_item_input,test_ratings,all_user_info_features,all_item_info_features=data_process()
    num_ratings=len(train_user_input)
    test_epoch=int(len(test_user_input)/test_lu)
    user_info_num=all_user_info_features.shape[1]
    item_info_num=all_item_info_features.shape[1]
    X=tf.placeholder(tf.int32)
    Y=tf.placeholder(tf.int32)
    s_u=tf.placeholder(tf.float32,[None,user_info_num],name='user_si')
    s_v=tf.placeholder(tf.float32,[None,item_info_num],name='item_si')
    rating=tf.placeholder(tf.float32,[None])
    regulation_rate=0.0000001
    regularizer=tf.contrib.layers.l2_regularizer(regulation_rate)
    prediction=reference(X,Y,s_u,s_v,user_info_num,item_info_num,num_users,num_items,True,regularizer)
    
    global_step=tf.Variable(0,trainable=False)
    #predict loss
    pred_loss=-1*tf.reduce_mean(rating*tf.log(prediction)+(1-rating)*tf.log(1-prediction))
    
    loss=tf.add_n(tf.get_collection('losses'))+pred_loss
    #loss=pred_loss
    batch_num=int(num_ratings/BATCH)+1
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,batch_num,
        LEARING_RATE_DECAY)
    learning_rate = tf.cond(tf.less(learning_rate,0.000001), lambda: 0.0000010, lambda: learning_rate)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    rank_output=tf.nn.top_k(prediction,TOP_K).indices
    accuracy_y=[]
    with tf.Session() as sess:
    
        tf.global_variables_initializer().run()
        for i in range (epoches): 
            #test...
            total_num=0            
            for k in range(test_epoch):
                X_feed=test_user_input[k]
                Y_feed=test_item_input[k]
                rating_feed=test_ratings[k]
                s_u_feed=all_user_info_features[X_feed].toarray()
                s_v_feed=all_item_info_features[Y_feed].toarray()
                pre,rank,r=sess.run([prediction,rank_output,rating],feed_dict={X:X_feed,Y:Y_feed,rating:rating_feed,s_u:s_u_feed,s_v:s_v_feed})
                index=np.argwhere(r==1)
                tmp=0
                for ind in index:
                    if ind in rank:
                        tmp=1
                        break
                total_num+=tmp
                del s_u_feed
                del s_v_feed
                gc.collect()
            hits=float(total_num)/test_epoch
            accuracy_y.append(hits)
            print('After %d epoches,hits on test data is %f'%(i,hits))
            for j in range(batch_num):
                time1=time.time()
                start=j*BATCH
                end=min(j*BATCH+BATCH,num_ratings)
                X_feed=train_user_input[start:end]
                Y_feed=train_item_input[start:end]
                #user_rating,item_rating
                s_u_feed=all_user_info_features[X_feed].toarray()
                s_v_feed=all_item_info_features[Y_feed].toarray()
                rating_feed=train_ratings[start:end]
                time2=time.time()    
                
                _,pre_loss,total_loss,step=sess.run([train_step,pred_loss,loss,global_step],feed_dict={X:X_feed,Y:Y_feed,rating:rating_feed,s_u:s_u_feed,s_v:s_v_feed})
                time3=time.time()
                loss_tmp+=total_loss
                if j%100==0:
                    print('After %d steps,training loss is %f,prediction loss is %f'%(step-1,total_loss,pre_loss))
                del s_u_feed
                del s_v_feed
                gc.collect()
            #saver.save(sess,path,global_step=global_step)
    x1=range(0,epoches) 
    plt.plot(x1, accuracy_y, label='accuracy')
    plt.show()
train()