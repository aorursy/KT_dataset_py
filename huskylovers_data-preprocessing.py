# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


from sklearn.preprocessing import LabelEncoder

import gc
basic_path = "../input/finance/data/data/"

listing_info_path = "listing_info.csv"

test_path = "test.csv"

train_path = "train.csv"

user_behaviors_path = "user_behavior_logs.csv"  #用户的一些操作（3组）

user_repay_logs_path = "user_repay_logs.csv"   #用户过去的还款信息

user_info_path = "user_info.csv"      #用户的信息，年龄等

user_taglist_path = "user_taglist.csv"   #用户的画像信息
data = pd.read_csv(basic_path+train_path)

test = pd.read_csv(basic_path+test_path)
data.head()
test.head()
data['repay_date'] = data[['due_date', 'repay_date']].apply(

    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1

)

data['repay_amt'] = data['repay_amt'].apply(

    lambda x: float(x) if x!= '\\N' else 0)

data['tag'] = data['repay_amt'] == 0

len(data[data['tag']==True])/len(data)  #有部分人未按due_amt还款
data['days'] = pd.to_datetime(data['due_date']) - pd.to_datetime(data['repay_date'])

data['days'] = data['days'].dt.days - data['tag']
print(max(data['days'].value_counts())/len(data)) 

data['days'].hist(bins=33)
data.head()
data_index = len(data)

test_index = len(test)



label = data['days']

data = data[data.columns[:5]]

data = data.append(test)

data.head()
list_data = pd.read_csv(basic_path+listing_info_path)

list_data.head()
list_data['principal'].describe()
# try_principal = list_data['principal'].apply(lambda x:8000 if x>8000 else x)

# list_data['principal_tag'] = np.array(list_data['principal'] == try_principal).astype(int)

# list_data['principal'] = try_principal
data_ = pd.merge(data,list_data,on=['listing_id','auditing_date','user_id'],how='left')

data_.head()
user_info_data = pd.read_csv(basic_path + user_info_path,parse_dates=['insertdate'])

user_info_data.head()
#保留最新信息

user_info_data = user_info_data.sort_values(by='insertdate', ascending=False)

user_info_data = user_info_data.drop_duplicates('user_id').reset_index(drop=True)

del user_info_data['insertdate']
user_info_data['cell_id_tag'] = np.array(user_info_data['cell_province'] == user_info_data['id_province']).astype(int)
for column in ['gender','cell_province','id_province']:

    le =LabelEncoder()

    user_info_data[column] = le.fit_transform(user_info_data[column])

user_info_data.head()

    
#排名编码

dict_tag = {}

i = 0

for tag in list(user_info_data['cell_province'].value_counts().index):

    i += 1

    dict_tag[tag] = i

user_info_data['cell_province'] = user_info_data['cell_province'].apply(lambda x:dict_tag[x])



dict_tag_1 = {}

i = 0

for tag in list(user_info_data['id_province'].value_counts().index):

    i += 1

    dict_tag_1[tag] = i

user_info_data['id_province'] = user_info_data['id_province'].apply(lambda x:dict_tag_1[x])





dict_tag_2 = {}

i = 0

for tag in list(user_info_data['id_city'].value_counts().index):

    i += 1

    dict_tag_2[tag] = i

user_info_data['id_city'] = user_info_data['id_city'].apply(lambda x:dict_tag_2[x])



user_info_data.head()
data_ = pd.merge(data_,user_info_data,on=['user_id'],how='left')

data_.head()
user_behavior_data = pd.read_csv(basic_path + user_behaviors_path)

user_behavior_data.head()
try_data = pd.get_dummies(user_behavior_data['behavior_type'])

try_data.columns = ['behavior_1','behavior_2','behavior_3']

try_data['user_id'] = user_behavior_data['user_id']



del user_behavior_data['behavior_type']

user_behavior_data['behavior_time'] = pd.to_datetime(user_behavior_data['behavior_time']).dt.hour

user_behavior_data = user_behavior_data.groupby(user_behavior_data['user_id']).mean()

user_behavior_data['user_id'] = user_behavior_data.index

user_behavior_data.head()
try_data = try_data.groupby(try_data['user_id']).sum()

try_data.head()

try_sum = np.zeros(len(try_data))

for column in try_data.columns:

    try_sum += try_data[column]

for column in try_data.columns:

    try_data[str(column)+'_percent'] = try_data[column]/try_sum

try_data['user_id'] = try_data.index
data_ = pd.merge(data_,try_data,on=['user_id'],how='left')

data_ = pd.merge(data_,user_behavior_data,on=['user_id'],how='left')

data_.head()
user_repay_data = pd.read_csv(basic_path+user_repay_logs_path)

user_repay_data.head()
user_repay_data_one = user_repay_data[user_repay_data['order_id']==1].reset_index(drop=True)

user_repay_data_one.head()
user_repay_data_one['one_forbidden'] = user_repay_data_one['repay_date'].apply(lambda x:1 if x=='2200-01-01' else 0)

user_repay_data_one['one_forbidden'].hist()
user_repay_data_one['due_date'] = pd.to_datetime(user_repay_data_one['due_date'])

user_repay_data_one['repay_date'] = pd.to_datetime(user_repay_data_one['repay_date'])

user_repay_data_one['one_distant_day'] = (user_repay_data_one['due_date'] - user_repay_data_one['repay_date']).dt.days 



user_repay_data_one['one_distant_day'] = user_repay_data_one['one_distant_day'].apply(lambda x:x if x>-1 else -1 )

#异常值处理

user_repay_data_one['one_distant_day'] = user_repay_data_one['one_distant_day'].apply(lambda x:31 if x>31 else x)
user_repay_data_one['one_distant_day'].plot(kind='box')
columns = ['listing_id','order_id','due_date','due_amt','repay_date']

for column in columns:

    del user_repay_data_one[column]

user_repay_data_one.head()
try_sum = user_repay_data_one.groupby(user_repay_data_one['user_id']).sum()

try_mean = user_repay_data_one.groupby(user_repay_data_one['user_id']).mean()

#try_std = user_repay_data_one.groupby(user_repay_data_one['user_id']).std()

try_max = user_repay_data_one.groupby(user_repay_data_one['user_id']).max()

try_min = user_repay_data_one.groupby(user_repay_data_one['user_id']).min()



try_sum['user_id'] = try_sum.index

try_mean['user_id'] = try_mean.index

try_max['user_id'] = try_max.index

try_min['user_id'] = try_min.index



try_data = pd.merge(try_sum,try_mean,on='user_id',how='left')

try_data = pd.merge(try_data,try_max,on='user_id',how='left')

try_data = pd.merge(try_data,try_min,on='user_id',how='left')

try_data.columns = ['user_id','repay_amt_sum','one_forbidden_sum','one_distant_day_sum',

                  'repay_amt_mean','one_forbidden_mean','one_distant_day_mean',

                  'repay_amt_max','one_forbidden_max','one_distant_day_max',

                  'repay_amt_min','one_forbidden_min','one_distant_day_min']

try_data.head()
data_ = pd.merge(data_,try_data,on='user_id',how='left')

data_.head()
user_repay_data['forbidden'] = user_repay_data['repay_date'].apply(lambda x:1 if x=='2200-01-01' else 0)

user_repay_data.head()
user_repay_data['due_date'] = pd.to_datetime(user_repay_data['due_date'])

user_repay_data['repay_date'] = pd.to_datetime(user_repay_data['repay_date'])

user_repay_data['distant_day'] = (user_repay_data['due_date'] - user_repay_data['repay_date'])



user_repay_data['distant_day'] = user_repay_data['distant_day'].dt.days      

user_repay_data['distant_day'] = user_repay_data['distant_day'].apply(lambda x:x if x>-1 else -1 )

#异常值处理

user_repay_data['distant_day'] = user_repay_data['distant_day'].apply(lambda x:31 if x>31 else x)

user_repay_data.head()
columns = ['listing_id','order_id','due_date','due_amt','repay_date']

for column in columns:

    del user_repay_data[column]
try_sum = user_repay_data.groupby(user_repay_data['user_id']).sum()

try_mean = user_repay_data.groupby(user_repay_data['user_id']).mean()

try_max = user_repay_data.groupby(user_repay_data['user_id']).max()

try_min = user_repay_data.groupby(user_repay_data['user_id']).min()



try_sum['user_id'] = try_sum.index

try_mean['user_id'] = try_mean.index

try_max['user_id'] = try_max.index

try_min['user_id'] = try_min.index



try_data = pd.merge(try_sum,try_mean,on='user_id',how='left')

try_data = pd.merge(try_data,try_max,on='user_id',how='left')

try_data = pd.merge(try_data,try_min,on='user_id',how='left')

try_data.columns = ['user_id','all_repay_amt_sum','forbidden_sum','distant_day_sum',

                  'all_repay_amt_mean','forbidden_mean','distant_day_mean',

                  'all_repay_amt_max','forbidden_max','distant_day_max',

                  'all_repay_amt_min','forbidden_min','distant_day_min']

try_data.head()
data_ = pd.merge(data_,try_data,on='user_id',how='left')

data_.head()
del data,list_data,user_info_data,user_behavior_data,user_repay_data,user_repay_data_one,try_sum,try_mean,test

gc.collect()
data_.head()
# #可知due_date和auditing_date相差一月，可知均为第一期

# date_list = pd.to_datetime(data_['due_date']) - pd.to_datetime(data_['auditing_date'])

# date_list = date_list.dt.days

# date_list.hist()
#应还款的月份和天数

data_['due_month'] = pd.to_datetime(data_['due_date']).apply(lambda x:x.month)

data_['due_day'] = pd.to_datetime(data_['due_date']).apply(lambda x:x.day)

data_['due_weekday'] = pd.to_datetime(data_['due_date']).apply(lambda x:x.weekday())
#贷款至注册时间的月数

data_['reg_to_audit'] = 12*(pd.to_datetime(data_['auditing_date']).apply(lambda x:x.year) - pd.to_datetime(data_['reg_mon']).apply(lambda x:x.year))+(pd.to_datetime(data_['auditing_date']).apply(lambda x:x.month) - pd.to_datetime(data_['reg_mon']).apply(lambda x:x.month))
del data_['auditing_date']

del data_['due_date']

del data_['reg_mon']
data_.head()
columns_null = []

for column in data_.columns:

    try:

        percent = data_[column].isnull().sum()/len(data_)

        print(column+':  '+str(percent))

        if percent!=0:

            columns_null.append(column)

    except:

        columns_null.append(column)
for column in columns_null:  #使用平均数填充

    try_data = np.array(data_[column]).copy()

    data_[column] = data_[column].fillna(data_[column].mean())

    data_[str(column)+'non_tag'] = (np.array(data_[column]) != try_data).astype(int)

import matplotlib.pyplot as plt
# for column in data_.columns:

#     print(data_[column].describe())

#     plt.boxplot(data_[column])

#     print(column)

#     plt.show()
for column in data_.columns:

    try_data = np.array(data_[column]).copy()

    distance = data_[column].quantile(0.75) - data_[column].quantile(0.25)

    tag = data_[column].quantile(0.75) + 1.5*distance

    data_[column] = data_[column].apply(lambda x:x if x<tag else tag)

    tag = data_[column].quantile(0.25) - 1.5*distance

    data_[column] = data_[column].apply(lambda x:x if x>tag else tag)

    data_[str(column)+'_tag'] = (np.array(data_[column]) != try_data).astype(int)

    if len(np.unique(data_[str(column)+'_tag'])) == 1:

        del data_[str(column)+'_tag']
data_.columns
# import matplotlib.pyplot as plt

# for column in data_.columns:

#     try:

#         print(column)

#         plt.boxplot(data_[column])

#         plt.show()

#     except:

#         continue
data_.head()
train_data = data_[:data_index]

train_data['label'] = label

test_data = data_[data_index:]

del data_

gc.collect()

train_data.to_csv("treemodel_train.csv",index=False)

test_data.to_csv("treemodel_test.csv",index=False)
# #使用

# user_taglist_data = pd.read_csv(basic_path+user_taglist_path)

# user_taglist_data.head()
# #保留最新信息

# user_taglist_data = user_taglist_data.sort_values(by='insertdate', ascending=False)

# user_taglist_data = user_taglist_data.drop_duplicates('user_id').reset_index(drop=True)

# del user_taglist_data['insertdate']
# len(np.unique(user_taglist_data['taglist']))
# len(np.unique(data_['user_id']))
# comment_text = []

# for i in range(len(user_taglist_data)):

#     try_data = user_taglist_data['taglist'][i].split('|')

#     comment_text.append(try_data)
# from collections import Counter

# counter = Counter([])

# for i in range(len(user_taglist_data)):

#     counter += Counter(comment_text[i])
# a = counter.most_common(100) #使用较常见的90个标签

# b = counter.most_common(10)

# a = list(set(a)-set(b))
# tags = []

# for (c,d) in a:

#     tags.append(c)

# tag_dict = {}

# for i in range(90):

#     tag_dict[tags[i]] = i

# tag_dict
#LDA时间

# from gensim import corpora

# dictionary = corpora.Dictionary(comment_text)

# doc_term_matrix = [dictionary.doc2bow(doc) for doc in comment_text            ]



# import gensim

# Lda = gensim.models.ldamodel.LdaModel

# ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=50)
# from collections import defaultdict



# tag_dict = defaultdict(int)

# max_num = 0

# max_try = 0

# for i in range(len(user_taglist_data)):

#     try_data = user_taglist_data['taglist'][i].split('|')

#     if max_try>max_num:

#         max_num = max_try

#     max_try = 0

#     for i in try_data:

#         tag_dict[i] += 1/len(user_taglist_data)

#         max_try += 1



# print(max_num) 

# tag_dict
# del counter,comment_text

# gc.collect()
# ont_hot_matrix = []

# for i in range(len(user_taglist_data)):

#     try_tag = np.zeros(90)

#     try_data = user_taglist_data['taglist'][i].split('|')

#     for column in try_data:

#         try:

#             num = tag_dict[column]

#             try_tag[num] = 1

#         except:

#             try_tag[0] = 1

#     ont_hot_matrix.append(try_tag)

# ont_hot_matrix
# try_data = pd.DataFrame(np.array(ont_hot_matrix))

# user_taglist_data = pd.concat([user_taglist_data,try_data],axis=1)

# user_taglist_data.head()
# for column in user_taglist_data.columns[1:3]:

#     del user_taglist_data[column]
# data_ = pd.merge(data_,user_taglist_data,on='user_id',how='left')

# data_.head()
# columns_remove = ['user_id','listing_id']

# date_columns = ['auditing_data','due_date']