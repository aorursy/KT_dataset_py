# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# ! git clone https://github.com/geek-ai/Texygen.git
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling 
import plotly.express as px

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from random import sample

import pickle
# class MultiColumnLabelEncoder:
#     def __init__(self,columns = None):
#         self.columns = columns # array of column names to encode

#     def fit(self,X,y=None):
#         return self # not relevant here

#     def transform(self,X):
#         '''
#         Transforms columns of X specified in self.columns using
#         LabelEncoder(). If no columns specified, transforms all
#         columns in X.
#         '''
#         output = X.copy()
#         if self.columns is not None:
#             for col in self.columns:
#                 output[col] = LabelEncoder().fit_transform(output[col])
#         else:
#             for colname,col in output.iteritems():
#                 output[colname] = LabelEncoder().fit_transform(col)
#         return output

#     def fit_transform(self,X,y=None):
#         return self.fit(X,y).transform(X)
    
def get_index_list(cat_num_list):
    '''
    e.g. 
    input: [4,5,6]
    output: [30,6,1]
    '''
    reverse_num_list = cat_num_list[::-1]
    index_list_reverse = []
    count = 1
    for i in reverse_num_list:
        index_list_reverse.append(count)
        count *= i
    index_list = index_list_reverse[::-1]
    return index_list
    
def encode_cat(data_list,cat_num_list):
    '''
    from [1,1,1] & [4,5,6] to get index 37
    '''
    index_list = get_index_list(cat_num_list)
    return np.dot(data_list,index_list)

def decode_cat(index,cat_num_list):
    '''
    from index 37 to get [1,1,1] given [4,5,6]
    '''
    index_list = get_index_list(cat_num_list)
    data_list = []
    for i in index_list:
        num = index // i
        data_list.append(num)
        index -= num * i
    return data_list
import random
class Shuffle_Transformer:
    def __init__(self,index):
        self.index = index
        index_2 = index.copy()
        random.shuffle(index_2)
        self.shuffle_index = index_2
        self.index_map = dict(zip(self.index,self.shuffle_index))
        self.inverse_index_map = {v: k for k, v in self.index_map.items()}  
    def get_index_map(self):
        return self.index_map
    def get_shuffle(self,lst):
        y = list(map(self.index_map.get, lst))
        return y
    def get_inverse(self,lst):
        z = list(map(self.inverse_index_map.get, lst))
        return z
    
from sklearn.preprocessing import OrdinalEncoder
# test_block = message.iloc[0:10,:]
# feature_encoder = OrdinalEncoder()
# feature_encoder.fit(test_block[['Type','Direction']])
# new = test_block.copy()
# new[['Type','Direction']] = feature_encoder.transform(test_block[['Type','Direction']])
# Change it
massege_file = "/kaggle/input/lob-msft/MSFT_2012-06-21_34200000_57600000_message_1.csv"
order_file = "/kaggle/input/lob-msft/MSFT_2012-06-21_34200000_57600000_orderbook_1.csv"
message= pd.read_csv(massege_file,header=None,names=['Time','Type','OrderID','Size','Price','Direction'])
orderbook=pd.read_csv(order_file,header=None,names=['AskPrice','AskSize','BidPrice','BidSize'])
message.head()
message.info()
##Time Difference
message['Time_Diff'] = message.Time.diff()
message = message.drop([0])
message['Time_Diff'] = message['Time_Diff'].apply(lambda x: int(x*1000000))
message = message[message['Time_Diff']<100]
message['Time_Diff_5'] = message['Time_Diff'].apply(lambda x: int((x//5)+1))


##Price
message['Real_Price'] = message["Price"]/10000
message['Round_Price_Times_10'] = message['Real_Price'].apply(lambda x: int(round(x,1)*10))


#SIZE DIRECTION AND TYPE
size_filter_list = list(range(100,1300,100))
message = message[message['Size'].isin(size_filter_list)]
message["Size_Int"] = message["Size"].apply(lambda x: int(x/100))

message = message.reset_index(drop=True)
sns.distplot(message.Time_Diff_5)
print("Time Diff unique values: {}".format(len(message['Time_Diff_5'].unique())))
print("Type unique values: {}".format(len(message['Type'].unique())))
print("Direction unique values: {}".format(len(message['Direction'].unique())))
print("Size Int unique values: {}".format(len(message['Size_Int'].unique())))
print("Round Price unique values: {}".format(len(message['Round_Price_Times_10'].unique())))
message.shape
message.head(100)
df = message[['Direction','Type','Size_Int','Time_Diff_5','Round_Price_Times_10']][0:100000]
df.head()
df.info()
fea_columns = ['Direction','Type','Size_Int','Time_Diff_5','Round_Price_Times_10']
feature_encoder = OrdinalEncoder()
feature_encoder.fit(block[fea_columns])
new_block = block.copy()
new_block[fea_columns] = feature_encoder.transform(block[fea_columns])

with open('OrdinalEncoder.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(feature_encoder, f)
new_block.info()
new_block = new_block.astype('int64')
new_block.head()
# #SIZE DIRECTION AND TYPE
# size_filter_list = list(range(100,1300,100))
# message_size = message[message['Size'].isin(size_filter_list)]
# message_size["Size"] = message_size["Size"].apply(lambda x: int(x/100))
# message_size['Size'].nunique()
# # message_size['Size'] = message_size['Size'].apply(str)
# test_block = message_size.loc[:,["Type","Size","Direction"]]
# # new_test_block = MultiColumnLabelEncoder(columns = ['Type','Size','Direction']).fit_transform(test_block)
# # new_test_block.head()
# test_block = test_block.reset_index(drop = True)
# size_block = test_block.iloc[0:10000,:]
# size_block.shape
# size_block.head()
num_list = [j.nunique() for i,j in new_block.iteritems()]
print(num_list)
new_block['index'] = new_block.apply(encode_cat,axis = 1,cat_num_list = num_list)
new_block.head()
shuffle_encoder = Shuffle_Transformer(list(new_block['index']))
shuffle_dict = shuffle_encoder.get_index_map()
new_block['shuffle_index'] = shuffle_encoder.get_shuffle(list(new_block['index']))
with open('ShuffleEncoder.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(shuffle_encoder, f)
new_block.to_csv("After_Process_Data.csv")
new_block
index_list = list(new_block["shuffle_index"])
def list_of_groups(init_list, childern_list_len):
    '''
    init_list为初始化的列表，childern_list_len初始化列表中的几个数据组成一个小列表
    :param init_list:
    :param childern_list_len:
    :return:
    '''
    list_of_group = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_group]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list
index_sub = list_of_groups(index_list,20)
index_str = [" ".join(map(lambda x:str(x), i)) for i in index_sub]

with open('SeqGAN_input.txt', 'w') as filehandle:
    for listitem in index_str:
        filehandle.write('%s\n' % listitem)
# Import Data
block = pd.read_csv('/kaggle/input/lob-msft/After_Process_Data.csv',index_col=0)

with open('/kaggle/input/lob-msft/ShuffleEncoder.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    shuffle_encoder = pickle.load(f)

with open('/kaggle/input/lob-msft/OrdinalEncoder.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    feature_encoder = pickle.load(f)    
    
#get output list
listOfLines = list()        
with open ("/kaggle/input/lob-msft/SeqGAN_output_may_7.txt", "r") as myfile:
    for line in myfile:
        listOfLines.append(line.strip()) 
output_list = []
for i in listOfLines:
    for s in i.split(' '):
        output_list.append(int(s))
block.head()
print("Time Diff unique values: {}".format(len(block['Time_Diff_5'].unique())))
print("Type unique values: {}".format(len(block['Type'].unique())))
print("Direction unique values: {}".format(len(block['Direction'].unique())))
print("Size Int unique values: {}".format(len(block['Size_Int'].unique())))
print("Round Price unique values: {}".format(len(block['Round_Price_Times_10'].unique())))
len(output_list)
num_list
sns.distplot(block['index'], bins = 100)
sns.distplot(np.array(output_list), bins = 100)
output_block = pd.DataFrame(columns = block.columns)
output_block['shuffle_index'] = np.array(output_list)
output_block['index'] = shuffle_encoder.get_inverse(list(output_block['shuffle_index']))

fea_columns = ['Direction','Type','Size_Int','Time_Diff_5','Round_Price_Times_10']

num_list = [j.nunique() for i,j in block[fea_columns].iteritems()]

output_block[fea_columns] = output_block['index'].apply(lambda x: pd.Series(decode_cat(x,cat_num_list = num_list)))

# for index, row in output_block.iterrows():
#     row['Direction','Type','Size_Int','Time_Diff_5','Round_Price_Times_10']=decode_cat(row['index'],num_list)
#     print(index)
output_block.shape
new_output_block.shape
new_output_block = output_block.copy()
new_output_block[fea_columns] = feature_encoder.inverse_transform(output_block[fea_columns])
new_block = block.copy()
new_block[fea_columns] = feature_encoder.inverse_transform(block[fea_columns])
new_output_block.to_csv('Synthetic_Data.csv')
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
plt.figure(2, figsize=(12,8))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0],  title='Real Data')
sns.countplot(x='Direction',data=new_block)

plt.subplot(the_grid[0, 1], title='Synthetic Data')
sns.countplot(x='Direction', data=new_output_block)

plt.suptitle('Direction Comparison', fontsize=16)
plt.figure(2, figsize=(12,8))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0],  title='Real Data')
sns.countplot(x='Type',data=new_block)

plt.subplot(the_grid[0, 1], title='Synthetic Data')
sns.countplot(x='Type', data=new_output_block)

plt.suptitle('Type Comparison', fontsize=16)
plt.figure(2, figsize=(12,8))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0],  title='Real Data')
sns.countplot(x='Size_Int',data=new_block)

plt.subplot(the_grid[0, 1], title='Synthetic Data')
sns.countplot(x='Size_Int', data=new_output_block)

plt.suptitle('Size Comparison *100', fontsize=16)
plt.figure(2, figsize=(12,8))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0],  title='Real Data')
sns.countplot(x='Time_Diff_5',data=new_block)

plt.subplot(the_grid[0, 1], title='Synthetic Data')
sns.countplot(x='Time_Diff_5', data=new_output_block)

plt.suptitle('Time Differenence Comparison (0.000001s)', fontsize=16)
plt.figure(2, figsize=(12,8))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='Source of Pies')
sns.countplot(x='Round_Price_Times_10',data=new_block)

plt.subplot(the_grid[0, 0], title='Selected Flavors of Pies')
sns.countplot(x='Round_Price_Times_10', data=new_output_block)

plt.suptitle('Pie Consumption Patterns in the United States', fontsize=16)
fig = px.pie(size_block,names='Size')
fig.show()
listOfLines = list()        
with open ("/kaggle/input/lob-msft/SeqGAN_input.txt", "r") as myfile:
    for line in myfile:
        listOfLines.append(line.strip()) 

input_list = []
for i in listOfLines:
    for s in i.split(' '):
        input_list.append(int(s))
len(input_list)
#get output list
listOfLines = list()        
with open ("/kaggle/input/lob-msft/SeqGAN_output.txt", "r") as myfile:
    for line in myfile:
        listOfLines.append(line.strip()) 
output_list = []
for i in listOfLines:
    for s in i.split(' '):
        output_list.append(int(s))
num_list = [j.nunique() for i,j in test_block.iteritems()]
num_list
fake_data = size_block.copy()
fake_data.head()
small_output_list = sample(output_list,10000)
#update the output_list to a-list
fake_data['index'] = small_output_list
fake_data.head()
for index, row in fake_data.iterrows():
    row["Type","Size","Direction"]=decode_cat(row['index'],num_list) 
fake_data.head()
    
#fake_data[["Type","Size","Direction"]] = fake_data['index'].apply(decode_cat,cat_num_list = num_list)
# len(output_list)
fig = px.pie(fake_data,names='Size')
fig.show()
message.head()
message["Price"]= message["Price"]/10000
message.head()
#pandas_profiling.ProfileReport(message)
message.shape
fig = px.pie(message,names='Type')
fig.show()
fig = px.pie(message,names='Size')
fig.show()
fig = px.pie(fake_data,names='Type')
fig.show()
fig = px.pie(message,names='Direction')
fig.show()
fig = px.pie(fake_data,names='Direction')
fig.show()
ax = sns.distplot(message['Price'])
message["Price"].describe()
message['time_diff'] = message.Time.diff()
message.head()
message['time_diff'].describe()
a = message[message['time_diff']<0.0002]
a['time_diff'] = a['time_diff']*100000

ax1 = sns.distplot(a['time_diff'])
a.describe()
#SIZE DIRECTION AND TYPE
size_filter_list = list(range(100,1300,100))
message_size = message[message['Size'].isin(size_filter_list)]
message_size["Size"] = message_size["Size"].apply(lambda x: int(x/100))
message_size['Size'].nunique()
# message_size['Size'] = message_size['Size'].apply(str)
test_block = message_size.loc[:,["Type","Size","Direction","Price"]]
new_test_block = MultiColumnLabelEncoder(columns = ['Type','Size','Direction','Price']).fit_transform(test_block)
new_test_block.head()
num_list = [j.nunique() for i,j in new_test_block.iteritems()]
output_chunk = new_test_block.iloc[1:10000,:]
output_chunk['index'] = output_chunk.apply(encode_cat,axis = 1,cat_num_list = num_list)
output_chunk.head()
ax = sns.distplot(output_chunk["index"])
