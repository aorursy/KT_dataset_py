# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import iqr
from scipy.stats import skew 
from scipy.stats import kurtosis
import scipy.stats 
import re

# Create Key Value from Data
def create_key_value(data):
    res = []
    for i in range(len(data)):
        key_value = {}
        for j in range(len(data[i])):
            info = data[i][j].split('=')
            key_value[info[0]] = info[1]
        res.append(key_value)
    return res
# Convert data from touple to list
def convert_touple_list(dict_data,label):
    complete_data = []
    for i in label:
        each_label_data = []
        for j in range(len(dict_data)):
            if i in dict_data[j] :
                each_label_data.append(dict_data[j].pop(i))
            else :
                each_label_data.append(None)
        complete_data.append(each_label_data)
    return complete_data
# Counter for calculate not None values
def value_counter(data):
    feature_size = len(data[0])
    res =[]
    for i in range(len(data)):
        count = 0
        for j in range(feature_size):
            if data[i][j] is not None:
                count = count + 1
        res.append(count)
    return np.array(res)
# seperate not None values from feature
def feature_label(frame_data,feature_label):
    t = frame_data[feature_label].values
    type(t)
    counter = []
    none_value = 0
    zero_value = 0
    for i in range(len(t)):
        if t[i] is not None :
            if int(t[i]) > 0 :
                counter.append(int(t[i]))
            elif int(t[i]) == 0 :
                zero_value = zero_value + 1
        else :
            none_value = none_value + 1
    return counter,none_value,zero_value
# statistical feature such as mean , median , std and ... .
def stats_feature(data):
    sent_byte_dict_info = {
        'mean' : np.mean(data),
        'median' : np.median(data),
        'Std' : np.std(data),
        'Iqr' : iqr(data),
        'Kurtosis' : kurtosis(data),
        'Skewness' : skew(data)
    }
    return sent_byte_dict_info
# plot histograms of features
def plot_features_histogram(data):
    plt.figure(figsize=(10,10))
    plt.hist(data,bins=40)
    plt.xlabel('Feature Size')
    plt.ylabel('Number of Samples')
    plt.title("Visualizing Number of Features Have Each Sample")
# plot statistical features
def plot_stats_features(data):
    keys_feature = []
    values_feature = []
    for x,y in data.items() :
        keys_feature.append(x)
        values_feature.append(y)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(keys_feature,values_feature)
    plt.show()
# plot complete data with boxplot
def plot_compelete_data(data):
    sns.set_style("darkgrid")
    sns.boxplot(data=data);
    plt.title("Before Delete Outliers")
# plot data after delete outliers
def plot_data_with_number_outlier(data, limit):
    c_less_t = [x for x in data if x <= limit]
    sns.set_style("darkgrid")
    sns.boxplot(data=c_less_t);
    plt.title("After Delete Outliers > "+ str(limit))
# plot histogram
def plot_histogram(data):
    plt.figure(figsize=(10,10))
    plt.hist(data,bins=50)
# delete outliers with limitation value
def delete_outliers(data, limit):
    for item in data:
        if item > limit :
            data.remove(item)
    return data
# plot normal distribution with mean and std of data
def plot_normal_distribution(data,mean,std):
    x = np.linspace(min(data)-100, max(data)+100, 1000)
    y = scipy.stats.norm.pdf(x,mean,std)
    plt.plot(x, y , color='coral')
# read data from csv
frame = pd.read_csv("/kaggle/input/graylog-search-result-absolute-2020-02-20T20_30_00.000Z-2020-02-21T20_30_00.000Z.csv")
# split message
data = frame.iloc[:,2]
# extract features from message to list
data = np.array(data)
for i in range(len(data)):
    data[i] = re.findall('([a-z]+="{0,1}[^="]+"{0,1} {1,})', data[i])
# labels 
label = ['date','time','devname','devid','logid','type',
         'subtype','level','vd','eventtime','srcip','srcport',
         'srcintf','srcintfrole','dstip','dstport','dstintfrole',
         'poluuid','sessionid','proto','action','policyid',
         'policytype','service','srccountry','trandisp','duration',
         'sentbyte','rcvdbyte','sentpkt','appcat','crscore',
         'craction','crlevel','devtype','mastersrcmac','srcmac',
         'srcserver','dstintf','dstcountry','wanin','wanout','lanin',
         'lanout','utmaction','countdlp','app','eventtype','id','epoch',
         'port','int','profile','kind','status','dir','transport','rcvdpkt',
         'from','logdesc','cpu','totalsession','bandwidth','setuprate',
         'disklograte','fazlograte','msg','mem','disk','interface','total',
         'used','dstdevtype','masterdstmac','dstmac','remip','locip','remport',
         'locport','outintf','cookies','user','group','xauthuser','xauthgroup',
         'assignip','vpntunnel','init','mode','stage','role','num','spi',
         'filteridx','filtername','dlpextra','filtertype','filtercat','severity',
         'eventid','filetype','direction','hostname','url','filename','filesize']
np.shape(label)
# create key value from data list
dict_data = create_key_value(data)
# print a sample of key value
dict_data[0]
# convert touple to list
complete_data = convert_touple_list(dict_data,label)
# create data frame from list (transpose)
frame_data = pd.DataFrame(complete_data,index=label).T
# show 20 top samples in data frame
frame_data.head(20)
# each data how many features have
feature_num = value_counter(frame_data.values)
# each feature exist to how many samples
num_feature = value_counter(frame_data.T.values)
# plot histogram of samples have how many features
plot_features_histogram(feature_num)
num_feature
# each feature not None in how many samples of dataset and you could see features less than 20 times repeat
# are not important
plt.figure(figsize=(20,20))
sns.barplot(x=num_feature, y=label)
# Add labels to your graph
plt.xlabel('Number of Samples')
plt.ylabel('Features')
plt.legend()
plt.show()
# get rcvdbyte 
sent_byte_data, sent_byte_none_value, sent_byte_zero_value = feature_label(frame_data,'rcvdbyte')
# plot histogram of rcvdbyte
plot_histogram(sent_byte_data)
# number of each value in sample set (not None sample set)
Counter(sent_byte_data)
# statistical features
sent_byte_dict_info = stats_feature(sent_byte_data)
# plot statistical features 
plot_stats_features(sent_byte_dict_info)
# normal distribution from std and mean rcvdbyte
plot_normal_distribution(sent_byte_data,sent_byte_dict_info['mean'],sent_byte_dict_info['Std'])
# baxplot of data
plot_compelete_data(sent_byte_data)
# delete data that greater than 3 standard devation from mean (99.7 data inside +- 3 std in normal distribution)
sent_byte_data_del_out = delete_outliers(sent_byte_data, sent_byte_dict_info['mean'] + sent_byte_dict_info['Std'] * 3)
plot_histogram(sent_byte_data_del_out)
# calculate and plot statistical features 
sent_byte_dict_info_1 = stats_feature(sent_byte_data_del_out)
plot_stats_features(sent_byte_dict_info_1)
# plot normal distribution with mean and std
plot_normal_distribution(sent_byte_data_del_out,sent_byte_dict_info_1['mean'],sent_byte_dict_info_1['Std'])
# boxplot 
plot_data_with_number_outlier(sent_byte_data_del_out,sent_byte_dict_info['mean'] + sent_byte_dict_info['Std'] * 3)
# delete data that greater than 3 standard devation from mean again (99.7 data inside +- 3 std in normal distribution)
sent_byte_data_del_out_2 = delete_outliers(sent_byte_data, sent_byte_dict_info_1['mean'] + sent_byte_dict_info_1['Std'] * 3)
plot_histogram(sent_byte_data_del_out_2)
# plot statistical feature 
sent_byte_dict_info_2 = stats_feature(sent_byte_data_del_out_2)
plot_stats_features(sent_byte_dict_info_2)
# plot normal distribution and you could see normal distribution very smooth and clear 
plot_normal_distribution(sent_byte_data_del_out_2,sent_byte_dict_info_2['mean'],sent_byte_dict_info_2['Std'])
# boxplot
plot_data_with_number_outlier(sent_byte_data_del_out_2, sent_byte_dict_info_1['Std'] * 3)