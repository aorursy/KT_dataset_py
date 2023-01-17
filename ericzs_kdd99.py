# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xlrd

import csv

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class config:

    kdd99train='../input/kddcup99data/kddcup.data'

    kdd99test='../input/kddcup99/KDDCup99.csv'

def attack_class(train):

    train.loc[(train['label'].isin(['back.','land.','neptune.','pod.','smurf.','teardrop.'])),'label_type']='DOS'

    train.loc[train['label'].isin(['ipsweep.','nmap.','portsweep.','satan.']),'label_type']='Probing'

    train.loc[train['label'].isin(['ftp_write.','guess_passwd.','imap.','multihop.','phf.','spy.','warezclient.','warezmaster.']),'label_type']='R2L'

    train.loc[train['label'].isin(['buffer_overflow.','loadmodule.','perl.','rootkit.']),'label_type']='U2R'

    train.loc[train['label']=='normal.','label_type']='Normal'

    return train



def attack_class2(train):

    train.loc[(train['label'].isin(['back','land','neptune','pod','smurf','teardrop'])),'label_type']='DOS'

    train.loc[train['label'].isin(['ipsweep','nmap','portsweep','satan']),'label_type']='Probing'

    train.loc[train['label'].isin(['ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster']),'label_type']='R2L'

    train.loc[train['label'].isin(['buffer_overflow','loadmodule','perl','rootkit']),'label_type']='U2R'

    train.loc[train['label']=='normal','label_type']='Normal'

    return train





cleanup_nums = {"protocol_type":     {"tcp": 1, "icmp": 2, "udp": 3},

                "service": {"vmnet": 1, "smtp": 2, "ntp_u":3, "shell":4, "kshell":5, "aol":6, "imap4":7, "urh_i":8, "netbios_ssn":9,

                           "tftp_u":10, "mtp":11, "uucp":12, "nnsp":13, "echo":14, "tim_i":15, "ssh":16, "iso_tsap":17, "time":18,

                           "netbios_ns":19,"systat":20, "hostnames":21, "login":22, "efs":23, "supdup":24, "http_8001":25, "courier":26,

                           "ctf":27,"finger":28,"nntp":29,"ftp_data":30,"red_i":31,"ldap":32,"http":33,"ftp":34,"pm_dump":35,"exec":36,

                           "klogin":37,"auth":38,"netbios_dgm":39,"other":40,"link":41,"X11":42,"discard":43,"private":44,"remote_job":45,

                           "IRC":46,"daytime":47,"pop_3":48,"pop_2":49,"gopher":50,"sunrpc":51,"name":52,"rje":53,"domain":54,"uucp_path":55,

                           "http_2784":56,"Z39_50":57,"domain_u":58,"csnet_ns":59,"whois":60,"eco_i":61,"bgp":62,"sql_net":63,"printer":64,

                           "telnet":65,"ecr_i":66,"urp_i":67,"netstat":68,"http_443":69,"harvest":70},

               "flag":{"RSTR":1,"S3":2,"SF":3,"RSTO":4,"SH":5,"OTH":6,"S2":7,"RSTOS0":8,"S1":9,"S0":10,"REJ":11},

               "label_type":{"Normal":1,"DOS":2,"Probing":3,"R2L":4,"U2R":5}}

attack_map={"label_type":{"Normal":1,"DOS":2,"Probing":3,"R2L":4,"U2R":5}}



column = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","lnum_compromised","lroot_shell","lsu_attempted","lnum_root","lnum_file_creations","lnum_shells","lnum_access_files","lnum_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]



print("train data loading")

train = pd.read_csv(config.kdd99train)

train.columns= column

train.insert(0,'label_type','NAN')

# train.info()

# train.head()

# train['label'].value_counts()

# train['label_type'].value_counts()

#Mapping of Categorical data in the dataset





datapre = train

# datapre.transpose()

#Replacing the encoded data in the dataset

datapre.replace(cleanup_nums, inplace=True)

# datapre.transpose()

# datapre.head()





train = attack_class(datapre)



train['label_type'].value_counts()



# df[['two', 'three']] = df[['two', 'three']].astype(float)

train.replace(attack_map, inplace=True)

# train.head()

train.drop('label', axis = 1, inplace = True)

# train.head()







print("test data loading")

test = pd.read_csv(config.kdd99test)

test.insert(0,'label_type','NAN')

# test.info()

# test.head()

# test['label'].value_counts()



datapre = test

# datapre.transpose()

#Replacing the encoded data in the dataset

datapre.replace(cleanup_nums, inplace=True)

# datapre.transpose()

# datapre.head()

# datapre.head()

test = attack_class2(datapre)



test['label_type'].value_counts()



# df[['two', 'three']] = df[['two', 'three']].astype(float)

test.replace(attack_map, inplace=True)

# test.head()

test.drop('label', axis = 1, inplace = True)

test.head()
train.info()
test.info()

#Cross validation

# from sklearn import cross_validation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



train_data = train.values

features_train = train_data[:,1:42]

labels_train = train_data[:,0]



test_data = test.values

features_test = test_data[:,1:42]

labels_test = test_data[:,0]
#Naive Bayes Classifier

from sklearn import metrics

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error

clf = GaussianNB()

clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

acc=accuracy_score(prediction, labels_test)

print("------------------------------------------")

print("Accuracy = ",acc*100," %")

matrix = confusion_matrix(labels_test, prediction)

print(matrix)

report = classification_report(labels_test, prediction)

print(report)



from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(labels_test, prediction)

roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Applying Logistic regression to find important features. Higher the rank more important the attribute

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

rfe = RFE(model, 10)

fit = rfe.fit(features_train, labels_train)

print("Num Features: ",fit.n_features_)

print("Selected Features: ",fit.support_)

print("Ranking of features: ",fit.ranking_)