#Importing Required Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.float_format = '{:,.4f}'.format

%matplotlib inline

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

import warnings

warnings.filterwarnings('ignore')
#The CSV file is imported using the read_csv command given below:

IDS_df = pd.read_csv("../input/Train.txt",header=None,names=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",

"wrong_fragment","urgent","hot","num_failed_logins","logged_in",

"num_compromised","root_shell","su_attempted","num_root","num_file_creations",

"num_shells","num_access_files","num_outbound_cmds","is_host_login",

"is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",

"rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",

"dst_host_diff_srv_rate","dst_host_same_src_port_rate",

"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",

"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

 )

IDS_df_Test = pd.read_csv("../input/Test (1).txt",header=None,names=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",

"wrong_fragment","urgent","hot","num_failed_logins","logged_in",

"num_compromised","root_shell","su_attempted","num_root","num_file_creations",

"num_shells","num_access_files","num_outbound_cmds","is_host_login",

"is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",

"rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",

"dst_host_diff_srv_rate","dst_host_same_src_port_rate",

"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",

"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"])

IDS_df_Test.head()
IDS_df.head()
#createing targets for Multinomial  classification

switcher={"neptune":"DoS","normal":"normal","saint":"Probe","mscan":"Probe",

"guess_passwd":"R2L","smurf":"DoS","apache2":"DoS","satan":"Probe","buffer_overflow":"U2R","back":"DoS",

"warezmaster":"R2L","snmpgetattack":"R2L","processtable":"DoS","pod":"DoS",

"httptunnel":"R2L","nmap":"Probe","ps":"U2R","snmpguess":"R2L","ipsweep":"Probe",

"mailbomb":"R2L","portsweep":"Probe","multihop":"R2L","named":"R2L","sendmail":"R2L","loadmodule":"U2R",

"xterm":"U2R","worm":"DoS","teardrop":"DoS","rootkit":"U2R","xlock":"R2L","perl":"U2R","land":"DoS","xsnoop":"R2L",

"sqlattack":"U2R","ftp_write":"R2L","imap":"R2L","udpstorm":"DoS","phf":"R2L"}

    

IDS_df["attack_class"]=IDS_df["attack"].apply(lambda x:switcher.get(x, "Invalid"))

IDS_df.head()
IDS_df_Test["attack_class"]=IDS_df_Test["attack"].apply(lambda x:switcher.get(x, "Invalid"))

IDS_df_Test.head()
dummy = pd.get_dummies(IDS_df_Test[['protocol_type', 'service', 'flag']], drop_first=True)

Test_df = pd.concat([IDS_df_Test, dummy], axis=1)

Test_df = Test_df.drop(['protocol_type', 'service', 'flag'], 1)

Test_df.head()

Test_y=Test_df[["attack_class"]]

Test_x=Test_df.drop(["attack","attack_class"], 1)

Test_y.head()

dummy = pd.get_dummies(IDS_df[['protocol_type', 'service', 'flag']], drop_first=True)

Train_df = pd.concat([IDS_df, dummy], axis=1)

Train_df = Train_df.drop(['protocol_type', 'service', 'flag'], 1)

Train_df.head()

Train_y=Train_df[["attack_class"]]

Train_x=Train_df.drop(["attack","attack_class"], 1)

Train_y.head()
from sklearn.preprocessing import StandardScaler

colnum=["duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot","num_failed_logins","num_compromised","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]



scaler = StandardScaler()



Train_x[colnum] = scaler.fit_transform(Train_x[colnum])



Test_x[colnum] = scaler.transform(Test_x[colnum])



Test_x.head()
Train_x.head()
set(list(Train_x.columns)).difference(set(list(Test_x.columns)))
Test_x['service_aol'] =  [0] * len(Test_x)

Test_x['service_harvest']=  [0] * len(Test_x)

Test_x['service_http_2784'] =  [0] * len(Test_x)

Test_x['service_http_8001'] =  [0] * len(Test_x)

Test_x['service_red_i'] =  [0] * len(Test_x)

Test_x['service_urh_i'] =  [0] * len(Test_x)
set(list(Train_x.columns)).difference(set(list(Test_x.columns)))
Train_y.attack_class.unique()
#createing targets for Multinomial  classification

switche1r={'normal':0, 'DoS':1, 'Invalid':2, 'Probe':3, 'R2L':4, 'U2R':5}

    

Train_y["attack_class"]=Train_y["attack_class"].apply(lambda x:switche1r.get(x, "Invalid"))

Train_y.head()
Test_y["attack_class"]=Test_y["attack_class"].apply(lambda x:switche1r.get(x, "Invalid"))

Test_y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, train_size=0.8, test_size=0.2, random_state=100)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model = model.fit(X_train, y_train)



# check the accuracy on the training set

model.score(X_train, y_train)
from sklearn import metrics

from sklearn.model_selection import cross_val_score

predicted = model.predict(X_test)

probs = model.predict_proba(X_test)

print (metrics.accuracy_score(y_test, predicted))

print (metrics.confusion_matrix(y_test, predicted))

print( metrics.classification_report(y_test, predicted))
model = LogisticRegression()

model = model.fit(Train_x, Train_y)



# check the accuracy on the training set

model.score(Train_x, Train_y)
predicted2 = model.predict(Test_x)

probs2 = model.predict_proba(Test_x)

print (metrics.accuracy_score(Test_y, predicted2))

print (metrics.confusion_matrix(Test_y, predicted2))

print( metrics.classification_report(Test_y, predicted2))
from sklearn.svm import LinearSVC

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(Train_x, Train_y)

# check the accuracy on the full training set

model.score(Train_x, Train_y)
# check the accuracy on the full test set

predictedsvc = clf.predict(Test_x)

print (metrics.accuracy_score(Test_y, predictedsvc))

print (metrics.confusion_matrix(Test_y, predictedsvc))

print( metrics.classification_report(Test_y, predictedsvc))