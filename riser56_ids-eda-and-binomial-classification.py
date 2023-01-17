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
#The first ten rows of the loan Dataframe are displayed below:

IDS_df.head(10)
IDS_df.shape
IDS_df.info()
#The DataFrame is analysed using the below commands. 





IDS_df.describe()
for col in list(IDS_df.columns):

    if isinstance(IDS_df[col][1],str):

        print(col)

        print(IDS_df[col].unique())
#createing target for binomail classification

IDS_df["target_attack"]=IDS_df["attack"].apply(lambda x:0 if x =="normal" else 1)

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
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 10)

sns.countplot(x="attack",data=IDS_df,order = IDS_df['attack'].value_counts().index)

print("Frequency of Attacks ")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 10)

sns.countplot(x="attack_class",  data=IDS_df )

print("Frequency of Attacks by class ")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 10)

sns.countplot(hue="attack_class",  data=IDS_df ,x="is_guest_login")

print("Frequency of Attacks class in guest login")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 10)

sns.countplot(hue="attack_class",  data=IDS_df ,x="is_host_login")

print("Frequency of Attacks class in guest login")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 10)

sns.countplot(x="protocol_type",  data=IDS_df,hue="target_attack" )

print("Frequency of Attacks by protocol_type ")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(15, 30)

sns.countplot(y="service",  data=IDS_df[IDS_df["target_attack"]!=1] ,order = IDS_df[IDS_df["target_attack"]!=1]['service'].value_counts().index)

print("Frequency of Attacks by service ")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(20, 12)

sns.barplot(x="is_guest_login", y="attack", data=IDS_df )

print("Guest login attack distrbution")
IDS_df.info()
y=IDS_df[["attack","attack_class","target_attack"]]

x=IDS_df.drop(["attack","attack_class","target_attack"], 1)

y.head()
x.head()
dummy = pd.get_dummies(x[['protocol_type', 'service', 'flag']], drop_first=True)

dummy.head()
x = pd.concat([x, dummy], axis=1)

x = x.drop(['protocol_type', 'service', 'flag'], 1)

x.head()
corr = x.corr()

corr.style.background_gradient()
# considering only targets for binomial classification

y=y.drop(["attack","attack_class"], 1)

y.head()
from sklearn.preprocessing import StandardScaler

colnum=["duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot","num_failed_logins","num_compromised","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]



scaler = StandardScaler()



x[colnum] = scaler.fit_transform(x[colnum])



x.head()
# Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=100)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import cross_val_score

model = LogisticRegression()

model = model.fit(X_train, y_train)



# check the accuracy on the training set

model.score(X_train, y_train)
# predict class labels for the test set

predicted = model.predict(X_test)

probs = model.predict_proba(X_test)

print (metrics.accuracy_score(y_test, predicted))

print (metrics.roc_auc_score(y_test, probs[:, 1]))
print (metrics.confusion_matrix(y_test, predicted))

print( metrics.classification_report(y_test, predicted))
model.score(X_test, y_test)
model = LogisticRegression()

model = model.fit(x, y)

model.score(x, y)
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
for col in list(IDS_df_Test.columns):

    if isinstance(IDS_df_Test[col][1],str):

        print(col)

        print(IDS_df_Test[col].unique())
#createing target for binomail classification

IDS_df_Test["target_attack"]=IDS_df_Test["attack"].apply(lambda x:0 if x =="normal" else 1)

IDS_df_Test.head()

#createing targets for multiclassification classification

switcher={"neptune":"DoS","normal":"normal","saint":"Probe","mscan":"Probe",

"guess_passwd":"R2L","smurf":"DoS","apache2":"DoS","satan":"Probe","buffer_overflow":"U2R","back":"DoS",

"warezmaster":"R2L","snmpgetattack":"R2L","processtable":"DoS","pod":"DoS",

"httptunnel":"R2L","nmap":"Probe","ps":"U2R","snmpguess":"R2L","ipsweep":"Probe",

"mailbomb":"R2L","portsweep":"Probe","multihop":"R2L","named":"R2L","sendmail":"R2L","loadmodule":"U2R",

"xterm":"U2R","worm":"DoS","teardrop":"DoS","rootkit":"U2R","xlock":"R2L","perl":"U2R","land":"DoS","xsnoop":"R2L",

"sqlattack":"U2R","ftp_write":"R2L","imap":"R2L","udpstorm":"DoS","phf":"R2L"}

    

IDS_df_Test["attack_class"]=IDS_df_Test["attack"].apply(lambda x:switcher.get(x, "Invalid"))

IDS_df_Test.head()
dummy = pd.get_dummies(IDS_df_Test[['protocol_type', 'service', 'flag']], drop_first=True)

dummy.head()

Test_df = pd.concat([IDS_df_Test, dummy], axis=1)

Test_df = Test_df.drop(['protocol_type', 'service', 'flag'], 1)

Test_df.head()
Test_y=Test_df[["target_attack"]]

Test_x=Test_df.drop(["attack","attack_class","target_attack"], 1)

Test_y.head()




Test_x[colnum] = scaler.transform(Test_x[colnum])



Test_x.head()
set(list(X_train.columns)).difference(set(list(Test_x.columns)))
Test_x['service_aol'] =  [0] * len(Test_x)

Test_x['service_harvest']=  [0] * len(Test_x)

Test_x['service_http_2784'] =  [0] * len(Test_x)

Test_x['service_http_8001'] =  [0] * len(Test_x)

Test_x['service_red_i'] =  [0] * len(Test_x)

Test_x['service_urh_i'] =  [0] * len(Test_x)
Test_x.head()
set(list(X_train.columns)).difference(set(list(Test_x.columns)))
predicted2 = model.predict(Test_x)

probs2 = model.predict_proba(Test_x)

print (metrics.accuracy_score(Test_y, predicted2))

print (metrics.roc_auc_score(Test_y, probs2[:, 1]))
print (metrics.confusion_matrix(Test_y, predicted2))

print( metrics.classification_report(Test_y, predicted2))
from sklearn.svm import LinearSVC

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(x, y) 
predictedsvc = clf.predict(Test_x)

print (metrics.accuracy_score(Test_y, predictedsvc))

print (metrics.confusion_matrix(Test_y, predictedsvc))

print( metrics.classification_report(Test_y, predictedsvc))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import auc

from sklearn.metrics import roc_curve
roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

roc
##Computing false and true positive rates

from sklearn import metrics

fpr, tpr,_=metrics.roc_curve(Test_y,predicted2,drop_intermediate=False)



import matplotlib.pyplot as plt

plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()