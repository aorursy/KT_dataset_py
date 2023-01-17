# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/kddcup.data"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/kddcup.data/kddcup.data')

df.shape
df.columns =["duration","protocol_type","service","flag","src_bytes",

    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",

    "logged_in","num_compromised","root_shell","su_attempted","num_root",

    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",

    "is_host_login","is_guest_login","count","srv_count","serror_rate",

    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",

    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",

    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",

    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",

    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
df.shape
df.head()
newlabeldf=df['label'].replace({ 'normal.' : 0, 'neptune.' : 1 ,'back.': 1, 'land.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1,'mailbomb.': 1, 'apache2.': 1, 'processtable.': 1, 'udpstorm.': 1, 'worm.': 1,

                           'ipsweep.' : 2,'nmap.' : 2,'portsweep.' : 2,'satan.' : 2,'mscan.' : 2,'saint.' : 2

                           ,'ftp_write.': 3,'guess_passwd.': 3,'imap.': 3,'multihop.': 3,'phf.': 3,'spy.': 3,'warezclient.': 3,'warezmaster.': 3,'sendmail.': 3,'named.': 3,'snmpgetattack.': 3,'snmpguess.': 3,'xlock.': 3,'xsnoop.': 3,'httptunnel.': 3,

                           'buffer_overflow.': 4,'loadmodule.': 4,'perl.': 4,'rootkit.': 4,'ps.': 4,'sqlattack.': 4,'xterm.': 4})

df['label'] = newlabeldf

#df['target'] = newlabeldf

print(df['label'].head())
df.info()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()

df['protocol_type'] = le.fit_transform(df['protocol_type'])

df['service']= le.fit_transform(df['service'])

df['flag'] = le.fit_transform(df['flag'])
X = df.iloc[:,:41]

y = df.iloc[:,-1]
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 34,test_size = 0.3)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

model1.fit(X_train_scaled,y_train)
predict = model1.predict(X_test_scaled)
model1.score(X_test_scaled,y_test)
model1.score(X_train_scaled,y_train)
from sklearn.metrics import confusion_matrix,recall_score,precision_score

print((confusion_matrix(y_test,predict)))
from sklearn.metrics import classification_report

print(classification_report(y_test, predict))