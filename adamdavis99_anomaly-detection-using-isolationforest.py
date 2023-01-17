# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dtypes = {
"duration": np.int8,
"protocol_type": np.object,
"service": np.object,
"flag": np.object,
"src_bytes":  np.int8,
"dst_bytes":  np.int8,
"land": np.int8,
"wrong_fragment":  np.int8,
"urgent": np.int8,
"hot": np.int8,
"m_failed_logins":  np.int8,
"logged_in":  np.int8,
"num_compromised":  np.int8,
"root_shell":  np.int8,
"su_attempted":  np.int8,
"num_root": np.int8,
"num_file_creations":  np.int8,
"num_shells":  np.int8,
"num_access_files":  np.int8,
"num_outbound_cmds":  np.int8,
"is_host_login":  np.int8,
"is_guest_login":  np.int8,
"count": np.int8,
"srv_count":  np.int8,
"serror_rate": np.float16,
"srv_serror_rate": np.float16,
"rerror_rate": np.float16,
"srv_rerror_rate": np.float16,
"same_srv_rate": np.float16,
"diff_srv_rate": np.float16,
"srv_diff_host_rate": np.float16,
"dst_host_count":  np.int8,
"dst_host_srv_count":  np.int8,
"dst_host_same_srv_rate": np.float16,
"dst_host_diff_srv_rate": np.float16,
"dst_host_same_src_port_rate": np.float16,
"dst_host_srv_diff_host_rate": np.float16,
"dst_host_serror_rate": np.float16,
"dst_host_srv_serror_rate": np.float16,
"dst_host_rerror_rate": np.float16,
"dst_host_srv_rerror_rate": np.float16,
"label": np.object
}

columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","m_failed_logins",
"logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label"]

df = pd.read_csv("/kaggle/input/kdd-cup-1999-data/kddcup.data.corrected", sep=",", names=columns,  index_col=None)
df.head()
df.describe()
#filter out the data to only include data entries that involve an http attack, and drop the service column
df=df[df["service"]=="http"]
df=df.drop("service",axis=1)
columns.remove("service")
df.head()
df.shape
df["label"].value_counts()
for col in df.columns:
    if(df[col].dtypes=="object"):
        encoded=LabelEncoder()
        encoded.fit(df[col])
        df[col]=encoded.transform(df[col])
df.head()
df.shape
# let's now shuffle the values in df and create our own training testing and validaion datasets

for f in range(0,3):
    df=df.iloc[np.random.permutation(len(df))]

df2=df[:500000]
labels=df2["label"]
df_validate=df[500000:]
x_train,x_test,y_train,y_test=train_test_split(df2,labels,test_size=0.2,random_state=42)
x_val,y_val=df_validate,df_validate["label"]
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
#let's build our Isolation Forest model
model=IsolationForest(n_estimators=500,max_samples=256,contamination=0.1,random_state=42)
#check sklearn website for details of Isolation Forest

model.fit(x_train)
anomaly_scores=model.decision_function(x_val)
plt.figure(figsize=(15,10))
plt.hist(anomaly_scores,bins=100)
plt.xlabel('Average Path Lengths',fontsize=14)
plt.ylabel('Number of Data Points',fontsize=14)
plt.show()
from sklearn.metrics import roc_auc_score
anomalies=anomaly_scores>-0.19
matches=y_val==list(encoded.classes_).index("normal.")
auc=roc_auc_score(anomalies,matches)
print("AUC: {:.2%}".format(auc))
anomaly_scores_test=model.decision_function(x_test)
plt.figure(figsize=(15,10))
plt.hist(anomaly_scores_test,bins=100)
plt.xlabel('Average Path Lengths',fontsize=14)
plt.ylabel('Number of Data Points',fontsize=14)
plt.show()
from sklearn.metrics import roc_auc_score
anomalies=anomaly_scores_test>-0.19
matches=y_test==list(encoded.classes_).index("normal.")
auc=roc_auc_score(anomalies,matches)
print("AUC: {:.2%}".format(auc))
