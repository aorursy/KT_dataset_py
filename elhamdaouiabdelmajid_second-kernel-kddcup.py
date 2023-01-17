!pip install sklearn_evaluation

!pip install xgboost

!pip install lightgbm



#=====================================#
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt # plotting

from sklearn_evaluation import plot

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split,GridSearchCV



from sklearn import preprocessing

from sklearn.externals import joblib





%matplotlib inline

#=====================================#
cols = """

    duration,

protocol_type,

service,

flag,

src_bytes,

dst_bytes,

land,

wrong_fragment,

urgent,

hot,

num_failed_logins,

logged_in,

num_compromised,

root_shell,

su_attempted,

num_root,

num_file_creations,

num_shells,

num_access_files,

num_outbound_cmds,

is_host_login,

is_guest_login,

count,

srv_count,

serror_rate,

srv_serror_rate,

rerror_rate,

srv_rerror_rate,

same_srv_rate,

diff_srv_rate,

srv_diff_host_rate,

dst_host_count,

dst_host_srv_count,

dst_host_same_srv_rate,

dst_host_diff_srv_rate,

dst_host_same_src_port_rate,

dst_host_srv_diff_host_rate,

dst_host_serror_rate,

dst_host_srv_serror_rate,

dst_host_rerror_rate,

dst_host_srv_rerror_rate"""

cols = [c.strip() for c in cols.split(",") if c.strip()]

cols.append('target')

#print(len(cols))



#=====================================#
attacks_type = {

'normal': 'normal',

'back': 'dos',

'buffer_overflow': 'u2r',

'ftp_write': 'r2l',

'guess_passwd': 'r2l',

'imap': 'r2l',

'ipsweep': 'probe',

'land': 'dos',

'loadmodule': 'u2r',

'multihop': 'r2l',

'neptune': 'dos',

'nmap': 'probe',

'perl': 'u2r',

'phf': 'r2l',

'pod': 'dos',

'portsweep': 'probe',

'rootkit': 'u2r',

'satan': 'probe',

'smurf': 'dos',

'spy': 'r2l',

'teardrop': 'dos',

'warezclient': 'r2l',

'warezmaster': 'r2l',

    }



#=====================================#
hajar_to_cup = {

    'is_hot_login' : 'is_host_login',

'urg' : 'urgent',

'protocol' : 'protocol_type',

'count_sec' : 'count',

'srv_count_sec' : 'srv_count',

'serror_rate_sec' : 'serror_rate',

'srv_serror_rate_sec' : 'srv_serror_rate',

'rerror_rate_sec' : 'rerror_rate',

'srv_error_rate_sec' : 'srv_rerror_rate',

'same_srv_rate_sec' : 'same_srv_rate',

'diff_srv_rate_sec' : 'diff_srv_rate',

'srv_diff_host_rate_sec' : 'srv_diff_host_rate',

'count_100' : 'dst_host_count',

'srv_count_100' : 'dst_host_srv_count',

'same_srv_rate_100' : 'dst_host_same_srv_rate',

'diff_srv_rate_100' : 'dst_host_diff_srv_rate',

'same_src_port_rate_100' : 'dst_host_same_src_port_rate',

'srv_diff_host_rate_100' : 'dst_host_srv_diff_host_rate',

'serror_rate_100' : 'dst_host_serror_rate',

'srv_serror_rate_100' : 'dst_host_srv_serror_rate',

'rerror_rate_100' : 'dst_host_rerror_rate',

'srv_rerror_rate_100' : 'dst_host_srv_rerror_rate',

}



#=====================================#
selcted_features  = ['duration', 'protocol_type', 'flag', 'src_bytes', 'dst_bytes', 'hot',

       'logged_in', 'num_compromised', 'count', 'srv_count', 'serror_rate',

       'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count',

       'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',

       'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',

       'dst_host_serror_rate']

#=====================================#
needed_cols_dump = []

for l in selcted_features:

    if l in hajar_to_cup.values():

        for k, v in hajar_to_cup.items():

            if v == l:

                needed_cols_dump.append(k)

    else:

        needed_cols_dump.append(l)

print(len(needed_cols_dump), len(selcted_features))

print(needed_cols_dump)
def standardize_columns(df, cols_map=hajar_to_cup):

    """

    1- Delete the 'service' column.

    2- Verify if TCPDUMP columns exists, then they will renamed

    """

    if 'service' in df.columns:

        df = df.drop(['service'], axis = 1)

    df = df.rename(columns = cols_map)

    return df



def do_what_we_want(X, 

                    scaler_1, 

                    le_X_cols, 

                    selcted_features, 

                    map_cols,

                    rdf_clf,

                    xgb_clf,

                    PathX=False):

    if PathX:

        X = pd.read_csv(PathX, names=cols, nrows=30000)

    X = standardize_columns(X, cols_map=map_cols) # Rename the columns, and delete the 

    X[['dst_bytes','src_bytes']] = scaler_1.fit_transform(X[['dst_bytes','src_bytes']])

    X = X[selcted_features]

    for c in X.columns:

        if str(X[c].dtype) == 'object': 

            le_X = le_X_cols[c]

            X[c] = le_X.transform(X[c])

            

    #====

    res = {

        'rd_prd_prb': rdf_clf.predict_proba(X),

        'rd_prd': rdf_clf.predict(X),

        'xgb_prd_prb': xgb_clf.predict_proba(X),

        'xgb_prd': xgb_clf.predict(X),

        

    }

    

    return res



#=====================================#
scaler_1 = joblib.load('../input/first-start-kernel/scaler_1.pkl') 

le_X_cols = joblib.load('../input/first-start-kernel/le_X_cols.pkl') 

le_y = joblib.load('../input/first-start-kernel/le_y.pkl') 

xgb_clf = joblib.load('../input/first-start-kernel/xgboost_classifier.pkl') 

rdf_clf = joblib.load('../input/first-start-kernel/random_forest_classifier.pkl') 



#=====================================#
X = pd.read_csv("../input/kdd-cup-1999-data/kddcup.data_10_percent/kddcup.data_10_percent", names=cols, nrows=100000)

Y = X.target.apply(lambda r: attacks_type[r[:-1]])



#=====================================#
X.tail(5)
print(np.unique(Y))

Y = le_y.transform(Y.values)

print(np.unique(Y))

print(le_y.inverse_transform(Y))

#=====================================#
res = do_what_we_want(X, 

                    scaler_1, 

                    le_X_cols, 

                    selcted_features, 

                    hajar_to_cup,

                    rdf_clf,

                    xgb_clf,

                    PathX=False)

res.keys()

#=====================================#
# print(len(res['rd_prd']), np.unique(res['rd_prd']))

# sum(np.argmax(res['rd_prd_prb'], axis=1) == res['rd_prd'])

# type(res['rd_prd_prb'])
atks = ['dos', 'normal', 'probe', 'r2l', 'u2r']

rd_prd_df = pd.DataFrame(data=res['rd_prd_prb'])

rd_prd_df= rd_prd_df.rename(columns = {l:'rd_'+atks[l] for l in range(len(atks))})

xg_prd_df = pd.DataFrame(data=res['xgb_prd_prb'])

xg_prd_df= xg_prd_df.rename(columns = {l:'xg_'+atks[l] for l in range(len(atks))})



df = pd.concat([rd_prd_df, xg_prd_df], axis=1)

df.head()

#=====================================#
params={"C":np.logspace(-7,7,7), "penalty":["l2"], "multi_class":['auto','ovr']}

lg = LogisticRegression(C=4.5, random_state = 42, multi_class = 'ovr', solver = 'lbfgs', max_iter = 1000)

clf = GridSearchCV(lg, params, cv=3)

clf.fit(df[:20000], Y[:20000])

print("Train score is:", clf.score(df[:20000], Y[:20000]))

print("Test score id:",clf.score(df[20000:], Y[20000:]))# New data, not included in Training data

#=====================================#
tcpdump_cols ="num_conn, startTimet, orig_pt, resp_pt, orig_ht, resp_ht, duration, protocol, resp_pt_2, flag, src_bytes, dst_bytes, land, wrong_fragment, urg, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_hot_login, is_guest_login, count_sec, srv_count_sec, serror_rate_sec, srv_serror_rate_sec, rerror_rate_sec, srv_error_rate_sec, same_srv_rate_sec, diff_srv_rate_sec, srv_diff_host_rate_sec, count_100, srv_count_100, same_srv_rate_100, diff_srv_rate_100, same_src_port_rate_100, srv_diff_host_rate_100, serror_rate_100, srv_serror_rate_100, rerror_rate_100, srv_rerror_rate_100"

tcpdump_cols = tcpdump_cols.split(", ")

needed_cols_dump = ['duration', 'protocol', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'logged_in', 'num_compromised', 'count_sec', 'srv_count_sec', 'serror_rate_sec', 'diff_srv_rate_sec', 'count_100', 'srv_count_100', 'same_srv_rate_100', 'diff_srv_rate_100', 'same_src_port_rate_100', 'srv_diff_host_rate_100', 'serror_rate_100']



TCP_DUMP = pd.read_csv("../input/tcpdumpdatakddcup/tcpdump.csv", sep=' ', lineterminator='\n', names=tcpdump_cols)

print(TCP_DUMP.shape)

TCP_DUMP.head()
res = do_what_we_want(TCP_DUMP, 

                    scaler_1, 

                    le_X_cols, 

                    selcted_features, 

                    hajar_to_cup,

                    rdf_clf,

                    xgb_clf,

                    PathX=False)

df = pd.DataFrame(index=TCP_DUMP.index)

df['typeAttack'] = le_y.inverse_transform(res['rd_prd'])

df['isAnomaly'] = (df['typeAttack']!='normal').astype('int')

print("Nb anomalies  is :",sum(df['isAnomaly']))

print(df['typeAttack'].value_counts())

df[1740:1750]