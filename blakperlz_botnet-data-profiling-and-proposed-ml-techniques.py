# EDA and Profiling the data with pandas_profiler
#Load Libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from pandas_profiling import ProfileReport
#Raw data from the first file
data = pd.read_csv('../input/unsw-nb15/UNSW-NB15_1.csv')
#quick info about the data
data.info
#that was hard to read....here is a prettier version, but missing some details in the middle
data.head(5)
#Well the data has no headers....we can find what they 'should be' via the features file.
#then lets reload the data
# and we'll make two copies of it in case we want to experiment later
data2 = data = pd.read_csv('../input/unsw-nb15/UNSW-NB15_1.csv', header = None, names = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'])
df = pd.DataFrame(data)
data.head(5)
#What were the column names again?
df.columns
#This is a noisy data set.  Traditional netflow is a lot simpler.  
#Let's make a smaller data set....if you don't know what these are...check out the data dictionary '....features.csv'

features = df[["sport","dsport","proto","Dpkts", "Spkts","Label"]]
features.head(5)
features.dtypes
features['sport'] = pd.to_numeric(features['sport'], errors='coerce')
#well shoot....now the coorect way is to do a .loc, but lets try a quicker route...
#copy/paste.  Note this is the the 'correct' way, but it works for now
features2=features.copy()
#Machines read numbers, so let's convert to numbers
#BTW when ran the first time sport and dsport were 'rejected', so this is a 'must'.
features2['sport'] = pd.to_numeric(features['sport'], errors='coerce')
features2['dsport'] = pd.to_numeric(features['sport'], errors='coerce')
#We can also do label encoding...
#Label encoding per: https://www.datacamp.com/community/tutorials/categorical-data
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
features2['proto'] = lb_make.fit_transform(features2['proto'])

#run the profiler
#https://github.com/pandas-profiling/pandas-profiling
profile = ProfileReport(features2, title = "Features to Evaluate Data Profile")
#Let's look at the data from within the notebook
profile.to_notebook_iframe()
features2.isnull().sum()
#drop all rows with null values
#make a new variable so you can trace back your work when troubleshooting
features3 = features2.dropna(how='any',axis=0) 
#Validate this has been corrected
features3.isnull().sum()

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
# Inspired by https://www.datacamp.com/community/tutorials/ensemble-learning-python
#Lets scale the data so all columns are relative to one another

scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(features3)
print(normalizedData)
#input variables = X and Y = Label.  There are 6 variables, and python starts at 0 
X = normalizedData[:,0:5]
Y = normalizedData[:,5]
# 10-fold cross-validation fold, then using decision tree clasifier with 100 trees
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
