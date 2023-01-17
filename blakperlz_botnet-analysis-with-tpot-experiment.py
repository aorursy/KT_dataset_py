# import the basics
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import TPOT and sklearn 
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
#Raw data from the first file
data = pd.read_csv('../input/unsw-nb15/UNSW-NB15_1.csv')
#Raw data from the first file
data = pd.read_csv('../input/unsw-nb15/UNSW-NB15_1.csv')
#sample top 5
data.head(5)
#Well the data has no headers....we can find what they 'should be' via the features file.
#then lets reload the data
# and we'll make two copies of it in case we want to experiment later
data2 = data = pd.read_csv('../input/unsw-nb15/UNSW-NB15_1.csv', header = None, names = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'])
#make a data frame
df = pd.DataFrame(data)
#Double check results
data.head(5)
#What were the column names again?
df.columns
# This is a noisy data set.  Traditional netflow is a lot simpler.  
# Let's make a smaller data set....if you don't know what these are...check out the data dictionary '....features.csv'
# Make a new df so there is no overwrite
features = df[["sport","dsport","proto","Dpkts", "Spkts","Label"]]
#check features
features.head(5)
#data types
features.dtypes
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

# are there any null values?
features2.isnull().sum()
#drop all rows with null values
#make a new variable so you can trace back your work when troubleshooting
features3 = features2.dropna(how='any',axis=0) 
#Validate this has been corrected
features3.isnull().sum()

# import the libraries
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

# Count the labels
features3['Label'].value_counts()
# double check there are no null values
pd.isnull(features3).any()
# Make a new Label variable
Label = features3['Label'].values
# create indices by spliting the data
from sklearn.model_selection import train_test_split
training_indices, validation_indices = training_indices, testing_indices = train_test_split(features3.index,
                                                                                            stratify = Label,
                                                                                            train_size=0.75, test_size=0.25)
#Test the size, aka is it what you were expecting
training_indices.size, validation_indices.size
from tpot import TPOTClassifier
from tpot import TPOTRegressor

tpot = TPOTClassifier(generations=5,verbosity=2)

tpot.fit(features3.drop('Label',axis=1).loc[training_indices].values,
         features3.loc[training_indices,'Label'].values)
tpot.score(tele.drop('class',axis=1).loc[validation_indices].values,
           tele.loc[validation_indices, 'class'].values)


