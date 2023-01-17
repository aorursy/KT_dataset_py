from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections

from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA

# Number of cores to use to perform parallel fitting of the forest model
n_jobs = -1
# read in data
countries = pd.read_csv('../input/countries.csv')
age_gender = pd.read_csv('../input/age_gender_bkts.csv')
train_users = pd.read_csv('../input/train_users.csv')
sessions = pd.read_csv('../input/sessions.csv')
# session features to create: session count, total session duration, median session duration,
# mean session duration, action count, action_type count, action_detail count, 
pd.unique(sessions['device_type'])
pd.unique(sessions['action'])
pd.unique(sessions['action_detail'])
sessions.head()
sessions_grouped = sessions.groupby(['user_id','action'])
out = sessions_grouped['action_type'].count()
out.head(50)
features_header = ['user_id','session_count', 'total_session_duration', 'median_session_duration', 'mean_session_duration', 'action_count', 'action_type_count', 'action_detail_count']

