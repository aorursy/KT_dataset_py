# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
train = pd.read_csv("../input/train.csv",sep=",",index_col=0)
submit = pd.read_csv("../input/test.csv",sep=",",index_col=0)
full_data_filter = train.copy()
del full_data_filter['target']
	
# Compute full rows for training data
full_data_filter = full_data_filter.notnull().astype(int)
row_pct_complete = full_data_filter.sum(axis=1).value_counts()
full_rows_model_filter = full_data_filter.sum(axis=1)/131>0.9
empty_rows_model_filter = full_data_filter.sum(axis=1)/131<0.3
# compute full rows for submission data
submit_full_data_filter = submit.copy()
submit_full_data_filter = submit_full_data_filter.notnull().astype(int)
submit_row_pct_complete = submit_full_data_filter.sum(axis=1).value_counts()
submit_full_rows_model_filter = submit_full_data_filter.sum(axis=1)/131>0.9

cat_features_list = []

# Keep the full features
numeric_features = ['v10','v12','v14','v21','v34','v40','v50','v114']
# Full categorical features
# full features : v24, v47, v66, v71, v72, v74, v75, v79, v110, 
# features with a few missing values : v52: 3 , v91: 3, v107: 3, v112: 382
categorical_features = ['v24','v47','v52','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112']
cat_features_list.append(categorical_features)
# Almost full categorical features
# v3 : 3457 missing, v31 3457 missing
almost_full_categorical_features = ['v3','v31']
cat_features_list.append(almost_full_categorical_features)
# Half empty categorical features
# v30 : 60110 missing, v113 : 55304 missing
# TODO : check wether missing values in synch with numeric missing features
half_empty_categorical_features = ['v30','v113']
cat_features_list.append(half_empty_categorical_features)
# v56: 6882 missing, v125: 77 missing, card=90, v22 : 500 missing but caridnality = 18210 
high_cardinality_features = ['v56','v125','v22']
cat_features_list.append(high_cardinality_features)
# v38 full, v62 full, v129 : full
cat_num_features = ['v38','v62','v129']
cat_features_list.append(cat_num_features)

features_to_keep = set(numeric_features)
features_to_keep = features_to_keep.union(categorical_features)
features_to_keep = features_to_keep.union(cat_num_features)
features_to_keep = features_to_keep.union(almost_full_categorical_features)
features_to_keep = features_to_keep.union(high_cardinality_features)
features_to_keep = features_to_keep.union(half_empty_categorical_features)
features_to_keep = list(features_to_keep)

def rnd_inverse(x):
	"""
		round data to 3 digits then inverse 
	"""
	return 1/(.1+round(x,3))

data_X = train.copy()
del data_X['target']
for feat_list in cat_features_list:
	if set(feat_list) <= set(features_to_keep):
		for f in feat_list:
			data_X[f], indexer = pd.factorize(data_X[f])
all_features = data_X.columns
for f in all_features:
	if f not in cat_features_list:
		the_mean = data_X[f].mean()
		data_X[f].fillna(the_mean,inplace=True)
for f in numeric_features:
	data_X[f]=data_X[f].apply(rnd_inverse)
	
# Add completion rate feature
data_X["complete"] = full_data_filter.sum(axis=1)/131

data_X = data_X[full_rows_model_filter]
data_Y = train.loc[full_rows_model_filter,'target'].copy()
np.random.seed(57)
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(data_Y.values,n_folds=5)
from sklearn.ensemble import ExtraTreesClassifier
extc = ExtraTreesClassifier(n_estimators = 600,max_features=25,max_depth=34,
			random_state=57,criterion='entropy',min_samples_leaf=1,min_samples_split=2,n_jobs=-1)
clearance = ['v36','v61','v98','v104','v9','v49','v65','v109','v86','v120','v127','v102','v121','v27','v97','v51','v103','v111',
					'v125','v28','v39','v11','v33','v58','v124','v2','v122','v10','v7','v105','v45',
					'v94','v90','v119','v1','v100','v95','v93','v123','v25','v118','v43','v108','v48',
					'v31','v12','v21','v14','v64','v71','v126','v73','v110','v79']
select_X = data_X.drop(clearance,axis=1)
results = cross_val_score(extc, select_X, data_Y, scoring='log_loss', cv=skf, verbose=0, n_jobs=1)
print("EXTC results : %0.5f/%0.5f" %(np.mean(results),np.std(results)))