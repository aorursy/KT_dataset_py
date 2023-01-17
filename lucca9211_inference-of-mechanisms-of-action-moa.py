# import warnings, sys

# warnings.filterwarnings("ignore")



# # Chris's RAPIDS dataset

# !cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

# !cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

# sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

# sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

# sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

# !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pickle
# Read the Dataset

# test = pd.read_csv("/kaggle/input/mechanisms-of-action-moa-eda/test_clean")

test = pd.read_csv("../input/lish-moa/test_features.csv")

target_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

sample = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
def clean_fn(data):

    data.loc[:, 'cp_type'] = data.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del data['sig_id']

    return data



test =  clean_fn(test)

from sklearn import preprocessing

x_test = test.copy()



quantile_transformer = preprocessing.QuantileTransformer(random_state=0)

X_test = quantile_transformer.fit_transform(x_test)
X_test.mean()
XX_test = preprocessing.scale(x_test,axis=0, with_mean=True, with_std=True, copy=True)
XX_test.mean()
import seaborn as sns

sns.distplot(XX_test);
sns.distplot(X_test);
sns.distplot(test);
qtest=pd.read_csv("../input/mechanisms-of-action-moa-eda/test_clean")

sns.distplot(qtest);
# First Five rows



test.head()
plt.plot(test.skew())
target_scored.sum()[1:].sort_values()
# # drop columns that have only one label for 1's in target

# copy_target = target_scored.copy()

# target_scored =target_scored.drop(['atp-sensitive_potassium_channel_antagonist',

#                     'erbb2_inhibitor'], axis=1)

# # Load from file

# with open(pkl_filename, 'rb') as file:

#     pickle_model = pickle.load(file)

#     pickle_model.predict(test)

from sklearn.kernel_approximation import Nystroem

kernel = Nystroem(kernel = 'rbf', n_components = 100, random_state = 0)



# test = kernel.fit_transform(test)

XX_test = kernel.fit_transform(XX_test)

# import cuml

# Select all columns from target(not id)

# select = target_scored.iloc[:,1:2]

MODEL_DIR = "../input/training-of-mechanisms-of-action-moa-eda/"

#select = target_scored.iloc[:,1:3]









preds = []

# for i in select:

for i, _col in enumerate(target_scored.columns):

    if _col != "sig_id":

        pkl_filename = str(MODEL_DIR)+"model"+str(i)+".pkl"

#         print(pkl_filename)

    

        with open(pkl_filename, 'rb') as file:

            pickle_model = pickle.load(file)

#             prediction = pickle_model.predict(test)

            

            prediction = pickle_model.predict(XX_test)

            preds.append(prediction)

        

            sample[_col]=(sum(preds)/len(preds))/5

        

        
sample.head()
sample.to_csv('submission.csv', index=False)