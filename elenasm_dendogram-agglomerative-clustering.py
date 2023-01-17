

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

db = pd.read_csv('/kaggle/input/ecommerce-users-of-a-french-c2c-fashion-store/6M-0K-99K.users.dataset.public.csv')
db.head()

db1 = db.drop(db.columns[[0,1,2,3,14]], axis = 1)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

cat_vars = ['gender','countryCode' ]

for var in cat_vars:
    var_cat = db1[[var]] #must use double brakets to make sure i'm taking a dataframe !!!!!
    var_cat_encoded = ordinal_encoder.fit_transform(var_cat)
    var_cat_df = pd.DataFrame(var_cat_encoded)
    var_cat_df.columns = [var + '_new'] # !!!!!!!!
    db1 = db1.merge(var_cat_df, how = 'inner', left_index = True, right_index = True)
    print(db1.head())

db2 = db1.drop(['gender','countryCode'], axis = 1)
db2.head()
db3 = db2.drop(['hasAnyApp','hasAndroidApp', 'hasIosApp','hasProfilePicture'], axis = 1)
db3.head()
print(db3.shape)
db_final = db3.sample(frac = 0.3)
print(db_final.shape)
fig = plt.figure(figsize = (10, 7))

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
dendogram = sch.dendrogram(sch.linkage(db_final, method = 'ward'))