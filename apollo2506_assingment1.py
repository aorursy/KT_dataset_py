import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score as DB
from sklearn.metrics import silhouette_score as SI
import os
import glob
import re
from openpyxl import load_workbook
# data = pd.DataFrame()
# file = '/kaggle/input/assingment-1-data/data/1.csv'

# di_index, si_index = [], []

# df = pd.read_csv(file,header=None)
# df = df.iloc[:,:-1]
# for i in range(2,11):
#     kmeans = KMeans(n_clusters=i,random_state=42).fit_predict(df)
#     di,si = DB(df,kmeans),SI(df,kmeans)
#     di_index.append(di)
#     si_index.append(si)
#     pass
# # print(si_index)
# # print(di_index)

# si_index.extend(di_index)
# print(si_index)

# data1 = pd.DataFrame([si_index])
# data = data.append(data1,ignore_index=True)
# data
base_path = '/kaggle/input/assingment-1-data/'
dirfiles = os.listdir(base_path+'data/')
dirfiles.sort(key=lambda f: int(re.sub('\D','',f)))

# for file in dirfiles:
#     print(file)
#     pass

sub = pd.DataFrame()

for file in dirfiles:
    di_index,si_index = [],[]
    df = pd.read_csv(base_path+'data/'+file)
    df = df.iloc[:,:-1]
    
    for i in range(2,11):
        kmeans = KMeans(n_clusters=i,random_state=42).fit_predict(df)
        di,si = DB(df,kmeans),SI(df,kmeans)
        di_index.append(di)
        si_index.append(si)
        pass
    
    si_index.extend(di_index)
    data = pd.DataFrame([si_index])
    sub = sub.append(data,ignore_index=True)
    pass

sub.to_excel('submission.xlsx',index=False)
print("Check submission.xlsx file")