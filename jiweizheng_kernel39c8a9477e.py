# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ds = pd.read_csv("/kaggle/input/result.csv", index_col="site_host")

int_cols = [col for col in ds.columns if ds[col].dtype in ["int64", "float64"]]
obj_cols = [col for col in new_ds.columns if new_ds[col].dtype == "object"]

int_ds = ds[int_cols].copy().fillna(0)
obj_ds = ds[obj_cols].copy().fillna("")

int_cols.extend(obj_cols)

left_ds = ds.copy().drop(columns=int_cols)

from sklearn.preprocessing import LabelEncoder
lable_enc = LabelEncoder()
obj_convert_ds = obj_ds.copy()
for col in obj_cols:
    obj_convert_ds[col] = lable_enc.fit_transform(obj_ds[col])

if len(left_ds.columns) == 0:
    new_ds = pd.concat([obj_convert_ds, int_ds], axis=1)
else:
    new_ds = pd.concat([obj_convert_ds, int_ds, left_ds], axis=1)

append_ds = pd.DataFrame()
for col in obj_ds.columns:
    append_ds[col + "_is_none"] = obj_ds[col].isnull()
    
new_ds = pd.concat([new_ds, append_ds], axis="columns")

def pearsonSimilar(inA,inB):  
    if len(inA)<3:  
        return 1.0  
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]  


def euclidSimilar(inA,inB):  
    return 1.0/(1.0+np.linalg.norm(np.array(inA)-np.array(inB)))


def cosSimilar(inA,inB):
	inA=np.mat(inA)
	inB=np.mat(inB)
	num=float(inA*inB.T)
	denom=la.norm(inA)*la.norm(inB)
	return 0.5+0.5*(num/denom)

len(new_ds)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_array = new_ds.to_numpy().tolist()
indexs = new_ds.index

result_df = pd.DataFrame()
for index in range(len(data_array)):
    result = list()
    compare_x = data_array[index]
    for line in data_array:
        result.append(pearsonSimilar(line, compare_x))
    result_df[indexs[index]] = result

result_df.index = new_ds.index

result_df.to_csv("/kaggle/working/site_pearson_result.csv")
result_df.head()

from numpy import linalg as la 

data_array = new_ds.to_numpy().tolist()
indexs = new_ds.index

result_df = pd.DataFrame()
for index in range(len(data_array)):
    result = list()
    compare_x = data_array[index]
    for line in data_array:
        result.append(euclidSimilar(line, compare_x))
    result_df[indexs[index]] = result

result_df.index = new_ds.index

result_df.to_csv("/kaggle/working/site_euclid_result.csv")
result_df.head()

data_array = new_ds.to_numpy().tolist()
indexs = new_ds.index

result_df = pd.DataFrame()
for index in range(len(data_array)):
    result = list()
    compare_x = data_array[index]
    for line in data_array:
        result.append(cosSimilar(line, compare_x))
    result_df[indexs[index]] = result

result_df.index = new_ds.index

result_df.to_csv("/kaggle/working/site_cos_result.csv")
result_df.head()
import os
os.remove("/kaggle/working/site_result.csv")