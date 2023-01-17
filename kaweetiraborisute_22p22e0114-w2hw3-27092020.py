
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

training = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
training.head()
training.info()
pd.isnull(training).sum()
pick_feature = ['neighbourhood_group','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','reviews_per_month']

training_p1 = training[pick_feature]
training_p1.head()
# ตรวจสอบข้อมูล Missing
pd.isnull(training_p1).sum()
training_p1["reviews_per_month"].fillna(training["reviews_per_month"].median(), inplace = True)
pd.isnull(training_p1).sum()
training_p1.head()

neighbourhood_group = pd.unique(training_p1.neighbourhood_group).tolist()
room_type = pd.unique(training_p1.room_type).tolist()
def transform_one_hot_encoding_for_nyc(df):
    dummies_neighbourhood_group = pd.get_dummies(df["neighbourhood_group"],drop_first = True) # Drop_First กำหนดเพื่อป้องกันปัญหา multicollinearity
    dummies_room_type  = pd.get_dummies(df["room_type"],drop_first = True)
    merged = pd.concat([df,dummies_neighbourhood_group,dummies_room_type],axis='columns')
    merged = merged.drop(["neighbourhood_group","room_type"],axis=1) # Drop unused columns
    return merged
train_p2 = transform_one_hot_encoding_for_nyc(training_p1)
train_p2.head()
from sklearn import preprocessing

def normalize_z_column(df,cols_list): # ฟังก์ชันสำหรับ normalize df columns ที่ต้องการ
    pre_process = preprocessing.PowerTransformer(method="yeo-johnson",standardize = True)
    temp1  = pre_process.fit_transform(df[cols_list])
    temp2 = pd.DataFrame(temp1,columns = cols_list)
    pre_merge = df.drop(cols_list,axis=1)
    merge = pd.concat([pre_merge,temp2],axis=1)


    return merge
cols = ['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month'] # Cols that we need to normalize

final_training = normalize_z_column(train_p2,cols)

final_training.head()
from scipy.cluster.hierarchy import complete
pick = final_training.sample(n=50) # ทำการสุ่มตัวอย่างมา 200 ตัวอย่างสำหรับการทำการแบ่งกลุ่ม
plink = complete(pick) # สร้าง complete point linkage
from scipy.cluster.hierarchy import  dendrogram
plt.figure(figsize=(10,10));
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel('Euclidean Distance')
plt.ylabel('Sample Index')
dendrogram(plink,orientation='right',count_sort=True);