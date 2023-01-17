import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math
#our dataset

df=pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
df.head()
type(df)
df.shape
df.describe()
df.head()
labels = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))   
df['fruit_name'].value_counts()
#taking random record and storing in xq

xq = df.sample()
# droping the xq from data using index value

df.drop(xq.index, inplace=True)
df.shape
xq_final = pd.DataFrame(xq[['mass', 'width', 'height', 'color_score']])
xq_final
# calculating ecludian distance

def cal_distance(x):      

    a = x.to_numpy()

    b = xq_final.to_numpy()    

    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))

    return distance
# calculating distance

df['distance'] = df[['mass', 'width', 'height', 'color_score']].apply(cal_distance, axis=1)
#sorting the values based on distance

df_sort = df.sort_values('distance',ascending=True)
# taking top 3 records because k is 3

df_after_sort = df_sort.head(3)
df_after_sort.reset_index()
df_after_sort.iloc[0]
count = [0 for i in range(0, len(df['fruit_label'].unique()))]

for xi in range(0, len(df_after_sort)):       

    if df_after_sort.iloc[xi]['fruit_label'] == 1:        

        count[0] = count[0]+1

    elif df_after_sort.iloc[xi]['fruit_label'] == 2:        

        count[1] = count[1]+1

    elif df_after_sort.iloc[xi]['fruit_label'] == 3:        

        count[2] = count[2]+1

    elif df_after_sort.iloc[xi]['fruit_label'] == 4:        

        count[3] = count[3]+1
def max_num_in_list_label(list):

    maxpos = list.index(max(list)) +1

    return labels[maxpos]
#getting the label and verifying with the class label in xq

if max_num_in_list_label(count) in xq.values:

    print("success")

else:

    print("not success")