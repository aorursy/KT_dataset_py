import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

data = pd.read_csv('../input/network_data.csv')
data.head()
data['total_data'] = data['num_packets']*data['num_bytes']
data.head()
data.sort_values('start_time',inplace=True)
data.reset_index(drop=True,inplace=True)
data.head()

data['source_ip'].nunique()
data['source_port'].nunique()
data['destination_port'].nunique()
data['destination_ip'].nunique()
source_ip = data.groupby('source_ip')['source_port'].count()
source_ip.sort_values(ascending=False,inplace=True)
source_ip = pd.DataFrame(source_ip)
source_ip.columns = ['count']
source_ip.head()
source_ip.describe()
data['num_packets'].nunique()
data['num_bytes'].nunique()
data['total_data'].nunique()

source_ip['count'].hist()
destination_ip = pd.DataFrame(data.groupby('destination_ip')['source_port'].count().sort_values(ascending=False))
destination_ip.head()
#same case here as the source_ip one.
destination_ip.columns = ['count']
destination_ip['count'].hist()
data['total_data'].describe()
plt.hist(data['total_data'],color='g',range=(0,10000))
sorted_total_data = data['total_data'].sort_values(ascending=False)
sorted_total_data.reset_index(drop=True,inplace=True)
#something is happening when the total data size lies in [2000,4000]
data['num_bytes'].describe()
plt.hist(data['num_bytes'],color='g',range=(0,1000))
data['num_packets'].describe()
plt.hist(data['num_packets'],color='g',range=(0,100))
single_packet_data = data[data['num_packets'] == 1]['num_packets']
len(single_packet_data)
source_ip['count'][:5]
type(data['source_ip'][0])
ports_1 = data['destination_port'][data['source_ip']=='135.0777d.04511.237']
ports_1.describe()
packet_1 = data['num_packets'][data['source_ip']=='135.0777d.04511.237']
packet_1.describe()
plt.hist(packet_1,range=(10000,90000))
sns.kdeplot(packet_1[:10000])
sns.kdeplot(packet_1[10000:])
bytes_1 = data['num_bytes'][data['source_ip']=='135.0777d.04511.237']
bytes_1.describe()
#let's check total data
total_data_1  = data['total_data'][data['source_ip']=='135.0777d.04511.237']
total_data_1.describe()
ports_2 = data['destination_port'][data['source_ip']=='135.0777d.04511.232']
ports_2.describe()

ports_1_ = data['source_port'][data['destination_ip']=='135.0777d.04511.237']
ports_1_.describe()
ports_2_ = data['source_port'][data['destination_ip']=='135.0777d.04511.232']
ports_2_.describe()
data.columns

def convert_timestamp(time):
    return datetime.datetime.fromtimestamp(int(time))
def calculate_time_diff(source_ip):
    time_diff = []
    flag = 0
    for i in range(len(data)):
        if data['source_ip'][i] == source_ip:
            if flag == 0:
                t = data['start_time'][i]
                flag = 1
            else:
                diff = convert_timestamp(data['start_time'][i]) - convert_timestamp(t)
                t = data['start_time'][i]
                time_diff.append(diff)
    return time_diff
time_diff_1 = calculate_time_diff('135.0777d.04511.237')
time_diff_1.sort(reverse=True)
time_diff_1[:5]
count = 0
for i in range(len(time_diff_1)):
    if time_diff_1[i] == datetime.timedelta(0):
        count+=1
print('Total '+str(count)+' entries have ZERO time difference')
time_diff_2 = calculate_time_diff('135.0777d.04511.232')
count = 0
for i in range(len(time_diff_2)):
    if time_diff_2[i] == datetime.timedelta(0):
        count+=1
print('Total '+str(count)+' entries have ZERO time difference')
len(time_diff_2)
sns.jointplot(data['num_packets'],data['num_bytes'],color='g')

len(data[data['destination_port']==22])
time_diff = []
time_diff.append(0)
t = data['start_time'][0]
for i in range(1,len(data)):
    time_diff.append(data['start_time'][i] - t)
    t = data['start_time'][i]
time_diff = pd.Series(time_diff)
num_bytes = data['num_bytes']
num_packets = data['num_packets']
print(len(data))
print(len(data[data['source_port']==22]))
print(data['source_port'].nunique())
print(len(data))
print(len(data[data['destination_port']==22]))
print(data['destination_port'].nunique())
data.groupby('destination_port')['source_ip'].count().sort_values(ascending=False)[:10]
data.groupby('source_port')['source_ip'].count().sort_values(ascending=False)[:10]

source_port = pd.Series([1 if x==22 else 0 for x in data['source_port']])
destination_port = pd.Series([1 if x==22 else 0 for x in data['destination_port']])
type(source_port)
source_ip = [1 if x=='135.0777d.04511.237' or x=='135.0777d.04511.232' else 0 for x in data['source_ip']]
dest_ip = [1 if x=='135.0777d.04511.237' or x=='135.0777d.04511.232' else 0 for x in data['destination_ip']]
df = pd.DataFrame({'time_diff':time_diff,'num_bytes':num_bytes,'num_packets':num_packets,'source_port':source_port,
                   'destination_port':destination_port,'destination_ip':dest_ip,'source_ip':source_ip})
df.columns
df_with_dummy = pd.get_dummies(df,prefix=['dest_with','source_with','dest_ip','source_ip'],
                               columns=['destination_port','source_port','destination_ip','source_ip'])
df_with_dummy.head()
mx = MinMaxScaler()
df_with_dummy = mx.fit_transform(df_with_dummy)
df_with_dummy.shape
final_df = pd.DataFrame(df_with_dummy,columns=['num_bytes', 'num_packets', 'time_diff','dest_with_0' ,'dest_with_1',
                                               'source_with_0','source_with_1','dest_ip_0',
                                               'dest_ip_1' ,'source_ip_0 ','source_ip_1'])
final_df.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0,n_jobs=-1).fit(df_with_dummy)
new_vals = pd.Series(kmeans.predict(df_with_dummy))
print(new_vals.groupby(new_vals).count())
final_df.insert((final_df.shape[1]),'kmeans',new_vals)
final_df.head()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(final_df['num_bytes'],final_df['dest_ip_1'],c=new_vals,s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('num_bytes')
ax.set_ylabel('num_pack')
plt.colorbar(scatter)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(final_df['num_packets'],final_df['dest_ip_1'],c=new_vals,s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('num_packets')
ax.set_ylabel('dest_ip')
plt.colorbar(scatter)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(final_df['time_diff'],final_df['dest_ip_1'],c=new_vals,s=50)
ax.set_title('time_difference')
ax.set_xlabel('num_packets')
ax.set_ylabel('dest_ip_1')
plt.colorbar(scatter)
cluster_count = new_vals.groupby(new_vals).count()
rating = (cluster_count[0]+cluster_count[1])/len(final_df) * 10
cluster_count
rating




