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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#kaggre:https://www.kaggle.com/rasyidstat/transjakarta-bus-gps-data
#トランスジャカルタバスGPSデータのサンプル（14:00〜18:00（2019年11月26日）、約30秒の時間間隔）
#出典：https://www.transjakarta.co.id/

gps_data = pd.read_csv('/kaggle/input/transjakarta-bus-gps-data/transjakarta_gps.csv')
gps_data.head()
#欠損値のチェック
print(len(gps_data))
print(gps_data.isnull().sum())
#np.arrayの配列を作成
bus_trip_gps = gps_data[["bus_code", "trip_id","gps_datetime", "longitude", "latitude"]]
bus_gps = bus_trip_gps.values
#距離の変位を調査する

bus_gps_plus = np.vstack((np.zeros(2).reshape(1,2), bus_gps[:, 3:5]))
bus_gps_minus = np.vstack((bus_gps[:, 3:5], np.zeros(2).reshape(1,2)))
delta_bus_gps = bus_gps_plus - bus_gps_minus

delta_bus_gps = delta_bus_gps[1:,:]

#グラフ表示でしたとき、異常値に気付けるように単位をkmにする
delta_bus_gps*=110.94297

#距離のデータを作成
delta_distans = (np.square(delta_bus_gps[:,0:1]) + np.square(delta_bus_gps[:, 1:2]))**0.5

#時間の変化のデータを作成
bus_time_minus = pd.to_datetime(bus_gps[:-1,2:3].reshape(-1))
bus_time_plus = pd.to_datetime(bus_gps[1:,2:3].reshape(-1))
delta_time_gps = bus_time_plus - bus_time_minus
delta_time_gps = delta_time_gps/np.timedelta64(1,'s')
delta_time_gps = np.array(delta_time_gps, dtype='float')
delta_time_gps = np.append(delta_time_gps,0)
delta_time_gps = delta_time_gps.reshape(len(delta_time_gps),1)
#変化量、速度を時速になおす
delta_bus_gps /= (delta_time_gps + 1e-8)/(60*60)
delta_distans /= (delta_time_gps + 1e-8)/(60*60)
#bus_gpsに変位と距離を追加する
bus_gps = np.insert(bus_gps,[5] , delta_bus_gps, axis=1)
bus_gps = np.insert(bus_gps,[7] , delta_distans, axis=1)
bus_gps = np.insert(bus_gps,[3] , delta_time_gps, axis=1)
#5回の測定で一回の時系列データとする
#bus_codeとtrip_idで別々の時系列のデータとなるようにする。

gps_input_data=[]
gps_correct_data =[]

len_sequence = 5             # 時系列の長さ

bus_gps[:,0:2] = bus_gps[:,0:2].astype(np.str)
bus_code_list = np.unique(bus_gps[:,0:1])

for i in range(100): #range(len(bus_code_list)):
    bus_code = bus_code_list[i]
    bus_code_gps = bus_gps[np.any(bus_gps==bus_code,axis=1)]

    trip_id_list =np.unique(bus_code_gps[:,1:2])
    
    if i%100 == 0:
        print(i)
    
    for j in range(len(trip_id_list)):    
        trip_id =  trip_id_list[j]
        trip_id_gps = bus_code_gps[np.any(bus_code_gps[:,1:2]==str(trip_id),axis=1)]
                
        if len(trip_id_gps)>(len_sequence+2):
            for k in range(len(trip_id_gps)-(len_sequence+2)):
                gps_input_data.append(trip_id_gps[k:k+len_sequence,2:])
                gps_correct_data.append(trip_id_gps[k+len_sequence:k+(len_sequence+1),2:].reshape(7))
                
                
#インプット用のデータ                
#axis0 データの数、axis1 同一データ内の時系列位置 axis2 緯度,経度,時間,時間変化,x変位,y変位,時速
gps_input_data=np.array(gps_input_data) 

#正解用のデータ
#axis0 データの数、axis1 緯度,経度,時間,時間変化,x変位,y変位,時速
gps_correct_data =np.array(gps_correct_data)
print(len(gps_input_data))
print(len(gps_correct_data))
gps_input_data[1,:,:]
gps_correct_data[0]
#異常な値の削除
#正解の時速が200kmを超えているものを異常とみなす

high_distans_index0 = np.where(gps_input_data[:,:,-1]>200)
high_distans_index0 = high_distans_index0[0]

high_distans_index1 = np.where(gps_correct_data[:,-1]>200)
high_distans_index1 = high_distans_index1[0]

error_data_index=np.concatenate([high_distans_index0, high_distans_index1])
error_data_index=np.unique(error_data_index)

print(len(error_data_index))
print(len(gps_input_data))
gps_input_data = np.delete(gps_input_data,error_data_index,axis=0)
gps_correct_data = np.delete(gps_correct_data,error_data_index,axis=0)
print(len(gps_input_data))
#異常な値の削除
#移動距離が少なすぎるものを異常とみなす
#データの時系列内での（1個目と5個目）移動距離が0.01km以下のものを異常とする

index_destans0 = np.where((((gps_input_data[:,0,4]-gps_input_data[:,-1,4])**2 + (gps_input_data[:,0,5]-gps_input_data[:,-1,5])**2)**0.5) < 0.01)
index_destans0 = np.array(index_destans0)
index_destans0 = index_destans0.reshape(-1)

print(len(index_destans0))
print(len(gps_input_data))
gps_input_data = np.delete(gps_input_data,index_destans0,axis=0)
gps_correct_data = np.delete(gps_correct_data,index_destans0,axis=0)
print(len(gps_input_data))
#異常な値の削除
#時間の変位が5より小さい,100より大きいデータを異常とみなす

index_time_short0 = np.where(gps_input_data[:,:,1]<20)
index_time_short0 = index_time_short0[0]
index_time_short1 = np.where(gps_correct_data[:,1]<20)
index_time_short1 = index_time_short1[0]
index_time_short = np.concatenate([index_time_short0, index_time_short1])

index_time_long0 = np.where(gps_input_data[:,:,1]>40)
index_time_long0 = index_time_long0[0]
index_time_long1 = np.where(gps_correct_data[:,1]>40)
index_time_long1 = index_time_long1[0]
index_time_long = np.concatenate([index_time_long0, index_time_long1])

#index_time_short = np.concatenate([index_time_short+i for i in range(len_sequence+1)])
#index_time_long = np.concatenate([index_time_long+i for i in range(len_sequence+1)])

index_time_error = np.concatenate([index_time_long, index_time_short])
index_time_error = np.unique(index_time_error)

print(len(index_time_error))
print(len(gps_input_data))
gps_input_data = np.delete(gps_input_data,index_time_error,axis=0)
gps_correct_data = np.delete(gps_correct_data,index_time_error,axis=0)
print(len(gps_input_data))
plt.plot(gps_correct_data[:,1])#時間変化
plt.scatter(gps_correct_data[:,2],gps_correct_data[:,3],s=2)#位置
plt.scatter(gps_correct_data[:,4],gps_correct_data[:,5])#変位
plt.plot(gps_correct_data[:,6])#速度
#リカレントネットワークを用いて時系列の時速データから次の時速を予測する
#予測した値と正解値の解離が大きい地点を時間ごとに地図にプロット
#学習用、ネットワーク評価用、GPSプロット用にデータを分ける

#データ分割用のインデックスの準備
np.random.seed(0)
all_data_number = len(gps_correct_data)
data_index =np.arange(all_data_number)
np.random.shuffle(data_index)

#時速データの準備
X = gps_input_data[:,:,(1,2,3,6)]
t = gps_correct_data[:,(1,2,3,6)]

#正規化
for i in range(len(t[0,:])):
    Xt_min = np.min((X[:,:,i].min(), t[:,i].min()))
    Xt_max = np.max((X[:,:,i].max(), t[:,i].max()))
    X[:,:,i] = (X[:,:,i]-Xt_min)/Xt_max
    t[:,i] = (t[:,i]-Xt_min)/Xt_max
                   
#データの分割
train_data_number =(all_data_number*3)//5
plot_data_number = (all_data_number*1)//5

X_train = X[data_index[:train_data_number],:,:]
t_train = t[data_index[:train_data_number]]

X_test = X[data_index[train_data_number:(-plot_data_number)],:,:]
t_test = t[data_index[train_data_number:(-plot_data_number)]]

X_plot = X[data_index[(-plot_data_number):],:,:]
t_plot = t[data_index[(-plot_data_number):]]
#ネットワークの定義
from keras.models import Sequential 
from keras.layers import Dense, SimpleRNN, GRU, LSTM
from keras.optimizers import RMSprop

input_dim = 4                # 入力データの次元数
output_dim = 1               # 出力データの次元数
num_hidden_units = 64        # 隠れ層のユニット数
batch_size = 100              # ミニバッチサイズ
num_of_training_epochs = 50   # 学習エポック数
#learning_rate = 0.001        # 学習率

model = Sequential()
model.add(LSTM(num_hidden_units,
              input_shape=(len_sequence,input_dim)))
model.add(Dense(output_dim))
model.compile(loss="mae", 
              optimizer=RMSprop())
model.summary()
history=model.fit(X_train,
                  t_train[:,-1],
                  epochs=num_of_training_epochs,
                  batch_size=batch_size,
                  validation_data=(X_test,t_test[:,-1]))
plt.figure()
plt.plot(range(num_of_training_epochs),history.history["loss"], 'bo', label='Training loss')
plt.plot(range(num_of_training_epochs),history.history["val_loss"], 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
 
# クラスタリング
data_number = 2000
# 正規化
#データ分割用のインデックスの準備
np.random.seed(0)
all_data_number = len(gps_correct_data)
data_index =np.arange(all_data_number)
np.random.shuffle(data_index)

#ts_dataset =(gps_input_data[:,:,6].reshape(all_data_number,len_sequence)-gps_correct_data[:,6].min())/gps_correct_data[:,6].max()

ts_dataset =X[:,3]

metric = 'dtw'
n_clusters = [n for n in range(2, 100)]
silhouette_data = []
for n in n_clusters: 
    # metricが「DTW」か「softdtw」なら異なるデータ数の時系列データでもOK
    km= TimeSeriesKMeans(n_clusters=n, metric=metric, verbose=False, random_state=1).fit(ts_dataset)
    print('クラスター数 ='+ str(n) + 'シルエット値 ='+ str(silhouette_score(ts_dataset, km.labels_, metric=metric)))
    silhouette_data.append(np.array([n, silhouette_score(ts_dataset, km.labels_, metric=metric)]))

silhouette_data=np.array(silhouette_data).reshape(48,2)
silhouette_max = silhouette_data[np.argmax(silhouette_data[:,1]),:]
plt.plot(silhouette_data[:,0], silhouette_data[:,1])
