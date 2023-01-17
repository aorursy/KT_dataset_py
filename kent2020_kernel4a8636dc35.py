import numpy as np

from scipy.io import loadmat

from pandas import DataFrame,concat,set_option

from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib import pyplot as plt

from pywt import swt,iswt

from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cdist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM,Bidirectional

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_squared_error

from numpy import concatenate

from math import sqrt

set_option('display.unicode.ambiguous_as_wide', True)

set_option('display.unicode.east_asian_width', True)

set_option('display.width', 180) # 设置打印宽度(**重要**)
m = loadmat('../input/trafficdata/corrected_data.mat') #载入MATLAB数据集



#分别读取4条公路流量数据

df_A1 = DataFrame(m['A1_10min'],columns=['A1_10min'])

df_A2 = DataFrame(m['A2_10min'],columns=['A2_10min'])

df_A4 = DataFrame(m['A4_10min'],columns=['A4_10min'])

df_A8 = DataFrame(m['A8_10min'],columns=['A8_10min'])





fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7, ax8) = plt.subplots(nrows=8, figsize=(22, 26))

plot_acf(df_A1, lags=5000,ax=ax1)

plot_acf(df_A2, lags=5000,ax=ax2)

plot_acf(df_A4, lags=5000,ax=ax3)

plot_acf(df_A8, lags=5000,ax=ax4)

plot_pacf(df_A1, lags=144,ax=ax5, method='ywmle')

plot_pacf(df_A2, lags=144,ax=ax6, method='ywmle')

plot_pacf(df_A4, lags=144,ax=ax7, method='ywmle')

plot_pacf(df_A8, lags=144,ax=ax8, method='ywmle')

ax1.legend(["A1"])

ax2.legend(["A2"])

ax3.legend(["A4"])

ax4.legend(["A8"])

ax5.legend(["A1"])

ax6.legend(["A2"])

ax7.legend(["A4"])

ax8.legend(["A8"])

#plt.tight_layout()

plt.show()
import numpy as np

from scipy.io import loadmat

from pandas import DataFrame,concat

from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib import pyplot as plt

from pywt import swt,iswt

from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cdist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM,Bidirectional

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_squared_error

from numpy import concatenate

from math import sqrt

np.random.seed(8601)

#####################################################################################################

#定义交通流预测函数

def traffic_forcast(df_data,N,strings):

    y = df_data.values.T #转置

    #print('原始序列长度: %.d' % len(df_data.values))#序列长度

    #print('原始序列数据: ',y)

    a = np.zeros(shape=(len(df_data.values),)) #生成0流量数组



    ca=swt(y[0],'db1',level =1) #小波分离低频和高频流量数据

    ya=iswt([ca[0][0],a],'db1') #还原低频流量数据

    yd=iswt([a,ca[0][1]],'db1') #还原高频流量数据

    

    

    #print('低频流量数据: ',ya)

    #print('高频流量数据: ',yd)

    ya= DataFrame(ya)

    yd= DataFrame(yd)



    #监督值为['原序列']；输入数据为['t-1序列','t-1低频序列','t-1高频频序列','t-2序列','t-2低频序列','t-2高频序列']

    df_dataset = concat([df_data.shift(-3),df_data.shift(-2),ya.shift(-2),yd.shift(-2),df_data.shift(-1),ya.shift(-1),yd.shift(-1)],axis=1)

    df_dataset.columns = ['原序列','t-1序列','t-1低频序列','t-1高频频序列','t-2序列','t-2低频序列','t-2高频序列']

#    df_dataset = concat([df_data.shift(-3),df_data.shift(-2),ya.shift(-2),yd.shift(-2),df_data.shift(-1),ya.shift(-1),yd.shift(-1),df_data,ya,yd],axis=1)

#    df_dataset.columns = ['1','2','3','4','5','6','7','8','9','10']  



######################################################################################################################

#    df_dataset = df_dataset.reindex(columns= df_dataset.columns.insert(7,'周期'))  

#    for i in range(0,len(df_dataset)):

#        if i%144 == 0: 

#            j = 0

#            df_dataset['周期'][i] = j

#        else:

#            j += 1

#            df_dataset['周期'][i] = j

#    print(df_dataset['周期'].values[142:147])





#    df_dataset = df_dataset.reindex(columns= df_dataset.columns.insert(6,'history1'))  

#    for i in range(0,len(df_dataset)):

#        if i < 144:df_dataset['history1'][i] = df_dataset['df_data-2'][i]

#        else:df_dataset['history1'][i] = df_dataset['df_data-3'][i-144]



#    df_dataset = df_dataset.reindex(columns= df_dataset.columns.insert(7,'history2'))  

#    for i in range(0,len(df_dataset)):

#        if i < 144:df_dataset['history2'][i] = df_dataset['df_data-2'][i]

#        else:           

#            distA = cdist(df_dataset.values[:i,1:6],np.array([df_dataset.values[:i+1,1:6][i]]),metric='euclidean')

#            df_dataset['history2'][i]=df_dataset.values[:i,0][np.argmin(distA)]

######################################################################################################################               

        

            

    df_dataset.dropna(inplace=True)#去掉空项

    print(df_dataset)

    print('输入数据为t-1序列,t-1低频序列,t-1高频频序列,t-2序列,t-2低频序列,t-2高频序列；预测t时刻序列，监督值为原序列')

    

    #归一化数据

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(df_dataset.values)



    #分割训练和测试数据集

    data_train = scaled[:N,:]

    data_test = scaled[N:,:]



    data_trainx, data_trainy = data_train[:,1:], data_train[:,0]

    print('归一化的输入值：')

    print(data_trainx)

    data_testx, data_testy = data_test[:,1:], data_test[:,0]

    print('归一化的监督值：')

    print(data_trainy)

    # reshape input to be 3D [samples, timesteps, features]

    data_trainx = data_trainx.reshape((data_trainx.shape[0], 1, data_trainx.shape[1]))

    data_testx = data_testx.reshape((data_testx.shape[0], 1, data_testx.shape[1]))



    # design network

    model = Sequential()

#    model.add(LSTM(32, input_shape=(data_trainx.shape[1], data_trainx.shape[2])))

    #Dropout(0.2)



    model.add(LSTM(64, input_shape=(data_trainx.shape[1], data_trainx.shape[2]),return_sequences=True))

#    model.add(LSTM(10,return_sequences=True))

    model.add(LSTM(4))

    model.add(Dense(1))



    #设置指数下降学习率

    initial_learning_rate = 0.1

    lr_schedule = ExponentialDecay(

        initial_learning_rate,

        decay_steps=80,

        decay_rate=0.96,

        staircase=True)

    model.compile(optimizer=Adam(learning_rate=lr_schedule),loss='mae')

    

    # fit network

    print('Start training')

    history = model.fit(data_trainx, data_trainy, epochs=80, batch_size=13, validation_data=(data_testx, data_testy), verbose=2, shuffle=True)

    # plot history

    plt.plot(history.history['loss'], label='train')

    plt.plot(history.history['val_loss'], label='test')

    plt.title("Loss")

    plt.legend()

    plt.show()



    # make a prediction

    yhat = model.predict(data_testx)

    test_X = data_testx.reshape((data_testx.shape[0], data_testx.shape[2]))

    # invert scaling for forecast

    inv_yhat = concatenate((yhat, test_X), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)

    #print(inv_yhat)

    #print(df_dataset.values[N:,:])

    inv_yhat = inv_yhat[:,0]

    inv_y = df_dataset['原序列'][N:].values

    

    #画出原始序列和预测序列

    fig=plt.figure(figsize=(22,4))#设置画布

    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(inv_y,color="r", label='origin')

    ax1.plot(inv_yhat,color="b", label='perdiction')

    plt.title("Compare")

    plt.legend()

    plt.show()



    #定义MAPE

    def mape(y_true, y_pred):

        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100



    mape = mape(inv_y, inv_yhat)# calculate MAPE

    print(strings +' Test MAPE: %.3f' % mape) 

    

    rmse = sqrt(mean_squared_error(inv_y, inv_yhat)) # calculate RMSE

    print(strings +' Test RMSE: %.3f' % rmse)

    

#################################################################################

m = loadmat('../input/trafficdata/corrected_data.mat') #载入MATLAB数据集



#分别读取4条公路流量数据

df_A1 = DataFrame(m['A1_10min'],columns=['A1_10min'])

df_A2 = DataFrame(m['A2_10min'],columns=['A2_10min'])

df_A4 = DataFrame(m['A4_10min'],columns=['A4_10min'])

df_A8 = DataFrame(m['A8_10min'],columns=['A8_10min'])



#输入序列做预测

#df_data = df_A1 #读取其中一条公路流量数据

N = 3000 #定义数据集训练和预测的分割点    

#traffic_forcast(df_A1,N,"A1") #训练及预测

#traffic_forcast(df_A2,N,"A2") 

#traffic_forcast(df_A4,N,"A4") 

traffic_forcast(df_A8,N,"A8") 