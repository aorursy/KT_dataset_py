import numpy as np
import pandas as pd
benign=pd.read_csv('../input/nbaiot-dataset/5.benign.csv')
g_c=pd.read_csv('../input/nbaiot-dataset/5.gafgyt.combo.csv')
g_j=pd.read_csv('../input/nbaiot-dataset/5.gafgyt.junk.csv')
g_s=pd.read_csv('../input/nbaiot-dataset/5.gafgyt.scan.csv')
g_t=pd.read_csv('../input/nbaiot-dataset/5.gafgyt.tcp.csv')
g_u=pd.read_csv('../input/nbaiot-dataset/5.gafgyt.udp.csv')
m_a=pd.read_csv('../input/nbaiot-dataset/5.mirai.ack.csv')
m_sc=pd.read_csv('../input/nbaiot-dataset/5.mirai.scan.csv')
m_sy=pd.read_csv('../input/nbaiot-dataset/5.mirai.syn.csv')
m_u=pd.read_csv('../input/nbaiot-dataset/5.mirai.udp.csv')
m_u_p=pd.read_csv('../input/nbaiot-dataset/5.mirai.udpplain.csv')

benign=benign.sample(frac=0.25,replace=False)
g_c=g_c.sample(frac=0.25,replace=False)
g_j=g_j.sample(frac=0.5,replace=False)
g_s=g_s.sample(frac=0.5,replace=False)
g_t=g_t.sample(frac=0.15,replace=False)
g_u=g_u.sample(frac=0.15,replace=False)
m_a=m_a.sample(frac=0.25,replace=False)
m_sc=m_sc.sample(frac=0.15,replace=False)
m_sy=m_sy.sample(frac=0.25,replace=False)
m_u=m_u.sample(frac=0.1,replace=False)
m_u_p=m_u_p.sample(frac=0.27,replace=False)

benign['type']='benign'
m_u['type']='mirai_udp'
g_c['type']='gafgyt_combo'
g_j['type']='gafgyt_junk'
g_s['type']='gafgyt_scan'
g_t['type']='gafgyt_tcp'
g_u['type']='gafgyt_udp'
m_a['type']='mirai_ack'
m_sc['type']='mirai_scan'
m_sy['type']='mirai_syn'
m_u_p['type']='mirai_udpplain'

data=pd.concat([benign,m_u,g_c,g_j,g_s,g_t,g_u,m_a,m_sc,m_sy,m_u_p],
               axis=0, sort=False, ignore_index=True)
#how many instances of each class
data.groupby('type')['type'].count()
#shuffle rows of dataframe 
sampler=np.random.permutation(len(data))
data=data.take(sampler)
data.head()
#dummy encode labels, store separately
labels_full=pd.get_dummies(data['type'], prefix='type')
labels_full.head()
#drop labels from training dataset
data=data.drop(columns='type')
data.head()
#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-df[col].mean())/df[col].std()

data_st=data.copy()
for i in (data_st.iloc[:,:-1].columns):
    standardize (data_st,i)

data_st.head()
#training data for the neural net
train_data_st=data_st.values
train_data_st
#labels for training
labels=labels_full.values
labels
#import libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping


# test/train split  25% test
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, labels, test_size=0.25, random_state=42)

#  create and fit model
model = Sequential()
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(40, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(labels.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
model.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
          callbacks=[monitor],verbose=2,epochs=500)
# metrics
pred_st = model.predict(x_test_st)
pred_st = np.argmax(pred_st,axis=1)
y_eval_st = np.argmax(y_test_st,axis=1)
score_st = metrics.accuracy_score(y_eval_st, pred_st)
print("accuracy: {}".format(score_st))
#second model
model2 = Sequential()
model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(72, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(1, kernel_initializer='normal'))
model2.add(Dense(labels.shape[1],activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
model2.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
          callbacks=[monitor],verbose=2,epochs=100)
# metrics
pred_st = model2.predict(x_test_st)
pred_st = np.argmax(pred_st,axis=1)
y_eval_st = np.argmax(y_test_st,axis=1)
score_st = metrics.accuracy_score(y_eval_st, pred_st)
print("accuracy: {}".format(score_st))