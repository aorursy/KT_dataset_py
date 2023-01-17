# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Deep Learning
import tensorflow as tf
# load in data
data = pd.read_csv("../input/liverpool-ion-switching/train.csv")
test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
sample_sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv" ,dtype={'time': 'str'})
plt.figure(figsize = (16,8))
plt.plot(data[data['open_channels']==0]['signal'], color = 'c')
plt.plot(data[data['open_channels']==1]['signal'], color = 'b')
plt.plot(data[data['open_channels']==2]['signal'], color = 'r')
plt.plot(data[data['open_channels']==3]['signal'], color = 'k')
plt.plot(data[data['open_channels']==4]['signal'], color = 'g')
plt.plot(data[data['open_channels']==5]['signal'], color = 'y')
plt.plot(data[data['open_channels']==6]['signal'], color = 'c')
plt.plot(data[data['open_channels']==7]['signal'], color = 'm')
plt.plot(data[data['open_channels']==8]['signal'], color = 'b')
plt.plot(data[data['open_channels']==9]['signal'], color = 'r')
plt.plot(data[data['open_channels']==10]['signal'], color = 'g')
combined = pd.concat([data,test])
combined = combined.reset_index()
modified = []
for i in range(70):
    a = i*10 
    b = i*10 + 10
    temp = combined[(combined['time']>a)&(combined['time']<=b)]
    par = np.polyfit(temp['time'],temp['signal'],2)
    modified += (temp['signal'] - (par[0]*temp['time']**2 + par[1]*temp['time']**1 + par[2])).tolist()
combined['modi'] = modified
plt.figure(figsize = (16,8))
plt.plot(combined[combined['open_channels']==0]['modi'], color = 'c')
plt.plot(combined[combined['open_channels']==1]['modi'], color = 'b')
plt.plot(combined[combined['open_channels']==2]['modi'], color = 'r')
plt.plot(combined[combined['open_channels']==3]['modi'], color = 'k')
plt.plot(combined[combined['open_channels']==4]['modi'], color = 'g')
plt.plot(combined[combined['open_channels']==5]['modi'], color = 'y')
plt.plot(combined[combined['open_channels']==6]['modi'], color = 'c')
plt.plot(combined[combined['open_channels']==7]['modi'], color = 'm')
plt.plot(combined[combined['open_channels']==8]['modi'], color = 'b')
plt.plot(combined[combined['open_channels']==9]['modi'], color = 'r')
plt.plot(combined[combined['open_channels']==10]['modi'], color = 'g')
plt.figure(figsize = (16,8))
plt.plot(combined['modi'], color = 'c')
#split
split1 = [combined[(combined['time']>0)&(combined['time']<=100)], 
          pd.concat([combined[(combined['time']>500)&(combined['time']<=510)],
                     combined[(combined['time']>530)&(combined['time']<=540)],
                     combined[(combined['time']>580)&(combined['time']<=590)],
                     combined[(combined['time']>600)&(combined['time']<=700)]])]

split2 = [pd.concat([combined[(combined['time']>100)&(combined['time']<=150)],
                     combined[(combined['time']>300)&(combined['time']<=350)]]),
          combined[(combined['time']>540)&(combined['time']<=550)]]
    
split3 = [combined[(combined['time']>150)&(combined['time']<=200)],
          pd.concat([combined[(combined['time']>510)&(combined['time']<=520)],
                     combined[(combined['time']>590)&(combined['time']<=600)]])]

split4 = [pd.concat([combined[(combined['time']>200)&(combined['time']<=250)],
                     combined[(combined['time']>450)&(combined['time']<=500)]]),
          pd.concat([combined[(combined['time']>550)&(combined['time']<=560)],
                     combined[(combined['time']>570)&(combined['time']<=580)]])]  
                     
split5 = [pd.concat([combined[(combined['time']>250)&(combined['time']<=300)],
                     combined[(combined['time']>400)&(combined['time']<=450)]]),
          pd.concat([combined[(combined['time']>520)&(combined['time']<=530)],
                     combined[(combined['time']>560)&(combined['time']<=570)]])]        
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')>0.999):
            print('\n yuhu, accuracy already reach 99.9%')
            self.model.stop_training = True 
def get_model(n):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(30, input_shape = [1,], activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(n,activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
for i in [split1,split2,split3,split4,split5]:
    split = i
    skf = StratifiedKFold(n_splits=4)
    callback = myCallback()
    record = []
    for train_index, test_index in skf.split(split[0]['modi'], split[0]['open_channels']):
        print('\n Cross validation Segment \n')
        X_train, X_test = split[0]['modi'].values[train_index], split[0]['modi'].values[test_index]
        y_train, y_test = pd.get_dummies(split[0]['open_channels']).values[train_index], pd.get_dummies(split[0]['open_channels']).values[test_index]
        model = get_model(y_train.shape[1])
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, 1))
        nn_history = model.fit(scaler.transform(X_train.reshape(-1, 1)), y_train, epochs = 10, 
                               validation_data = (scaler.transform(X_test.reshape(-1, 1)),y_test),callbacks = [callback],batch_size = 4000)
        record.append(nn_history.history['val_accuracy'][-1])
    print('\n mean validation accuracy is {}'.format(np.array(record).mean()))
    
    #predict
    split[1].loc[split[1].index,'open_channels'] = np.argmax(model.predict(scaler.transform(np.array(split[1]['modi']).reshape(-1, 1))), axis=-1)
    
output = pd.concat([split1[1],split2[1],split3[1],split4[1],split5[1]])
output = output.sort_values('time')
real_output = output[['time','open_channels']].reset_index()
real_output = real_output.drop('index', axis = 1)
real_output['time'] =  sample_sub['time']
real_output.to_csv('ion.csv', index = False)
