import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
import wfdb
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Dropout,MaxPooling1D,Add,LSTM,Bidirectional
from keras.utils import to_categorical
from keras.metrics import AUC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from keras.layers import Conv1D,Conv2D,TimeDistributed, BatchNormalization
from pprint import pprint

#set path of dataset location
dataPath='/kaggle/input/hearta/mit-bih-arrhythmia-database-1.0.0/'

#list of patients
pts=[
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]
df = pd.DataFrame()

#take required headers out of the file
#i.e. values ()
for pt in pts:
    file = dataPath+pt
    annotation = wfdb.rdann(file,'atr')
    sym = annotation.symbol

    values,counts= np.unique(sym,return_counts=True)
    df_sub = pd.DataFrame({'sym':values,'val':counts,'pt':[pt]*len(counts)})
    df = pd.concat([df,df_sub],axis=0)


    df.groupby('sym').val.sum().sort_values(ascending = False)

#list of nonbeat and abnormal symbols
nonbeat = [
    '[','!',']','x','(',')','p','t','u','`','\'','^',
    '|','~','+','s','T','*','D','=','"','@','Q','?'
]
abnormal =["L","R","V","/","A","f","F","j","a","E","J","e","S" ]

#break into normal, abnormal or nonbeat
df["cat"]=-1
df.loc[df.sym =="N","cat"]==0
df.loc[df.sym.isin(abnormal),"cat"]= 1
df.groupby("cat").val.sum()

def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file,"atr")
    p_signal = record.p_signal
    assert record.fs ==360, "sample freq is not 360"
    atr_sym = annotation.symbol
    atr_sample = annotation.sample

    return p_signal,atr_sym,atr_sample

def make_dataset(pts,num_sec,fs):
    num_cols = 2*num_sec*fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []

    max_rows =[]
    for pt in pts:
        file = dataPath+pt
        p_signal,atr_sym,atr_sample=load_ecg(file)
        p_signal = p_signal[:,0]
        df_ann= pd.DataFrame({"atr_sym":atr_sym,
                            "atr_sample":atr_sample})

        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal +["N"])]

        num_rows=len(df_ann)
        X=np.zeros((num_rows,num_cols))
        Y = np.zeros((num_rows,1))
        
        max_row = 0    
        for atr_sample,atr_sym in zip(df_ann.atr_sample.values, df_ann.atr_sym.values):

            left = max([0,(atr_sample - num_sec*fs)])
            right = min([len(p_signal),(atr_sample+num_sec*fs)])
            x = p_signal[left:right]
            if len(x)==num_cols:
                X[max_row,:] =x 
                Y[max_row,:]=int(atr_sym in abnormal)

                sym_all.append(atr_sym)
                max_row +=1

        X=X[:max_row,:]
        Y=Y[:max_row,:]
        max_rows.append(max_row)
        X_all= np.append(X_all,X,axis =0)
        Y_all = np.append(Y_all,Y,axis = 0)

    X_all = X_all[1:,:]
    Y_all=Y_all[1:,:]

    assert np.sum(max_rows)== X_all.shape[0], "number of rows messed up 1"
    assert Y_all.shape[0]== X_all.shape[0], "number of rows messed up 2"
    assert Y_all.shape[0] == len(sym_all), "number of rows messed up 3"

    return X_all,Y_all,sym_all
    
def calc_prevalance(y_actual):
    return(sum(y_actual)/len(y_actual))

def calc_specificity(y_actual,y_pred, thresh):
    return(((y_pred<thresh) & (y_actual==0))/sum(y_actual==0))

def print_report(y_actual,y_pred, thresh):
    auc = roc_auc_score(y_actual,y_pred)
    accuracy = accuracy_score(y_actual, (y_pred>thresh))
    recall = recall_score(y_actual,(y_actual >thresh))
    precision=precision_score(y_actual,(y_pred > thresh))
    specificity= calc_specificity(y_actual, y_pred, thresh)
    print("AUC: {}".format(auc))
    print("accuracy: {}".format(accuracy))
    print("recall: {}".format(recall))
    print("precision: {}".format(precision))
    #print("specificity: {}".format(specificity))
    print("prevalance: {}".format(calc_prevalance(y_actual)))
    print("  ")
    return auc,accuracy,recall,precision, specificity

        
X_all,Y_all,sym_all = make_dataset(pts, 6,6)

X_train,X_valid,y_train,y_valid = train_test_split(X_all,Y_all,test_size=0.33, random_state = 42)



###
#Start CNN#
###
X_train_cnn = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],1))

print(X_train_cnn.shape)
print(X_valid_cnn.shape)
model1 = Sequential()
model1.add((Conv1D(filters = 128, kernel_size = 5, activation = 'relu',input_shape=(X_train_cnn.shape[1],1))))
model1.add(Bidirectional(LSTM(64,input_shape=(X_train_cnn.shape[1],X_train_cnn.shape[2]))))
model1.add(Dropout(rate=0.25))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',AUC()])
history=model1.fit(X_train_cnn,y_train,batch_size=32, epochs=50, verbose= 1)


y_train_preds_cnn = model1.predict_proba(X_train_cnn,verbose=1)
y_valid_preds_cnn = model1.predict_proba(X_valid_cnn,verbose=1)
thresh = (sum(y_train)/len(y_train))[0]


print("Train")
print_report(y_train,y_train_preds_cnn,thresh)
print("Valid")
auc,accuracy,recall,precision, specificity= print_report(y_valid,y_valid_preds_cnn,thresh)

print(history.history.keys())
print(history.history['accuracy'])
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history.history['auc_2'])
plt.title('AUC ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

