import os
print(os.listdir('../input/ctcfloop-data/Data'))
from Bio import SeqIO
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import backend as K
import tensorflow as tf
np.random.seed(12345)
def get_num(fasta_name, fasta_name2):
    num = 0
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        if(not(re.search('N',str(seq_record.seq.upper())))):
            num+=1
    for seq_record2 in SeqIO.parse(fasta_name2,"fasta"):
        if(not(re.search('N',str(seq_record2.seq.upper())))):
            num+=1
    return num
'''Convert the input sequences into binary matrixs'''
def get_seq_matrix(fasta_name,seqmatrix,rank): 
    labellist = []
    for seq_record in SeqIO.parse(fasta_name,"fasta"):
        label = seq_record.id
        sequence = seq_record.seq.upper()
        if(re.search('N',str(sequence))):
            continue
        Acode = np.array(get_code(sequence,'A'),dtype=int)
        Tcode = np.array(get_code(sequence,'T'),dtype=int)
        Gcode = np.array(get_code(sequence,'G'),dtype=int)
        Ccode = np.array(get_code(sequence,'C'),dtype=int)
        seqcode = np.vstack((Acode,Tcode,Gcode,Ccode))
        labellist.append(label)
        seqmatrix[rank] = seqcode
        rank +=1
    return seqmatrix,labellist,rank
def get_code(seq,nt):
    nts = ['A','T','G','C']
    nts.remove(nt)
    codes = str(seq).replace(nt,str(1))
    for i in range(0,len(nts)):
        codes = codes.replace(nts[i],str(0))
    coding = list(codes)
    for i in range(0,len(coding)):
        coding[i] = float(coding[i])
    return coding
'''Get the train, validation and test set from the input'''
def get_data(infile,infile2):
    rank = 0
    num = get_num(infile,infile2)
    seqmatrix = np.zeros((num,4,1038))
    (seqmatrix, poslab, rank) = get_seq_matrix(infile,seqmatrix,rank)
    (seqmatrix, neglab, rank) = get_seq_matrix(infile2,seqmatrix,rank)
    X = seqmatrix
    Y = np.array(poslab + neglab,dtype = int)
    validation_size = 0.10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size)
    return np.transpose(X_train,axes=(0,2,1)), np.transpose(X_val,axes=(0,2,1)), np.transpose(X_test,axes=(0,2,1)), Y_train, Y_val, Y_test
'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP/P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
def one_hot_to_seq(matrix):
    nts = ['A','T','G','C']
    seqs = []
    index = [np.where(r==1)[0][0] for r in matrix]
    for i in index:
        seqs.append(nts[i])
    seq = ''.join(seqs)
    return seq
infile   = '../input/ctcfloop-data/Data/GM12878_pos_seq.fasta'
secondin = '../input/ctcfloop-data/Data/GM12878_neg_seq.fasta'
X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data(infile,secondin)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
labels = ['negative', 'positive']
dense_unit    = 128  #112
droprate_cnn  = 0.5  #0.4279770967884926
droprate_lstm = 0.05 #0.05028428952624636
filter_unit = 208
l2_reg = 5.2164660610264974e-05
#learning_rate = 0.00010199140620075788
lstm_unit = 104
pool_size = 4
window_size = 13
## Build Model
inputs = Input(shape = (1038, 4))

# 1st Conv1D
cnn_out = Conv1D(filter_unit, window_size,
                 kernel_regularizer=l2(l2_reg),
                 activation="relu")(inputs)
pooling_out = MaxPooling1D(pool_size=pool_size, strides=pool_size)(cnn_out)
dropout1 = Dropout(droprate_cnn)(pooling_out)

# 2nd Conv1D
cnn_out2 = Conv1D(filter_unit, window_size,
                  kernel_regularizer=l2(l2_reg),
                  activation="relu")(dropout1)
pooling_out2 = MaxPooling1D(pool_size=pool_size, strides=pool_size)(cnn_out2)
dropout2 = Dropout(droprate_cnn)(pooling_out2)

# LSTM
lstm_out = Bidirectional(LSTM(lstm_unit, return_sequences=True, 
                              kernel_regularizer=l2(l2_reg)
                             ),merge_mode = 'concat')(dropout2)

# Attention
a = Permute((2, 1))(lstm_out)
a = Dense(61, activation='softmax')(a)
a_probs = Permute((2, 1), name='attention_vec')(a)

attention_out = multiply([lstm_out, a_probs])
attention_out = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
dropout3 = Dropout(droprate_lstm)(attention_out)

# FC
dense1_out = Dense(128, 
                  kernel_regularizer=l2(l2_reg),
                  activation='relu')(dropout3)

dense2_out = Dense(16, 
                  kernel_regularizer=l2(l2_reg),
                  activation='relu')(dense1_out)

output = Dense(1, activation='sigmoid')(dense2_out)

model = Model(inputs=[inputs], outputs=output)

model.summary()
# Compile Model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[roc_auc])
batch_size = 200
num_epochs = 100
filepath = "deepctcfloop_model.hdf5"
#checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
#earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, validation_data=(X_val,Y_val))#, callbacks=[checkpointer,earlystopper])
## Save Model
model.save(filepath)
y_pred = model.predict(X_test)
rounded = [round(y[0]) for y in y_pred]
cm = confusion_matrix(Y_test, rounded)
print(cm)
print(classification_report(Y_test, rounded, target_names=labels))
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
specificity = TN / float( TN + FP)
sensitivity = TP / float(FN + TP)
print('Specificity:',specificity)
print('Sensitivity:',sensitivity)
