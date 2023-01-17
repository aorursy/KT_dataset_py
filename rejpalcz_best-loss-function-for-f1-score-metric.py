from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import numpy as np
from prettytable import PrettyTable
from prettytable import ALL
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
# ground truth
Y = np.zeros((5,2))
# first label is assigned to 20 % of observations
Y[0,0] = 1
# second label is assigned to 80 % of observations
Y[0:4,1] = 1

# ground truth with shape (BATCH_SIZE, NO_OF_LABELS)
print(Y)
first = {}

for TP in range(2): # TP can be 0..1
    for FP in reversed(range(5)): # FP can be 0..4
        idx = TP*5+(4-FP)
        name = 'TP' + str(TP) + 'FP' + str(FP)
        Yhat1 = np.zeros(5)
        Yhat1[0:TP] = 1
        Yhat1[5-FP:] = 1
        first.update({name: Yhat1})

second = {}

for TP in range(5): # TP can be 0..4
    for FP in reversed(range(2)): # FP can be 0..1
        idx = TP*5+(4-FP)
        name = 'TP' + str(TP) + 'FP' + str(FP)
        Yhat2 = np.zeros(5)
        Yhat2[0:TP] = 1
        Yhat2[5-FP:] = 1
        second.update({name: Yhat2})
t = PrettyTable(['1st/2nd']+list(first.keys()))
pltX = []
pltY = []

for name2,data2 in second.items():

    row = [name2]
    for name1,data1 in first.items():
        data = np.stack((data1, data2), axis=1)
        loss = np.mean(K.eval(binary_crossentropy(K.variable(Y), K.variable(data))))
        f1 = f1_score(Y, data, average='macro')
        pltX.append(loss)
        pltY.append(f1)
        row.append('{:2f}\n{:2f}'.format(loss, f1))
    t.add_row(row)


print('Displaying result')
print('Columns = prediction for first label (20 % of 1s in ground truth)')
print('Rows = prediction for second label (80 % of 1s in ground truth)')
print('Cell = 1st number binary_crossentropy loss, 2nd number macro F1-score')
print('')
t.hrules = ALL
print(t)
plt.scatter(pltX, pltY)
plt.ylabel('Macro F1-score')
plt.xlabel('Binary crossentropy loss')
plt.show()
t = PrettyTable(['2nd\1st']+list(first.keys()))
pltX = []
pltY = []

for name2,data2 in second.items():

    row = [name2]
    for name1,data1 in first.items():
        data = np.stack((data1, data2), axis=1)
        # nan -> 0 this is dubious, is that correct? 
        loss = np.mean(np.nan_to_num(K.eval(categorical_crossentropy(K.variable(Y), K.variable(data)))))
        f1 = f1_score(Y, data, average='macro')
        pltX.append(loss)
        pltY.append(f1)
        row.append('{:2f}\n{:2f}'.format(loss, f1))
    t.add_row(row)
print('Displaying result')
print('Columns = prediction for first label (20 % of 1s in ground truth)')
print('Rows = prediction for second label (80 % of 1s in ground truth)')
print('Cell = 1st number categorical_crossentropy loss, 2nd number macro F1-score')
print('')
t.hrules = ALL
print(t)
plt.scatter(pltX, pltY)
plt.ylabel('Macro F1-score')
plt.xlabel('Categorical crossentropy loss')
plt.show()
import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])
model.summary()
t = PrettyTable(['1st/2nd']+list(first.keys()))
pltX = []
pltY = []

for name2,data2 in second.items():

    row = [name2]
    for name1,data1 in first.items():
        data = np.stack((data1, data2), axis=1)
        loss = K.eval(f1_loss(Y, data))
        f1 = f1_score(Y, data, average='macro')
        pltX.append(loss)
        pltY.append(f1)
        row.append('{:2f}\n{:2f}'.format(loss, f1))
    t.add_row(row)
print('Displaying result')
print('Columns = prediction for first label (20 % of 1s in ground truth)')
print('Rows = prediction for second label (80 % of 1s in ground truth)')
print('Cell = 1st number focal loss, 2nd number macro F1-score')
print('')
t.hrules = ALL
print(t)
plt.scatter(pltX, pltY)
plt.ylabel('Macro F1-score')
plt.xlabel('Differentiable F1 loss')
plt.show()
