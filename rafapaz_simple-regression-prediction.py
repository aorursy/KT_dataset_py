import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
df = pd.read_csv('../input/student-mat.csv',sep=',')
df.head()
def to_sequence(df, col):
    res = df.copy()
    dic = {k:v for k,v in zip(range(len(df[col].unique())),df[col].unique())}
    for k,v in dic.items():
        res[col][res[col] == v] = k
        
    return res
df.head(n=20)
tab = to_sequence(df, 'sex')
tab = to_sequence(tab, 'school')
tab = to_sequence(tab, 'address')
tab = to_sequence(tab, 'famsize')
tab = to_sequence(tab, 'Pstatus')
tab = to_sequence(tab, 'Mjob')
tab = to_sequence(tab, 'Fjob')
tab = to_sequence(tab, 'reason')
tab = to_sequence(tab, 'guardian')
tab = to_sequence(tab, 'schoolsup')
tab = to_sequence(tab, 'famsup')
tab = to_sequence(tab, 'paid')
tab = to_sequence(tab, 'activities')
tab = to_sequence(tab, 'nursery')
tab = to_sequence(tab, 'higher')
tab = to_sequence(tab, 'internet')
tab = to_sequence(tab, 'romantic')
tab.head()
y = np.array(tab['G3'])
aux = tab.copy()
del aux['G3']
x = np.array(aux)
x_train = x[:round(len(x)*0.85)]
y_train = y[:round(len(y)*0.85)]
x_test = x[round(len(x)*0.85):]
y_test = y[round(len(x)*0.85):]
x_train.shape
y_train.shape
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=32))
model.add(Dense(units=16, activation='relu', input_dim=32))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=4, verbose=1)
pred = model.predict(x_test)
acul = 0.0
total = 0.0
for p, r in zip(pred, y_test):
    if r != 0:
        acul += min(p,r)/max(p,r)
        total += 1

print("Accuracy = {0:.2f} %".format(float(acul/total*100)))
