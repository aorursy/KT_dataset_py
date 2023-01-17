import csv
import pandas as pd
import numpy as np
data = pd.read_csv('../input/train.csv')# as f:
    #data = list(csv.reader(f))
train_data = np.array(data[1:])
labels = train_data[:, 0].astype('float')
train_data = train_data[:, 1:].astype('float') #/ 255.0
class cls(object):
    def __repr__(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        return ''

cls = cls()
y=0
for i in train_data[0:1
,:]:
    k=0
    for t in i:
        if t > 0: 
            k=k + 1
    #i = np.append(train_data[0:5,0:],k)
    #np.concatenate(t,i)
    np.insert(train_data[y:y+1,:],slice(0,1),123456)
    y = y + 1
    print (train_data[y])
#print (train_data.shape)
    #print(train_data[y:y+1,-1]) 
    #print(len(train_data[y:y+1,0:]))
