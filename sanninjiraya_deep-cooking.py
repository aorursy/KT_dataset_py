import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
dataset = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
cuisines = dataset['cuisine'].unique()
ing=[]
for ij in dataset['ingredients']:
    for jj in ij:
        if jj not in ing:
            ing.append(jj)
for ingredient in ing:
    dataset[ingredient]=np.zeros(len(dataset["ingredients"]))

def One_Hot(dt1, dt2):    
    kj=0
    for nj in dt1:
        for ingredient in nj:
            if ingredient in ing:
                dt2.loc[kj,ingredient]=1
            else:
                pass
        kj +=1
One_Hot(dataset['ingredients'], dataset)
ind = ing
cus = 'cuisine'
x = dataset[ind]
y = dataset[cus]
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
y_train = encode.fit_transform(y)
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU,Dropout
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
clas=Sequential()
clas.add(Dense(output_dim=1024,init='uniform',activation='relu',input_dim=6714))
clas.add(Dense(output_dim=512,init='uniform',activation='relu'))
clas.add(Dropout(p=0.2))
clas.add(Dense(output_dim=256,init='uniform',activation='linear'))
clas.add(LeakyReLU(alpha=0.3))
clas.add(Dropout(p=0.2))
clas.add(Dense(output_dim=64,init='uniform',activation='relu'))
clas.add(Dropout(p=0.2))
clas.add(LeakyReLU(alpha=0.3))
clas.add(Dense(output_dim=20,init='uniform',activation='softmax'))
clas.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
clas.fit(x,y_train,nb_epoch=6,batch_size=64)
for ingredient in ing:
    test[ingredient]=np.zeros(len(test["ingredients"]))
One_Hot(test['ingredients'], test)
y_pred = clas.predict(test[ind])
y_pred = encode.inverse_transform(y_pred.all())
outputs = pd.DataFrame(test['id'])
outputs['cuisine'] = pd.Series(y_pred, name='cuisine')
outputs.to_csv('outputs.csv', index=False)

