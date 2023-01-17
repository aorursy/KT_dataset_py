import json
data=json.load(open("../input/train.json",'r'))

import numpy as np
from keras.utils import to_categorical


X_raw=[]
Y_raw=[]
bagOfWords_ingredients={}
bagOfWords_cuisine={}
bagOfWords_cuisine_inverted={}
maxIngLen=0


for aDish in data:
    maxIngLen=max(maxIngLen,len(aDish['ingredients']))
    xi=[]
    for ingredient in aDish['ingredients']:
        if ingredient not in bagOfWords_ingredients:
            bagOfWords_ingredients[ingredient]=len(bagOfWords_ingredients)+1
        xi.append(bagOfWords_ingredients[ingredient])
    X_raw.append(xi)
    if aDish['cuisine'] not in bagOfWords_cuisine:
        bagOfWords_cuisine[aDish['cuisine']]=len(bagOfWords_cuisine)
        bagOfWords_cuisine_inverted[len(bagOfWords_cuisine)-1]=aDish['cuisine']
    Y_raw.append(bagOfWords_cuisine[aDish['cuisine']])

X=np.zeros(shape=(len(X_raw),len(bagOfWords_ingredients)+1))

for i in range(len(X_raw)):
    for j in range(len(X_raw[i])):
        X[i,X_raw[i][j]]=1


Y=to_categorical(Y_raw)

print(X.shape,Y.shape,len(bagOfWords_ingredients))
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop

model=Sequential()
model.add(Dense(2000,activation='relu', input_shape=X.shape[1:]))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(250,activation='relu'))
model.add(Dense(125,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='softmax'))
model.compile(loss=categorical_crossentropy,optimizer=RMSprop(lr=0.0005),metrics=['accuracy'])
model.summary()
model.fit(X,Y,epochs=10,batch_size=500)
model.save("model.h5")
import json
test_data=json.load(open("../input/test.json",'r'))
print(len(test_data))
import numpy as np


with open('predictions.csv','w') as csvFile:
    csvFile.write("id,cuisine\n")
    for aDish in test_data:
        #maxIngLen=max(maxIngLen,len(aDish['ingredients']))
        xi=[]
        for ingredient in aDish['ingredients']:
            if ingredient not in bagOfWords_ingredients:
                xi.append(0)
            else:
                xi.append(bagOfWords_ingredients[ingredient])
        
        xt=np.zeros(shape=(1,len(bagOfWords_ingredients)+1))
        
        for index in xi:
            xt[0,index]=1
        
        prediction=model.predict(xt)
        dish_id=aDish['id']
        label=bagOfWords_cuisine_inverted[np.argmax(prediction)]
        csvFile.write(f"{dish_id},{label}\n")
    
!head predictions.csv