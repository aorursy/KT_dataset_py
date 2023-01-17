import numpy as np 

import json, csv
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers import Input, Embedding, Dense,Lambda,Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import os
import matplotlib.pyplot as plt
with open("../input/train.json") as f:
    data=json.load(f)
#First, let's make encoders for the ingredients and cuisines. I preffer to do this manually using dictionaries
cuisines=[]
ingredients=[]
n_ingredients=[]
for each in data:
    ingredients+=each['ingredients']
    cuisines+=[each['cuisine']]
ingredients=list(set(ingredients))
cuisines=list(set(cuisines))

cuisine_encoder={c:i for i,c in enumerate(cuisines)}
ingredients_encoder={c:(i+1) for i,c in enumerate(ingredients)}
#For the ingredients we reserve the 0 for unknown
#Then, I create the input and output lists for both the pre-training of the embedding and the final model
X,y,y_embedding, X_embedding=[],[],[],[]
for each in data:
    y.append(cuisine_encoder[each['cuisine']])
    each_ingredients=[ingredients_encoder[ingredient] for ingredient in each['ingredients']]
    X.append(each_ingredients)
    y_embedding+=[cuisine_encoder[each['cuisine']]]*len(each_ingredients)
    X_embedding+=each_ingredients

X=np.array(X)
y=to_categorical(y)
X_embedding=np.array(X_embedding)
y_embedding=to_categorical(y_embedding)
# Here, I define the split strategy and the generator for Keras Library

def split(n_samples,p_split=0.9):
    idx=np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(idx)
    train_samples=int(n_samples*p_split)
    idx_train,idx_test = idx[:train_samples],idx[train_samples:]
    return idx_train,idx_test

class generator(object):
    def __init__(self,X,y,idx,batch_shape=1):
        self.X = X
        self.y = y
        self.indexes = idx.copy()
        self.batch_shape = batch_shape
    def next(self):
        while True:
            self.indexes=np.roll(self.indexes,self.batch_shape)
            if self.batch_shape != 1:
                batch_x= np.expand_dims(self.X[self.indexes[:self.batch_shape]],0)
                batch_y= np.expand_dims(self.y[self.indexes[:self.batch_shape]],0)
            else:
                batch_x= np.expand_dims(self.X[self.indexes[0]],0)
                batch_y= np.expand_dims(self.y[self.indexes[0]],0)
            yield batch_x,batch_y
    def __len__(self):
        return len(self.indexes)//self.batch_shape
# Now we set the lenght of the embedings and the model for pretraining

embedding_dims = 128

input_ingredients = Input(batch_shape=(None, None))

embed=Embedding(output_dim=embedding_dims,input_dim=10000)(input_ingredients)
out = Dense(len(cuisines), activation='softmax',name='output')(embed)

model = Model(inputs=[input_ingredients], outputs=[out])

model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])
# After setting up the generators, we train the model
idx_train,idx_test=split(len(X_embedding))
g_train=generator(X_embedding,y_embedding,idx_train,batch_shape=32)
g_test=generator(X_embedding,y_embedding,idx_test,batch_shape=32)

history=model.fit_generator(generator=g_train.next(), steps_per_epoch = len(g_train),
                            validation_data = g_test.next(), validation_steps = len(g_test),
                            epochs=10)

plt.figure()
[plt.plot(v,label=str(k)) for k,v in history.history.items()]
plt.legend()
plt.show()

weights = model.layers[1].get_weights()[0]
input_ingredients = Input(batch_shape=(None,None))

embed=Embedding(output_dim=embedding_dims,input_dim=10000,weights=[weights],trainable=False)(input_ingredients)

x = Lambda(lambda x: K.sum(x, axis=1))(embed)
x=Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
out = Dense(len(cuisines), activation='softmax',name='output')(x)

model = Model(inputs=[input_ingredients], outputs=[out])

model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])
idx_train,idx_test=split(len(X))
g_train=generator(X,y,idx_train)
g_test=generator(X,y,idx_test)
log=model.fit_generator(generator=g_train.next(), steps_per_epoch = len(idx_train),
                            validation_data = g_test.next(), validation_steps = len(idx_test),
                            use_multiprocessing=False,
                            epochs=10)
plt.figure()
[plt.plot(v,label=str(k)) for k,v in log.history.items()]
plt.legend()
plt.show()
#Stochastic GD unique shapes: acc: 73/70
#Stochastic GD with pre-training of the embedding: acc: 77/78
#128: categorical_accuracy: 0.7888 - val_loss: 0.7004 - val_categorical_accuracy: 0.8122
# And finally, predict the test set and write the submission
with open("../input/test.json") as f:
    data=json.load(f)
    
cuisine_decoder={i:c for c,i in cuisine_encoder.items()}

cuisines,ids = [], []
for each in data:
    ingredients=[]
    for ingredient in each['ingredients']:
        if ingredient in ingredients_encoder.keys():
            ingredients.append(ingredients_encoder[ingredient])
        else:
            ingredients.append(0)
    ingredients = np.expand_dims(ingredients,0)
    cuisine = model.predict(ingredients)

    cuisines.append(cuisine_decoder[np.argmax(cuisine)])
    ids.append(each['id'])

with open('submission.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(('id','cuisine'))
    for a,b in zip(ids, cuisines):
        csvwriter.writerow([a,b])
