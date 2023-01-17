# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv('../input/zomato.csv')

print(data.info())

print(data.head())
"""

first thing  is to  explore  the  favorite  food  of  top 100

"""

favorite_food={}

for dish_liked  in  data['dish_liked']:

    try:

        for n in dish_liked.split(','):

            n=n.strip()

            favorite_food[n] =favorite_food.get(n,0)+1

    except:

        pass

favorite_food=list(favorite_food.items())

favorite_food.sort(key=lambda x:x[1],reverse=True )

#print(favorite_food)



import matplotlib.pyplot as plt 

import  seaborn as sns



dish_like_food=[]

dish_like_count=[]

for  l,m in zip(range(100),favorite_food):

    if  l<100:

        dish_like_food.append(m[0])

        dish_like_count.append(m[1])

#print(len(dish_like_food))

plt.figure(figsize=(20,50))

sns.barplot(y=dish_like_food,x=dish_like_count)

for i,l in  zip(range(len(dish_like_food)),dish_like_count):

    plt.text(l,i,l)

plt.xlabel('dish_like_count')

plt.ylabel('dish_like_food')

plt.title('the top 100  of favorite food')

plt.show()
data.rename(columns={'approx_cost(for two people)':'approx_cost'},inplace=True)

data['approx_cost']=data['approx_cost'].fillna('0').str.replace(',','')

data_pri=data.approx_cost.unique().astype('str')



the_pri=[]

the_pri_count=[]

image_pri={}

for i  in data_pri:

    image_pri[i]=len(data[data['approx_cost']==i])



image_pri=list(image_pri.items())

image_pri.sort(key=lambda x:x[1],reverse=True)

#print(image_pri)

for  l,m in zip(range(100),image_pri):

    if  l<45:

        the_pri.append("Pri"+ m[0])

        the_pri_count.append(m[1])



plt.figure(figsize=(10,10))

sns.barplot(y=the_pri,x=the_pri_count)

for i,l in zip(range(len(the_pri)),the_pri_count):

    plt.text(l,i,l)

plt.xlabel('price_count')

plt.ylabel('the  price')

plt.title('price_count')

plt.show()

"""

as we see:most of price  between  200 and 600 

"""
location=data.location.unique()

data=data[data['approx_cost'].astype('int')>0]

lacation_name=[]

location_ave_pri=[]

for i  in location:

    try:

        s=sum(data[data['location']==i].approx_cost.astype('int'))/(len(data[data['location']==i]))

    except:

        s=sum(data[data['location']==i].approx_cost.astype('int'))/((len(data[data['location']==i]))+0.0001)

    location_ave_pri.append(round(s,0))

        

plt.figure(figsize=(10,45))

sns.barplot(y=location,x=location_ave_pri)

for i,l in zip(range(len(location)),location_ave_pri):

    plt.text(l,i,l)

plt.xlabel('the average price of location')

plt.ylabel('location')

plt.title('The Average Price of Location')

plt.show()

print(data[data['location']=='Sankey Road'][['cuisines','rest_type','rate']])

"""

Sankey Road  must be  most busiest  location 

"""
label=[]

text=[]

for  m,n in zip(range(3000),data['reviews_list']):

    for l  in eval(n):

        word=l[0][-3:]

        if eval(word)<3:

            label.append(0)

        else :

            label.append(1)

        text.append(l[1].replace('RATED\n ',''))

#print(len(label))   56377



from  keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



maxlen= 100

max_word=10000



tokenizer= Tokenizer(num_words=max_word)

tokenizer.fit_on_texts(text)

data1=tokenizer.texts_to_sequences(text)

data1=np.asarray(data1)



word_index= tokenizer.word_index

label=np.asarray(label)







import numpy as np 

def vectorize_sequences(sequences,dimension=10000):

    results = np.zeros((len(sequences),dimension))

    for i ,sequence in enumerate(sequences):

        results[i,sequence] = 1.

    return results



x_train = vectorize_sequences(data1[:28000])

y_train=label[:28000]

y_test=label[28000:]

x_test =vectorize_sequences(data1[28000:])



y_train = np.asarray(y_train).astype('float16')

y_test = np.asarray(y_test).astype('float16')

from keras import models

from keras import  regularizers

from keras import layers

model = models.Sequential()

model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.002),activation='relu',input_shape=(10000,)))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(32,kernel_regularizer=regularizers.l2(0.002),activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.002),activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(1,activation='sigmoid'))







from keras.callbacks  import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from keras import losses

from keras import metrics

from keras import optimizers





callbacks_list=[

    EarlyStopping(

        monitor='acc',

        patience=2,

        verbose=1,

        restore_best_weights=True,

    ),

    ModelCheckpoint(

        filepath='..\input\mymodel.h5',

        monitor='val_loss',

        save_best_only=True,),

    ReduceLROnPlateau(

        montor='val_loss',

        factor=0.5,

        patience=2,

    )

]





model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss=losses.binary_crossentropy,

              metrics=[metrics.binary_accuracy])
x_val = x_train[:20000]

partial_x_train = x_train[20000:28000]

y_val = y_train[:20000]

partial_y_train = y_train[20000:28000]

history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=20,

                    batch_size=512,

                    validation_data=(x_val, y_val))
import matplotlib.pyplot as plt 





plt.figure(figsize=(25,10))



plt.subplot(121)

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(val_loss) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss',fontsize=17)

plt.xlabel('Epochs',fontsize=15)

plt.ylabel('Loss',fontsize=15)

plt.legend()



plt.subplot(122)

acc = history.history['binary_accuracy']

val_acc = history.history['val_binary_accuracy']

epochs = range(1, len(val_loss) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, acc, 'bo', label='Training accuracy')

# b is for "solid blue line"

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy',fontsize=17)

plt.xlabel('Epochs',fontsize=15)

plt.ylabel('accuracy',fontsize=15)

plt.legend()



plt.show()



result  =model.evaluate(x_test,y_test)

print('the  accuracy of test is %f' % result[1])