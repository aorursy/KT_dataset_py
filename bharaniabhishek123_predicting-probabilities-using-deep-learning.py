# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

def read_data(train):

    data_comments=[]

    data_y = []

    for index,row in train.iterrows():

        comment = row.comment_text

        data_comments.append(comment)

        y_values = np.array([row.toxic,row.severe_toxic,row.obscene,row.threat,row.insult,row.identity_hate])

        data_y.append(y_values)

    return data_comments,data_y



train = pd.read_csv('../input/train.csv',encoding='utf-8',error_bad_lines=False)

train_comments,train_y = read_data(train)



from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(train_comments)

# one_hot_encoding_results = tokenizer.texts_to_matrix(train_comments,mode='binary')

# print(one_hot_encoding_results.shape = [95851,10000])



x_train = tokenizer.texts_to_matrix(train_comments,mode='binary')



#vectorize the labels as well

y_train = np.asarray(train_y).astype('float32')





from keras import models

from keras import layers

num_classes =6



model = models.Sequential()



model.add(layers.Dense(32,activation='relu',input_shape=(10000,)))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(num_classes,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])





history = model.fit(x_train,y_train,epochs=8,batch_size=512,

                   validation_split=0.2)







history_dict = history.history



print(history_dict.keys())



acc =history_dict['acc']

val_acc =history_dict['val_acc']



#Plot the training and val loss and accuracy

import matplotlib.pyplot as plt



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(acc) + 1)





plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()





plt.clf()

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()







def read_data_test(test_data):

    data_comments = []

    for index, row in test_data.iterrows():

        comment = row.comment_text

        if comment is not np.nan:

            data_comments.append(comment)

    return data_comments



test = pd.read_csv('../input/test.csv',encoding='utf-8',error_bad_lines=False)



test_comments = read_data_test(test)



tokenizer.fit_on_texts(test_comments)

x_test = tokenizer.texts_to_matrix(test_comments,mode='binary')



# values of probabilities for each id

probabilities =model.predict(x_test)



#creating dataframe

prob_df= pd.DataFrame(probabilities)





#creating dataframe of id's

id_df = test['id']





#final results will be

results = pd.concat([id_df,prob_df],axis=1)



print(results)



# results.to_csv('final_results.csv',index=False)






