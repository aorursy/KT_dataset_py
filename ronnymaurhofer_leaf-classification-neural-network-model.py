import pandas as pd

import os

from shutil import copyfile

import matplotlib.pyplot as plt

from matplotlib.image import imread

import time



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit
def plot_img(img):

    plt.imshow(img, cmap='gray')

    plt.axis("off")

    plt.show()

    

def plot_img_by_id(id, species = ''):

    src = './LeafClassification/' + str(id) + '.jpg'

    img = imread(src)

    plt.imshow(img, cmap='gray')

    plt.suptitle('Predicted species: ' + species)

    plt.axis("off")

    plt.show()

    

def plot_img_by_species(species):

    ldir = './training_data/' + str(species) + '/'

    plt.figure(figsize=(28,28))

    #plt.suptitle('Predicted species: ' + species)

    x, y = len(os.listdir(ldir)), 1

     

    i = 1

    print(species)

    for d in os.listdir(ldir):

        src = ldir + d

        img = imread(src)

        

        plt.subplot(y, x, i)

        plt.imshow(img, cmap='gray')

        plt.axis("off")

        i += 1

            

    plt.show()
# method to reload data

def reload_data():

    # Load test & train datasets

    train_data = pd.read_csv("train.csv")

    test_data = pd.read_csv("test.csv")

    df = [train_data, test_data]

    df = pd.concat(df, axis=0, sort=False)

    

    return train_data, test_data, df



# load predictions

def load_pred():

    # id should be index

    pred = pd.read_csv("predictions.csv", index_col='id')

    return pred



train_data, test_data, df = reload_data()

# for backup reason

train_data_copy, test_data_copy, df = reload_data()
print(train_data.shape)

train_data.describe()

train_data.head()



print(df.shape)

df.describe()

df.head()



print(test_data.shape)

test_data.describe()

test_data.head()
#Source file 

sourcefolder = os.getcwd() + '/'

subdirs = ['/training_data/', '/test_data/']

labeldirs = train_data['species'].unique()



for subdir in subdirs:

   

    newdir_parent = '.' + subdir#+ labldir

    if not os.path.exists(newdir_parent):

        print(newdir_parent)

        os.mkdir(newdir_parent)

    

    ##### merge df and images; seperate according to df['species']; put image in species folder;

    

    for labldir in labeldirs:    

    # create label subdirectories

        newdir_child = newdir_parent + labldir

        #print(newdir_child)

        if not os.path.exists(newdir_child) and subdir == '/training_data/':

            #print(newdir_child)

            os.mkdir(newdir_child)
# move train data info folder

for i, val in train_data.iterrows():

    cdir = val['species']

    fname = int(val['id'])

    src = './LeafClassification/' + str(fname) + '.jpg'

    dst = sourcefolder + '/training_data/' + str(cdir) + '/' + str(fname) + '.jpg'

    #print(i, int(val['id']), val['species'])

    #print('src: ' + str(src))

    #print('dst:' + str(dst))

    copyfile(src, dst)



# move test data into folder    

for i, val in test_data.iterrows():

    fname = int(val['id'])

    src = './LeafClassification/' + str(fname) + '.jpg'

    dst = sourcefolder + '/test_data/' + str(fname) + '.jpg'

    #print(i, int(val['id']))

    #print('src: ' + str(src))

    #print('dst:' + str(dst))

    copyfile(src, dst)
# remove column 'id' from train data und save in variable train_id

train_id = train_data.pop('id')

test_id = test_data.pop('id')



# remove column 'species' from train data und save in variable train_y, then transform into categorical

train_y = train_data.pop('species')

train_y = LabelEncoder().fit(train_y).transform(train_y)

train_y = to_categorical(train_y)



#scale training data

train_x = StandardScaler().fit(train_data).transform(train_data)

test_x = StandardScaler().fit(test_data).transform(test_data)
## retain class balances

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=12345)

train_index, val_index = next(iter(sss.split(train_x, train_y)))

x_train, x_val = train_x[train_index], train_x[val_index]

y_train, y_val = train_y[train_index], train_y[val_index]

print("x_train dim: ",x_train.shape)

print("x_val dim:   ",x_val.shape)
input_dim = train_x.shape[1]

EPOCHS = 100

batch_size = 128
model = Sequential()

model.add(Dense(1024,input_dim=input_dim))

model.add(Dropout(0.2))

model.add(Activation('sigmoid'))

model.add(Dense(512))

model.add(Dropout(0.3))

model.add(Activation('sigmoid'))

model.add(Dense(99))

model.add(Activation('softmax'))
# compile model

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
# fit model

start = time.time()

history = model.fit(train_x,train_y,validation_data=(x_val, y_val),batch_size=batch_size,epoch=EPOCHS,verbose=0)

end = time.time()

print(round((end-start),2), "seconds")
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

#plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='lower right')

plt.show()



print('-'*50)

print('Training accuracy: ' + str(max(history.history['acc'])))

print('Validation accuracy: ' + str(max(history.history['val_acc'])))

print('-'*50)



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

#plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper right')

plt.show()



print('-'*50)

print('Training loss: ' + str(min(history.history['loss'])))

print('Validation loss: ' + str(min(history.history['val_loss'])))

print('-'*50)
predict_y = model.predict_proba(test_x)
species = train_data_copy.species.unique()

predict_out = pd.DataFrame(predict_y,index=test_id,columns=sorted(species))

predict_out['predicted species'] = predict_out.idxmax(axis=1)
predict_out.head()
# check predicted with training data

check_limit = 2

for (i,val) in predict_out.iterrows():

    plot_img_by_id(i, species = val['predicted species']) # i is should be same as val['id']

    print('-'*50)

    plot_img_by_species(val['predicted species'])

    check_limit -= 1

    if check_limit <= 0:

        break

        

#plot_img_by_id(4, species = 'Quercus_Agrifolia')

#plot_img_by_species('Quercus_Agrifolia')
#model.save_weights('./models/leaf_classification_weights_best.h5')

model.save('./models/leaf_classification_model_best.h5')
# write file to csv

fp = open('predictions_neuralnetwork_1.csv','w')

fp.write(predict_out.to_csv())