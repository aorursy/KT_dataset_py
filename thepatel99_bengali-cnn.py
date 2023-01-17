import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

#for data handaling
from tqdm.auto import tqdm
import numpy as np
import pandas as pd 

#for image processing
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#for calc accuaracy and spliting the data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
PATH='/kaggle/input/bengaliai-cv19/'
HEIGHT = 137
PATH='/kaggle/input/bengaliai-cv19/'
HEIGHT = 137
WIDTH = 236
SIZE = 64
batch_size = 256
epochs = 64
def crop_resize_image(image_df):
    cropped_resized_img={}
    for i in tqdm(range(len(image_df))):
        image=image_df.iloc[i].values.reshape(HEIGHT,WIDTH)
        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        x_min=[]
        x_max=[]
        y_min=[]
        y_max=[]
        for cordinate in contours:
            x,y,w,h=cv2.boundingRect(cordinate)
            x_min.append(x)
            x_max.append(x+w)
            y_min.append(y)
            y_max.append(y+h)
        x1=min(x_min)
        x2=max(x_max)
        y1=min(y_min)
        y2=max(y_max)
        cropped_img=image[y1:y2,x1:x2]
        resized_img=cv2.resize(cropped_img,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
        cropped_resized_img[i]=resized_img.reshape(-1)
    return pd.DataFrame(cropped_resized_img).T
inputs= Input(shape=(SIZE,SIZE,1))
model=Conv2D(filters=32,kernel_size=(3,3),padding='SAME',activation='relu',input_shape=(SIZE,SIZE,1))(inputs)
#model=MaxPool2D(pool_size=(2,2))(model)

model=Conv2D(filters=64,kernel_size=(3,3),padding='SAME',activation='relu')(model)
model=MaxPool2D(pool_size=(2,2))(model)

model=Conv2D(filters=128,kernel_size=(3,3),padding='SAME',activation='relu')(model)
model=MaxPool2D(pool_size=(2,2))(model)

model=Conv2D(filters=128,kernel_size=(3,3),padding='SAME',activation='relu')(model)
model=MaxPool2D(pool_size=(2,2))(model)

model=Conv2D(filters=128,kernel_size=(3,3),padding='SAME',activation='relu')(model)
model=MaxPool2D(pool_size=(2,2))(model)

model=Dropout(0.3)(model)
model=BatchNormalization(momentum=0.15)(model)
model=Flatten()(model)
model=Dense(1000,activation='relu')(model)
model=Dropout(0.3)(model)
model=Dense(500,activation='relu')(model)

root=Dense(168,activation='softmax',name='root')(model)
vowel=Dense(11,activation='softmax',name='vowel')(model)
consonant=Dense(7,activation='softmax',name='consonant')(model)

model=Model(inputs=inputs,outputs=[root,vowel,consonant])
model.summary()
from tensorflow.keras.utils import plot_model
cnn1model = keras.Model(inputs=inputs, outputs=[root,vowel ,consonant])

plot_model(cnn1model, to_file='mode4.png')
lr_root=ReduceLROnPlateau(monitor='root_acc',factor=0.9,patience=3,min_lr=0.00001,verbose=1)
lr_vowel=ReduceLROnPlateau(monitor='vowel_acc',factor=0.9,patience=3,min_lr=0.00001,verbose=1)
lr_consonant=ReduceLROnPlateau(monitor='consonant_acc',factor=0.9,patience=3,min_lr=0.00001,verbose=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=batch_size,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)
            
        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict
train_image=pd.read_parquet(PATH+'train_image_data_3.parquet').drop(['image_id'],axis=1)
train_image_info=pd.read_csv(PATH+'train.csv')

X_train=crop_resize_image(train_image).values.reshape(-1,SIZE,SIZE,1)
Y_train=train_image_info[3*50210:(3+1)*50210]

_, x_test, _, y_test = train_test_split(X_train, Y_train, test_size=0.85, random_state=420)
y_test_root = pd.get_dummies(y_test['grapheme_root']).values
y_test_vowel = pd.get_dummies(y_test['vowel_diacritic']).values
y_test_consonant = pd.get_dummies(y_test['consonant_diacritic']).values
print(y_test_root.shape)
del train_image
del X_train
del Y_train
def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest
records={}

for i in range(1,4):
    train_image=pd.read_parquet(PATH+'train_image_data_'+str(i)+'.parquet').drop(['image_id'],axis=1)
    
    y_train=train_image_info[i*50210:(i+1)*50210]
    x_train=crop_resize_image(train_image).values.reshape(-1,SIZE,SIZE,1)
    #plt.imshow(train_image.iloc[100].values.reshape(HEIGHT,WIDTH))
    del train_image

    #plt.imshow(X_train[100])
    #print(Y_train.iloc[100])
    #print('splitting about to start')


    #print('splitting completed')

    data_generator=MultiOutputDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        shear_range=0.3,
        height_shift_range=0.08,
        zoom_range=0.08
    )
    data_generator.fit(x_train)
    #print('fitting completed')
    
    y_train_root = pd.get_dummies(y_train['grapheme_root']).values
    y_train_vowel = pd.get_dummies(y_train['vowel_diacritic']).values
    y_train_consonant = pd.get_dummies(y_train['consonant_diacritic']).values
    
    del y_train


    #print('learning about to start')
    result=model.fit_generator(
        data_generator.flow(
            x_train,
            {
                'root':y_train_root,'vowel':y_train_vowel,'consonant':y_train_consonant
            },
            batch_size=batch_size
        ),
        epochs=epochs,
        validation_data=(x_test,[y_test_root,y_test_vowel,y_test_consonant]),
        steps_per_epoch=y_train_root.shape[0]//batch_size,
        callbacks=[lr_root,lr_vowel,lr_consonant]
    )
    del x_train
    del y_train_root
    del y_train_vowel
    del y_train_consonant

    records=appendHist(records,result.history) 
print(records)
def plot_graph(x,s):
    plt.figure(figsize=(10,10))
    plt.plot(x['val_root_'+s])
    plt.plot(x['val_vowel_'+s])
    plt.plot(x['val_consonant_'+s])
    plt.plot(x['root_'+s])
    plt.plot(x['vowel_'+s])
    plt.plot(x['consonant_'+s])
    plt.title('Learning Dataset '+s)
    plt.ylabel(s)
    plt.xlabel('epoch')
    if s=='accuracy':
        plt.legend(['val_root_'+s,'val_vowel_'+s,'val_consonant_'+s,'root_'+s,'vowel_'+s,'consonant_'+s], loc='best')
    elif s=='loss':
        plt.plot(x['val_'+s])
        plt.plot(x[s])
        plt.legend(['val_root_'+s,'val_vowel_'+s,'val_consonant_'+s,'root_'+s,'vowel_'+s,'consonant_'+s,'val_'+s,s], loc='best')
    plt.show()
plot_graph(records,'accuracy')
plot_graph(records,'loss')
preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}
row_id=[]
target=[]
for i in range(4):
    test_image=pd.read_parquet(PATH+'test_image_data_'+str(i)+'.parquet')
    test_image.set_index('image_id',inplace=True)
    x_test=crop_resize_image(test_image).values.reshape(-1,SIZE,SIZE,1)
    preds = model.predict(x_test)
    
    for j, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[j], axis=1)
    
    for k,id in enumerate(test_image.index.values):
        row_id+=[id+'_grapheme_root',id+'_vowel_diacritic',id+'_consonant_diacritic']
        target+=[preds_dict['grapheme_root'][k],preds_dict['vowel_diacritic'][k],preds_dict['consonant_diacritic'][k]]

submission = pd.DataFrame({'row_id': row_id, 'target': target})
submission.to_csv('submission.csv', index=False)
print(submission)
temp = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}
preds_dict = {}
preds_dict['grapheme_root']=[]
preds_dict['consonant_diacritic']=[]
preds_dict['vowel_diacritic']=[]
y_test=pd.read_csv(PATH+'test.csv')
for i in range(4):
    test_image=pd.read_parquet(PATH+'test_image_data_'+str(i)+'.parquet')
    test_image.set_index('image_id',inplace=True)
    x_test=crop_resize_image(test_image).values.reshape(-1,SIZE,SIZE,1)
    preds = model.predict(x_test)
    for j, p in enumerate(temp):
        temp[p] = np.argmax(preds[j], axis=1)
    preds_dict['grapheme_root'].extend(temp['grapheme_root'])
    preds_dict['vowel_diacritic'].extend(temp['vowel_diacritic'])
    preds_dict['consonant_diacritic'].extend(temp['consonant_diacritic'])
    #print(len(preds_dict['grapheme_root']))
root_acc=accuracy_score(y_test['grapheme_root'],preds_dict['grapheme_root'])
vowel_acc=accuracy_score(y_test['vowel_diacritic'],preds_dict['vowel_diacritic'])
consonant_acc=accuracy_score(y_test['consonant_diacritic'],preds_dict['consonant_diacritic'])

print('Grapheme Roots accuracy: ',root_acc)
print('Vowel Diacritic accuracy: ',vowel_acc)
print('consonant Diacritic accuracy: ',consonant_acc)

acc=[root_acc,vowel_acc,consonant_acc]
score=np.average(acc,weights=[2,1,1])
print('Score: ',score)