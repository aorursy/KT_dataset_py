import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.metrics import accuracy_score as acc
#loading the training data csv
train=pd.read_csv(r'../input/mnsit-train-and-test-images/train.csv')
train.head()
### imgae data will be in 28*28 size, for giving them as input data it will need to be flattened, thus columns are being created 
cols=[]
for i in range (28*28):
    cols.append('input {}'.format(i))
temp_val=pd.DataFrame(columns=cols)
temp_info=train.copy()
for i in range(train.shape[0]):   
    path=r"../input/mnsit-train-and-test-images/Images/train/{0}".format(train.filename[i])
    image_2d=cv2.imread(path,0)
    image_flattened=image_2d.flatten()
    image_scaled=image_flattened/255
    image_scaled=image_scaled.reshape(1,len(image_scaled))
    temp_val=temp_val.append(pd.DataFrame(image_scaled,index=[i],columns=temp_val.columns))
temp_info=pd.concat([temp_info,temp_val],axis=1)
temp_info=temp_info.reset_index(drop=True)
### saving the data extracted in a csv file for future use

#temp_info.to_csv('train_inputs.csv')
train_inputs_final=temp_info.copy()
# train_inputs_final=pd.read_csv('train_inputs.csv')
train_inputs_final.head()
inputs=train_inputs_final.drop(['filename','label'],axis=1)
targets=train_inputs_final.label
train_count=int(0.8*inputs.shape[0])
val_count=int(0.1*inputs.shape[0])

train_inputs=inputs.iloc[:train_count,:]
train_labels=targets.iloc[:train_count]

val_inputs=inputs.iloc[train_count:train_count+val_count,:]
val_labels=targets.iloc[train_count:train_count+val_count]

test_inputs=inputs.iloc[train_count+val_count:,:]
test_labels=targets.iloc[train_count+val_count:]
import tensorflow as tf
input_size=28*28
output_size=10
hls=100
batch_size=100
max_epoch=100
early_stopping= tf.keras.callbacks.EarlyStopping(patience=2)
model=tf.keras.Sequential([
                                        tf.keras.layers.Dense(hls,activation='relu'),
                                        tf.keras.layers.Dense(hls,activation='relu'),
                                        tf.keras.layers.Dense(output_size,activation='softmax'),
                                        ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_inputs,train_labels,batch_size=batch_size,epochs=max_epoch,callbacks=[early_stopping],validation_data=(val_inputs,val_labels),verbose=2)
test_loss,test_accuracy=model.evaluate(test_inputs,test_labels)
print('test loss: {0:.2f}, test accuracy: {1:.2f}%'.format(test_loss,test_accuracy*100))
temp_info=pd.DataFrame(columns=['filename'])
file_count=range(49000,69999+1)
filename=[]
for i in file_count:
    filename.append(str(i)+'.png')
temp_info['filename']=filename
temp_info.head()
temp_val=pd.DataFrame(columns=cols)
train_inputs_final=pd.DataFrame()
for i in range(temp_info.shape[0]):   
    path=r"../input/mnsit-train-and-test-images/Images/test/{0}".format(temp_info.filename[i])
    image_2d=cv2.imread(path,0)
    image_flattened=image_2d.flatten()
    image_scaled=image_flattened/255
    image_scaled=image_scaled.reshape(1,len(image_scaled))
    temp_val=temp_val.append(pd.DataFrame(image_scaled,index=[i],columns=temp_val.columns))

temp_info=pd.concat([temp_info,temp_val],axis=1)
temp_info=temp_info.reset_index(drop=True)
# from tensorflow.keras.models import Sequential, save_model, load_model
# filepath = r"loaction/of/the/save"
# save_model(model, filepath)
y_pred=model.predict(temp_info.drop('filename',axis=1),verbose=1)
pred=np.argmax(y_pred,axis=1)
outputs=pd.DataFrame(columns=['filename','label'])
outputs['filename']=temp_info.filename
outputs['label']=pred
outputs.head()
#saving the data in csv format
# outputs.to_csv('predicted',index=False)