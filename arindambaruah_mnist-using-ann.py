import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt
train_df=pd.read_csv('../input/digit-recognizer/train.csv')

train_df.head()
test_df=pd.read_csv('../input/digit-recognizer/test.csv')

test_df.head()
temp_df=train_df.copy()

temp_df.drop('label',axis=1,inplace=True)

temp_df=np.array(temp_df).reshape(-1,28,28,1)

temp_df.shape
labels=train_df['label']

plt.imshow(temp_df[9][:,:,0])

print('The number is:{}'.format(labels[9]))
unscaled_inputs=train_df.iloc[:,1:].values
targets=train_df.iloc[:,0].values
scaled_inputs=preprocessing.scale(unscaled_inputs)
total_indices=scaled_inputs.shape[0]
print('Total amount of data in the training dataset: {}'.format(total_indices))
shuffled_indices=np.arange(total_indices)
np.random.shuffle(shuffled_indices)
shuffled_indices
shuffled_inputs=scaled_inputs[shuffled_indices]

shuffled_targets=targets[shuffled_indices]
samples_count=total_indices



train_samples_count=int(0.8*samples_count)

validation_samples_count=int(0.1*samples_count)

test_samples_count=samples_count-train_samples_count-validation_samples_count
train_inputs=shuffled_inputs[:train_samples_count]

train_targets=shuffled_targets[:train_samples_count]



validation_inputs=shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]

validation_targets=shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]



test_inputs=shuffled_inputs[train_samples_count+validation_samples_count:]

test_targets=shuffled_targets[train_samples_count+validation_samples_count:]
np.savez('MNIST_train',inputs=train_inputs,target=train_targets)

np.savez('MNIST_validation',inputs=validation_inputs,target=validation_targets)

np.savez('MNIST_test',inputs=test_inputs,target=test_targets)
npz=np.load('MNIST_train.npz')

train_inputs=npz['inputs'].astype(np.float)

train_targets=npz['target'].astype(np.int)
npz=np.load('MNIST_test.npz')

test_inputs=npz['inputs'].astype(np.float)

test_targets=npz['target'].astype(np.int)
npz=np.load('MNIST_validation.npz')

validation_inputs=npz['inputs'].astype(np.float)

validation_targets=npz['target'].astype(np.int)
input_size=784

output_size=10

hidden_layer_size=50



model=tf.keras.Sequential([

    #Input layer

    tf.keras.layers.Dense(input_size),

    

    #Hidden layer 1

    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),

    #Hidden layer 2

    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),

    #Hidden layer 3

    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),

    

    #Output layer

    tf.keras.layers.Dense(output_size,activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
NUM_EPOCHS=50

BATCH_SIZE=100



early_stopping=tf.keras.callbacks.EarlyStopping(patience=20)



model.fit(train_inputs,train_targets,

          batch_size=BATCH_SIZE,

          epochs=NUM_EPOCHS,

          callbacks=[early_stopping],

          validation_data=(validation_inputs,validation_targets),

          verbose=2,validation_steps=10)
test_loss,test_accuracy=model.evaluate(test_inputs,test_targets)
print('\n Test loss:{0:.2f} Test accuracy: {1:.2f} %'.format(test_loss,test_accuracy*100))
values=model.predict(test_inputs)
pd.DataFrame(values).head()
test_df.head()
unscaled_inputs_test=test_df.values

scaled_inputs=preprocessing.scale(unscaled_inputs_test)
test_inputs=scaled_inputs
test_inputs.shape
np.savez('Final_test',inputs=test_inputs)
npz_test=np.load('Final_test.npz')

test_inputs=npz_test['inputs'].astype(float)
values_df=pd.DataFrame(model.predict(test_inputs))
values_df.head()
values_df=values_df[values_df>0.5]
values_df
values_df[values_df>0.5]=1

values_df
values_df.fillna(0,inplace=True)

values_df.head()
values_df.size
predictions_df=pd.DataFrame(values_df[values_df==1].stack())

predictions_df
predictions_df[0].isna().value_counts()
predictions_df.shape[0]
predictions_df.drop(0,axis=1,inplace=True)
predictions_df
predictions_df.reset_index(inplace=True)
predictions_df.rename(columns={'level_1':'Label'},inplace=True)

predictions_df.head()
image_id=pd.DataFrame(np.arange(0,28000),columns=['ImageId'])

image_id['ID']=image_id['ImageId']

image_id.head()


predictions_df.rename(columns={'level_0':'ImageId'},inplace=True)
predictions_df.head()
final_preds=predictions_df.copy()
final_preds=pd.merge_ordered(final_preds,image_id,on='ImageId',fill_method='None')
final_preds['Label'].isna().value_counts()
final_preds['Label'].value_counts()
final_preds['Label'].fillna(final_preds['Label'].mode()[0],inplace=True)
final_preds=final_preds.astype(int)
final_preds['ImageId']=final_preds['ImageId']+1
final_preds.drop('ID',axis=1,inplace=True)

final_preds.head()
final_preds['Label'].unique()
sample_sub=pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sample_sub.dtypes
final_preds['ImageId']=sample_sub['ImageId']
final_preds.to_csv('Final_submission.csv',index=False)