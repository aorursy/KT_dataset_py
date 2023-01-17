import tensorflow as tf

import numpy as np

import pandas as pd

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from tensorflow.keras import layers

from steelcomputility import *
#model_loading stuff

class FixedDropout(layers.Dropout):

    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:

            return self.noise_shape



        symbolic_shape = K.shape(inputs)

        noise_shape = [symbolic_shape[axis] if shape is None else shape

                       for axis, shape in enumerate(self.noise_shape)]

        return tuple(noise_shape)
name_list = os.listdir('../input/severstal-steel-defect-detection/test_images/')

result_df = pd.DataFrame({'defectYN':False,'df1':'','df2':'','df3':'','df4':''},index=name_list)
g1=tf.Graph()

#classification model

with g1.as_default():

    model=tf.keras.models.load_model('../input/modelupload5/20191214effb0_classification_4outs.h5',compile=False,custom_objects={'swish':tf.compat.v2.nn.swish,'FixedDropout':FixedDropout})

#    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.3)

    #model.compile(optimizer='SGD', loss=loss, metrics=['accuracy'])

    #model.load_weights('../input/saved-model/defect_classification_1129.h5')

#    result=model.predict(pr_dataset)

    def split_input(x):

        x1=x[:,:,0:416,:]

        x2=x[:,:,400:816,:]

        x3=x[:,:,800:1216,:]

        x4=x[:,:,1184:1600,:]

        return [x1,x2,x3,x4]

    def merge_result(xlist):

        x=K.stack(xlist,axis=-1)

        return K.max(x,axis=-1)

    Inpt=layers.Input(shape=(256,1600,3))

    clip1,clip2,clip3,clip4=layers.Lambda(split_input)(Inpt)

    o1=model(clip1)

    o2=model(clip2)

    o3=model(clip3)

    o4=model(clip4)

    out=layers.Lambda(merge_result)([o1,o2,o3,o4])

    model2=Model(inputs=Inpt, outputs=out)

#    model2.compile(optimizer='SGD', loss=loss, 

#    metrics=[Precision(thresholds=0.5), Recall(thresholds=0.5), BinaryAccuracy(threshold=0.5)])
#checkdf = result_df = result_df.iloc[100:120]

classification_threshold=[0.8,0.8,0.9,0.8]

with g1.as_default():

    pr_dataset = tf.data.Dataset.from_generator(test_data_gen(result_df),output_types=(tf.string,tf.uint8),output_shapes=((),(256,1600,3))).map(lambda x,y:y).batch(16)

    cls_defect=model2.predict(pr_dataset)

pred = cls_defect>classification_threshold

result_df['defectYN1']=pred[:,0]

result_df['defectYN2']=pred[:,1]

result_df['defectYN3']=pred[:,2]

result_df['defectYN4']=pred[:,3]

result_df['defectYN']=pred.any(axis=-1)
seg_df = result_df[result_df.defectYN==True]

g2=tf.Graph()

thresholds=np.array([0.5,0.5,0.2,0.4]).reshape(1,1,-1)

discard=[800,1000,1900,5000]

with g2.as_default():

    segmodel=tf.keras.models.load_model('../input/modelupload5/20191210segmodel_no_opt2.h5',compile=False,custom_objects={'swish':tf.compat.v2.nn.swish,'FixedDropout':FixedDropout})

with g2.as_default():

#    seg_dataset = tf.data.Dataset.from_generator(test_data_gen(seg_df),output_types=(tf.string,tf.uint8),output_shapes=((),(256,1600,3))).batch(32)

    for idx, img in test_data_gen(seg_df)():

        img=img[np.newaxis,...]

        out =segmodel.predict(img)

        result = (out.squeeze()>thresholds).astype('int8')

        result = remove_small(result,discard)

        result_df.loc[idx,'df1'],result_df.loc[idx,'df2'],result_df.loc[idx,'df3'],result_df.loc[idx,'df4']=tuple(Msk2Encoding(result[...,i]) for i in range(4))
#post process

for i in range(1,5):

    idx = result_df['defectYN'+str(i)]==False

    result_df.loc[idx,'df'+str(i)]=''
submission_df = pd.DataFrame(None,columns=['EncodedPixels'])

for idx,record in result_df.iterrows():

    for i in range(4):

        submission_df.loc[idx+'_'+str(i+1),'EncodedPixels']=record['df'+str(i+1)]

submission_df.head(10)

submission_df.to_csv('submission.csv',index_label='ImageId_ClassId')