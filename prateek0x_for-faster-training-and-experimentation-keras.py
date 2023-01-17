!pip install -q efficientnet >> /dev/null
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support
import cv2
import matplotlib.pyplot as plt
import keras
import pydicom
from tqdm import tqdm

import efficientnet.tfkeras as efn
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
train= pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv",nrows=50000)
test= pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

df = pd.DataFrame()
prefix = 'input/rsna-str-pulmonary-embolism-detection/train'
df['image_path'] = '../'+prefix+'/'+train['StudyInstanceUID']+'/'+ train['SeriesInstanceUID'] + '/' + train['SOPInstanceUID']+'.dcm'
df['negative_exam_for_pe'] = train['negative_exam_for_pe']
df['rv_lv_ratio_gte_1'] = train['rv_lv_ratio_gte_1']
df['rv_lv_ratio_lt_1'] = train['rv_lv_ratio_lt_1']
df['leftsided_pe'] = train['leftsided_pe']
df['chronic_pe'] = train['chronic_pe']
df['rightsided_pe'] = train['rightsided_pe']
df['acute_and_chronic_pe'] = train['acute_and_chronic_pe']
df['central_pe'] = train['central_pe']
df['indeterminate'] = train['indeterminate']

del train

test_df = pd.DataFrame()
test_prefix = 'input/rsna-str-pulmonary-embolism-detection/train'
test_df['image_path'] = '../'+test_prefix+'/'+test['StudyInstanceUID']+'/'+ test['SeriesInstanceUID'] + '/' + test['SOPInstanceUID']+'.dcm'
del test
class DataGenerator(keras.utils.Sequence):
    def __init__(self,image_paths,labels,isTrain,batch_size=32,image_dimension=(512,512,3),shuffle=False):
        self.reader = vtk.vtkDICOMImageReader()
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.isTrain = isTrain
        self.image_dimension = image_dimension
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size ))
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def get_img(self,path):
        self.reader.SetFileName(path)
        self.reader.Update()
        _extent = self.reader.GetDataExtent()
        ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

        ConstPixelSpacing = self.reader.GetPixelSpacing()
        imageData = self.reader.GetOutput()
        pointData = imageData.GetPointData()
        arrayData = pointData.GetArray(0)
        ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
        ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
        ArrayDicom = cv2.resize(ArrayDicom,(96,96))
        
        return ArrayDicom
    
    def convert_to_rgb(self,array):
        array = array.reshape((96, 96,))
        return np.stack([array, array, array], axis=1).reshape((96, 96, 3))


    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
            
        batch_y = np.array([self.labels.iloc[k] for k in indexes])
        if self.isTrain:
            batch_x = [self.convert_to_rgb(self.get_img(self.image_paths.iloc[k])) for k in indexes]
        
        #batch_x = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        #batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        
        if self.isTrain:
            return np.stack([x for x in batch_x], axis=0), np.array(batch_y)  
        else:
            return np.stack([x for x in batch_x], axis=0)
        
def get_model():
    inp = tf.keras.layers.Input(shape=(96,96,3))
    base = efn.EfficientNetB0(input_shape=(96,96,3),weights='imagenet',include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    nepe = tf.keras.layers.Dense(1, activation='sigmoid', name='negative_exam_for_pe')(x)
    rlrg1 = tf.keras.layers.Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(x)
    rlrl1 = tf.keras.layers.Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(x) 
    lspe = tf.keras.layers.Dense(1, activation='sigmoid', name='leftsided_pe')(x)
    cpe = tf.keras.layers.Dense(1, activation='sigmoid', name='chronic_pe')(x)
    rspe = tf.keras.layers.Dense(1, activation='sigmoid', name='rightsided_pe')(x)
    aacpe = tf.keras.layers.Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(x)
    cnpe = tf.keras.layers.Dense(1, activation='sigmoid', name='central_pe')(x)
    indt = tf.keras.layers.Dense(1, activation='sigmoid', name='indeterminate')(x)
    
    model = tf.keras.Model(inputs=inp, outputs={'negative_exam_for_pe':nepe,
                                      'rv_lv_ratio_gte_1':rlrg1,
                                      'rv_lv_ratio_lt_1':rlrl1,
                                      'leftsided_pe':lspe,
                                      'chronic_pe':cpe,
                                      'rightsided_pe':rspe,
                                      'acute_and_chronic_pe':aacpe,
                                      'central_pe':cnpe,
                                      'indeterminate':indt})

    
    return model
test_gen = DataGenerator(           test_df['image_path'],
                                    labels = None,
                                    isTrain=False,
                                    batch_size=500,
                                    image_dimension=(96,96,3),
                                    shuffle=True
                                    ) 

def get_generators(train_idx,val_idx):
    #self,dataset,image_paths,labels,batch_size=32,image_dimension=(512,512,3),shuffle=False
    
    train_datagen = DataGenerator(  
                                    df['image_path'].iloc[train_idx],
                                    labels =df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe', 'rightsided_pe','acute_and_chronic_pe', 'central_pe', 'indeterminate']].iloc[train_idx],
                                    isTrain=True,
                                    batch_size=500,
                                    image_dimension=(96,96,3),
                                    shuffle=True)
    val_datagen=DataGenerator(      
                                    df['image_path'].iloc[val_idx],
                                    labels = df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe', 'rightsided_pe','acute_and_chronic_pe', 'central_pe', 'indeterminate']].iloc[val_idx],
                                    isTrain=True,
                                    batch_size=100,
                                    image_dimension=(96,96,3),
                                    shuffle=True)
    
    return train_datagen,val_datagen
gkf = KFold(n_splits = 2)

acc_score = []
plots = []

predictions = {}

for i,(train_idx, val_idx) in enumerate(gkf.split(df)):
    
    model = get_model()
    model.compile(optimizer="SGD",
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    print("--- Fold :: ",i," --- ")
    
    train_gen,val_gen = get_generators(train_idx,val_idx)

    history = model.fit_generator(train_gen,steps_per_epoch=1000,epochs = 5)#,validation_data=val_gen,validation_steps=5)
    
    print("\n\n ----- Prediction on Test -------------")
    
    predicts = model.predict_genertator(test_gen,steps=294)
    
    try:
        for key in predicts.keys():
            predictions[key] += predicts[key].flatten().tolist()
            
    except Exception as e:
        print(e)
        for key in predicts.keys():
            predictions[key] = predicts[key].flatten().tolist()
    
for x,y in train_gen:
    img = x[0]
    plt.imshow(img)
    break