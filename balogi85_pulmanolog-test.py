import os

import cv2

import pydicom

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation,Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D,LeakyReLU, ReLU, Concatenate, GaussianNoise,MaxPooling2D

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adadelta, Adamax

from tensorflow.keras.applications import VGG19

from sklearn.model_selection import train_test_split
path = '../input/osic-pulmonary-fibrosis-progression/'
items = os.listdir(path)

print(items)
train_df = pd.read_csv(path + 'train.csv')
train_df.head()
train_df.SmokingStatus.unique()
def changing_columns(df):

    age_sex_smoke = [(df.Age.values[0]-30) / 30]

    

    if df['Sex'].values[0] == 'Male':

        age_sex_smoke.append(0)

    else:

        age_sex_smoke.append(1)

    

    if df['SmokingStatus'].values[0]== 'Never smoked':

        age_sex_smoke.extend([0,0])

    elif df['SmokingStatus'].values[0]== 'Ex-smoker':

        age_sex_smoke.extend([1,1])

    elif df['SmokingStatus'].values[0]== 'Currently smokes':

        age_sex_smoke.extend([0,1])

    else :

        age_sex_smoke.extend([1,0])

    

    return np.array(age_sex_smoke)   
changing_columns(train_df)
A = {}

T = {}

P = []



for i,p in enumerate(train_df['Patient'].unique()):

    sub = train_df.loc[train_df.Patient == p, :]

   # print(sub)

    fvc = sub['FVC'].values

   # print(fvc)

    weeks = sub['Weeks'].values

   # print(weeks)

    ver_stack= np.vstack([weeks,np.ones(len(weeks))]).T

   # print(ver_stack)

    a,b = np.linalg.lstsq(ver_stack, fvc)[0]

   # print(a)

   # print(b)

    

    A[p] = a

    T[p] = changing_columns(sub)

    P.append(p)
BATCH_S = 32

shape = 299
def get_im_from_dicom(path):

    d = pydicom.dcmread(path)

    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))
class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = BATCH_S

        

        self.train_data = {}

        for p in train_df.Patient.unique():

            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            numb = [float(i[:-4]) for i in ldir]

            self.train_data[p] = [i for i in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/') if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15]

    

    def __len__(self):

        return 1000

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_im_from_dicom(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                mask = cv2.resize(cv2.imread(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear/{k}/{i[:-4]}.jpg', 0), (512, 512))> 0

                img[~mask] = 0

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print('ooooo, we have a problem')

                print(k)

                print(i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a

    
def the_model(input_shape=(shape,shape,1)):

   

    vgg = VGG19(include_top = False, weights= None, input_shape= input_shape)

   

    inp_1 = Input(shape = input_shape)

    

    half_1 = vgg(inp_1)

    half_1 = GlobalAveragePooling2D()(half_1)

    

    inp_2 = Input(shape=(4,))

    half_2 = GaussianNoise(0.3)(inp_2)

    

    whole = Concatenate()([half_1, half_2])

    whole = Dense(100, activation='relu')(whole)

    whole = Dense(100, activation='relu')(whole)

    whole = Dropout(0.2)(whole)

    whole = Dense(10, activation='relu')(whole)

    whole = Dropout(0.2)(whole)

    whole = Dense(1)(whole)

    

    return Model([inp_1,inp_2], whole)
model = the_model()
train_part, valid_part = train_test_split(P, shuffle= True, train_size= 0.9)
train_generator = IGenerator(keys= train_part, a=A, tab=T)

valid_generator = IGenerator(keys= valid_part, a=A, tab=T)
s_p_e = 200

epochs = 25

learn_r = 0.001


early_stopp = tf.keras.callbacks.EarlyStopping(monitor="val_loss",

                                               min_delta=1e-3,patience=10,

                                               verbose=1,mode="auto",

                                               baseline=None,

                                               restore_best_weights=True)



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 

                                                 factor=0.2,

                                                 patience=5, min_lr=0.0001)



opt_1 = SGD(learning_rate=learn_r, momentum=0.9)



opt_2 = Adam(learning_rate= learn_r)



opt_3 = Nadam(learning_rate= learn_r)



opt_4 = Adamax(learning_rate= learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

model.compile(optimizer= opt_4, loss= 'mae')
hist = model.fit_generator(train_generator, steps_per_epoch=s_p_e, validation_data= valid_generator,

                   validation_steps=20, callbacks= [reduce_lr, early_stopp], epochs= epochs, workers= 4)
def score(fvc_true, fvc_pred, sigma):    

    sigma_clip = np.maximum(sigma, 70)

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)


metric = []

for q in range(1, 10):

    m = []

    for p in valid_part:

        x = [] 

        tab = [] 

        

        if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

            continue

            

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                x.append(get_im_from_dicom(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

                tab.append(changing_columns(train_df.loc[train_df.Patient == p, :])) 

        if len(x) < 1:

            continue

        tab = np.array(tab) 

    

        x = np.expand_dims(x, axis=-1)

        _a = model.predict([x, tab])

        a = np.quantile(_a, q / 10)

        

        percent_true = train_df.Percent.values[train_df.Patient == p]

        fvc_true = train_df.FVC.values[train_df.Patient == p]

        weeks_true = train_df.Weeks.values[train_df.Patient == p]

        

        fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

        percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

        m.append(score(fvc_true, fvc, percent))

    print(np.mean(m))

    metric.append(np.mean(m))
q = (np.argmin(metric) + 1)/ 10

print(q)
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

sub.head() 
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

test.head()
A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 

STD, WEEK = {}, {} 

for p in test.Patient.unique():

    x = [] 

    tab = [] 

    ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

    for i in ldir:

        if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

            x.append(get_im_from_dicom(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

            tab.append(changing_columns(test.loc[test.Patient == p, :])) 

    if len(x) <= 1:

        continue

    tab = np.array(tab) 

            

    x = np.expand_dims(x, axis=-1) 

    _a = model.predict([x, tab]) 

    a = np.quantile(_a, q)

    A_test[p] = a

    B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

    P_test[p] = test.Percent.values[test.Patient == p] 

    WEEK[p] = test.Weeks.values[test.Patient == p]

for k in sub.Patient_Week.values:

    p, w = k.split('_')

    w = int(w) 

    

    fvc = A_test[p] * w + B_test[p]

    sub.loc[sub.Patient_Week == k, 'FVC'] = fvc

    sub.loc[sub.Patient_Week == k, 'Confidence'] = (

        P_test[p] - A_test[p] * abs(WEEK[p] - w)

)

    
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)